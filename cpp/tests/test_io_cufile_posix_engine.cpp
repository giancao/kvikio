/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <kvikio/io/cufile_posix_engine.hpp>
#include <kvikio/io/engine_registry.hpp>
#include <kvikio/io/errors.hpp>

#include "utils/utils.hpp"

using namespace kvikio::io;
using namespace std::chrono_literals;

namespace {

constexpr std::size_t KiB = 1024;

std::uint8_t pattern(std::size_t offset)
{
  return static_cast<std::uint8_t>((offset * 31 + 7) % 253);
}

std::vector<std::uint8_t> pattern_bytes(std::size_t offset, std::size_t size)
{
  std::vector<std::uint8_t> result(size);
  for (std::size_t i = 0; i < size; ++i) {
    result[i] = pattern(offset + i);
  }
  return result;
}

/// Page-locked-like blocks from the heap, with a fixed number of them.
class test_staging final : public staging_source {
 public:
  test_staging(std::size_t block_size, std::size_t n_blocks) : _block_size(block_size)
  {
    for (std::size_t i = 0; i < n_blocks; ++i) {
      _storage.emplace_back(block_size);
      _free.push_back(reinterpret_cast<std::byte*>(_storage.back().data()));
    }
    _total = n_blocks;
  }

  [[nodiscard]] std::size_t block_size() const noexcept override { return _block_size; }

  [[nodiscard]] staging_result acquire(std::size_t bytes) noexcept override
  {
    auto const need = (bytes + _block_size - 1) / _block_size;
    std::lock_guard lock(_mutex);
    ++acquires;
    staging_result result;
    if (need > _total) { return result; }
    if (need > _free.size()) {
      ++unavailable;
      result.status = staging_status::unavailable;
      return result;
    }
    std::vector<std::byte*> blocks(_free.end() - static_cast<std::ptrdiff_t>(need), _free.end());
    _free.resize(_free.size() - need);
    result.lease  = std::make_unique<lease>(*this, std::move(blocks));
    result.status = staging_status::ok;
    return result;
  }

  [[nodiscard]] std::size_t free_blocks()
  {
    std::lock_guard lock(_mutex);
    return _free.size();
  }

  int acquires{0};
  int unavailable{0};

 private:
  class lease final : public staging_lease {
   public:
    lease(test_staging& owner, std::vector<std::byte*> blocks)
      : _owner(owner), _blocks(std::move(blocks))
    {
    }
    ~lease() override
    {
      std::lock_guard lock(_owner._mutex);
      _owner._free.insert(_owner._free.end(), _blocks.begin(), _blocks.end());
    }
    [[nodiscard]] std::span<std::byte* const> blocks() const noexcept override { return _blocks; }
    [[nodiscard]] std::size_t block_size() const noexcept override { return _owner._block_size; }

   private:
    test_staging& _owner;
    std::vector<std::byte*> _blocks;
  };

  std::size_t _block_size;
  std::size_t _total{0};
  std::mutex _mutex;
  std::vector<std::vector<std::uint8_t>> _storage;
  std::vector<std::byte*> _free;
};

class CufilePosixEngineTest : public testing::Test {
 protected:
  static constexpr std::size_t file_bytes = 1 * KiB * KiB + 123;

  void SetUp() override
  {
    path             = (tmpdir.path() / "data.bin").string();
    auto const bytes = pattern_bytes(0, file_bytes);
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<char const*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.close();
    ASSERT_TRUE(out);
  }

  /// An engine on the POSIX path, so device transfers go through @p staging.
  std::shared_ptr<engine> make(std::shared_ptr<staging_source> staging = nullptr,
                               unsigned int threads                    = 4,
                               kvikio::CompatMode mode                 = kvikio::CompatMode::ON)
  {
    cufile_posix_options options;
    options.staging             = std::move(staging);
    options.num_threads         = threads;
    options.compat_mode         = mode;
    options.max_operation_bytes = 256 * KiB;
    auto result                 = make_cufile_posix_engine(options);
    result->start();
    return result;
  }

  std::vector<std::uint8_t> device_bytes(kvikio::test::DevBuffer<std::uint8_t> const& buffer)
  {
    std::vector<std::uint8_t> host(buffer.nbytes);
    KVIKIO_CHECK_CUDA(cudaMemcpy(host.data(), buffer.ptr, buffer.nbytes, cudaMemcpyDeviceToHost));
    return host;
  }

  kvikio::test::TempDir tmpdir{true};
  std::string path;
};

}  // namespace

TEST_F(CufilePosixEngineTest, capabilities_are_what_the_cache_needs)
{
  auto eng         = make();
  auto const& caps = eng->capabilities();
  EXPECT_EQ(eng->name(), "cufile_posix");
  EXPECT_TRUE(caps.vector_host_read);
  EXPECT_TRUE(caps.host_to_device_read);
  EXPECT_TRUE(caps.fragmented_host_buffer);
  EXPECT_TRUE(caps.device_read);
  EXPECT_TRUE(caps.write);
  EXPECT_TRUE(caps.device_write);
  // What it does not have is declared, not faked.
  EXPECT_FALSE(caps.priority);
  EXPECT_FALSE(caps.prefers_bulk_io);
  EXPECT_TRUE(eng->supports(path));
  EXPECT_FALSE(eng->supports("s3://bucket/key"));
  EXPECT_FALSE(eng->supports(tmpdir.path().string()));
}

TEST_F(CufilePosixEngineTest, is_built_in_and_selectable_without_a_plugin)
{
  engine_registry registry;
  EXPECT_TRUE(registry.has_factory("cufile_posix"));
  auto selected = registry.select(engine_route::local, default_engine_name(engine_route::local));
  EXPECT_EQ(selected->name(), "cufile_posix");
  EXPECT_EQ(registry.selected_name(engine_route::local), "cufile_posix");
}

TEST_F(CufilePosixEngineTest, open_records_size_and_a_stat_identity)
{
  auto eng    = make();
  auto object = eng->open(path, open_options{});
  EXPECT_EQ(object->size(), file_bytes);
  EXPECT_EQ(object->object_path(), path);
  EXPECT_FALSE(object->writable());
  EXPECT_EQ(object->identity().kind, identity_kind::local_stat);
  // A second open of the unchanged file has the same identity.
  EXPECT_EQ(eng->open(path, open_options{})->identity_cache_key(), object->identity_cache_key());
  EXPECT_ANY_THROW((void)eng->open((tmpdir.path() / "missing").string(), open_options{}));
}

TEST_F(CufilePosixEngineTest, synchronous_host_read_clamps_to_the_end)
{
  auto eng    = make();
  auto object = eng->open(path, open_options{});
  std::vector<std::uint8_t> bytes(300);
  EXPECT_EQ(eng->host_read(*object, 1000, 300, bytes.data()), 300u);
  EXPECT_EQ(bytes, pattern_bytes(1000, 300));
  EXPECT_EQ(eng->host_read(*object, file_bytes - 10, 300, bytes.data()), 10u);
  EXPECT_EQ(eng->host_read(*object, file_bytes + 5, 300, bytes.data()), 0u);
}

TEST_F(CufilePosixEngineTest, reads_a_batch_of_host_slices)
{
  auto eng    = make();
  auto object = eng->open(path, open_options{});
  // One slice larger than max_operation_bytes, one small, one crossing the end of the file.
  std::vector<std::uint8_t> a(700 * KiB), b(100), c(4096);
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{10, a.size()}, host_buffer{a.data()});
  slices.emplace_back(range{900 * KiB, b.size()}, host_buffer{b.data()});
  slices.emplace_back(range{file_bytes - 96, c.size()}, host_buffer{c.data()});
  auto future = eng->read(*object, std::move(slices));
  EXPECT_EQ(std::move(future).get(30s), a.size() + b.size() + 96);
  EXPECT_EQ(a, pattern_bytes(10, a.size()));
  EXPECT_EQ(b, pattern_bytes(900 * KiB, b.size()));
  c.resize(96);
  EXPECT_EQ(c, pattern_bytes(file_bytes - 96, 96));
}

// Each fragment is read straight into its own memory and handed back by itself.
TEST_F(CufilePosixEngineTest, fragments_are_filled_in_place_and_handed_back_one_by_one)
{
  auto eng                    = make();
  auto object                 = eng->open(path, open_options{});
  constexpr std::size_t piece = 64 * KiB;
  std::vector<std::uint8_t> memory(4 * piece);
  // Fragments that are neither adjacent in the file nor in file order in memory.
  std::vector<std::size_t> const offsets{512 * KiB, 0, 900 * KiB, 128 * KiB};
  std::vector<fragment> list;
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    list.push_back(fragment{offsets[i], memory.data() + i * piece, piece, &memory[i * piece]});
  }
  std::mutex mutex;
  std::vector<fragment_handle> handed_back;
  std::size_t calls = 0;
  auto completion   = std::make_shared<prepared_io_completion>(
    [&](std::span<fragment_handle const> handles, bool ok) noexcept {
      EXPECT_TRUE(ok);
      std::lock_guard lock(mutex);
      ++calls;
      handed_back.insert(handed_back.end(), handles.begin(), handles.end());
    });
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, 964 * KiB}, host_buffer{std::move(list)});
  slices.back().on_complete = completion;

  EXPECT_EQ(eng->read(*object, std::move(slices)).get(30s), 964 * KiB);
  EXPECT_EQ(calls, offsets.size());
  ASSERT_EQ(handed_back.size(), offsets.size());
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    EXPECT_NE(std::find(handed_back.begin(), handed_back.end(), &memory[i * piece]),
              handed_back.end());
    std::vector<std::uint8_t> const got(
      memory.begin() + static_cast<std::ptrdiff_t>(i * piece),
      memory.begin() + static_cast<std::ptrdiff_t>((i + 1) * piece));
    EXPECT_EQ(got, pattern_bytes(offsets[i], piece)) << "fragment " << i;
  }
}

TEST_F(CufilePosixEngineTest, device_read_stages_through_the_injected_source)
{
  auto staging = std::make_shared<test_staging>(128 * KiB, 8);
  auto eng     = make(staging);
  auto object  = eng->open(path, open_options{});
  kvikio::test::DevBuffer<std::uint8_t> device(600 * KiB);
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{1000, device.nbytes},
                      device_buffer{static_cast<std::uint8_t*>(device.ptr), nullptr});
  EXPECT_EQ(eng->read(*object, std::move(slices)).get(30s), device.nbytes);
  // The bytes are in place when the future settles: no stream synchronization here.
  EXPECT_EQ(device_bytes(device), pattern_bytes(1000, device.nbytes));
  EXPECT_GT(staging->acquires, 0);
  EXPECT_EQ(staging->free_blocks(), 8u) << "every lease was returned";
}

TEST_F(CufilePosixEngineTest, device_read_waits_for_staging_instead_of_failing)
{
  // Two blocks, four threads, a read cut into several operations of two blocks each:
  // only one operation can hold staging at a time, the others wait their turn.
  auto staging = std::make_shared<test_staging>(128 * KiB, 2);
  auto eng     = make(staging, 4);
  auto object  = eng->open(path, open_options{});
  kvikio::test::DevBuffer<std::uint8_t> device(1000 * KiB);
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, device.nbytes},
                      device_buffer{static_cast<std::uint8_t*>(device.ptr), nullptr});
  EXPECT_EQ(eng->read(*object, std::move(slices)).get(60s), device.nbytes);
  EXPECT_EQ(device_bytes(device), pattern_bytes(0, device.nbytes));
  EXPECT_EQ(staging->free_blocks(), 2u);
}

TEST_F(CufilePosixEngineTest, device_read_without_any_staging_fails_cleanly)
{
  auto eng    = make(nullptr);
  auto object = eng->open(path, open_options{});
  kvikio::test::DevBuffer<std::uint8_t> device(4096);
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, device.nbytes},
                      device_buffer{static_cast<std::uint8_t*>(device.ptr), nullptr});
  EXPECT_ANY_THROW(eng->read(*object, std::move(slices)).get(30s));

  // A source that can never serve the request fails it too, rather than parking forever.
  auto tiny = make(std::make_shared<test_staging>(64 * KiB, 1));
  auto obj2 = tiny->open(path, open_options{});
  std::vector<prepared_io_slice> again;
  again.emplace_back(range{0, device.nbytes},
                     device_buffer{static_cast<std::uint8_t*>(device.ptr), nullptr});
  EXPECT_EQ(tiny->read(*obj2, std::move(again)).get(30s), device.nbytes);
}

// A slice with a host and a device destination: the bytes land in the host
// fragment and are copied to the device from there.
TEST_F(CufilePosixEngineTest, mixed_slice_fills_the_fragment_and_the_device)
{
  auto eng                    = make();
  auto object                 = eng->open(path, open_options{});
  constexpr std::size_t chunk = 64 * KiB;
  // The fragment covers a whole chunk; the caller wants the middle of it on the device.
  std::vector<std::uint8_t> host(chunk);
  constexpr std::size_t want_offset = 2 * chunk + 1000;
  constexpr std::size_t want_size   = 5000;
  kvikio::test::DevBuffer<std::uint8_t> device(want_size);
  bool ok_seen    = false;
  auto completion = std::make_shared<prepared_io_completion>(
    [&ok_seen](std::span<fragment_handle const> handles, bool ok) noexcept {
      ok_seen = ok && handles.size() == 1;
    });
  std::vector<fragment> list{fragment{2 * chunk, host.data(), chunk, host.data()}};
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{want_offset, want_size},
                      host_buffer{std::move(list)},
                      device_buffer{static_cast<std::uint8_t*>(device.ptr), nullptr});
  slices.back().on_complete = completion;
  // The returned count is the caller's range, not the widened fragment.
  EXPECT_EQ(eng->read(*object, std::move(slices)).get(30s), want_size);
  EXPECT_TRUE(ok_seen);
  EXPECT_EQ(host, pattern_bytes(2 * chunk, chunk));
  EXPECT_EQ(device_bytes(device), pattern_bytes(want_offset, want_size));
}

TEST_F(CufilePosixEngineTest, writes_host_and_device_sources_and_reads_them_back)
{
  auto staging = std::make_shared<test_staging>(128 * KiB, 4);
  auto eng     = make(staging);
  auto target  = (tmpdir.path() / "out.bin").string();
  auto object  = eng->open_for_write(target, write_open_options{});
  EXPECT_TRUE(object->writable());
  EXPECT_EQ(object->size(), 0u);

  auto const first  = pattern_bytes(0, 300 * KiB);
  auto const second = pattern_bytes(300 * KiB, 200 * KiB);
  kvikio::test::DevBuffer<std::uint8_t> device(second);
  std::vector<prepared_io_slice> slices;
  slices.push_back(make_host_write_slice(range{0, first.size()}, first.data()));
  slices.push_back(make_device_write_slice(
    range{first.size(), second.size()}, static_cast<std::uint8_t const*>(device.ptr), nullptr));
  EXPECT_EQ(eng->write(*object, std::move(slices)).get(30s), first.size() + second.size());
  EXPECT_EQ(object->size(), first.size() + second.size());
  eng->flush_async(*object, io_priority::automatic).get(30s);
  eng->flush(*object);

  std::vector<std::uint8_t> back(first.size() + second.size());
  auto reader = eng->open(target, open_options{});
  EXPECT_EQ(eng->host_read(*reader, 0, back.size(), back.data()), back.size());
  EXPECT_EQ(back, pattern_bytes(0, back.size()));

  // Synchronous write, and the other open modes.
  std::uint8_t const tail[4] = {1, 2, 3, 4};
  EXPECT_EQ(eng->host_write(*object, back.size(), 4, tail), 4u);
  EXPECT_EQ(object->size(), back.size() + 4);
  write_open_options keep;
  keep.mode = write_mode::create_or_open;
  EXPECT_EQ(eng->open_for_write(target, keep)->object_path(), target);
  write_open_options fresh;
  fresh.mode = write_mode::create_new;
  EXPECT_THROW((void)eng->open_for_write(target, fresh), std::system_error);
  write_open_options existing;
  existing.mode = write_mode::open_existing;
  EXPECT_THROW((void)eng->open_for_write((tmpdir.path() / "nope").string(), existing),
               std::system_error);
}

TEST_F(CufilePosixEngineTest, misuse_fails_the_future_and_gives_fragments_back)
{
  auto eng    = make();
  auto other  = make();
  auto object = eng->open(path, open_options{});
  std::uint8_t byte{};

  // Writing through a read-only object.
  std::vector<prepared_io_slice> write;
  write.push_back(make_host_write_slice(range{0, 1}, &byte));
  EXPECT_THROW(eng->write(*object, std::move(write)).get(30s), std::invalid_argument);
  EXPECT_THROW(eng->flush(*object), std::invalid_argument);

  // The per-thread default stream cannot be honoured from a worker thread.
  kvikio::test::DevBuffer<std::uint8_t> device(16);
  std::vector<prepared_io_slice> ptds;
  ptds.emplace_back(range{0, 16},
                    device_buffer{static_cast<std::uint8_t*>(device.ptr), CU_STREAM_PER_THREAD});
  EXPECT_THROW(eng->read(*object, std::move(ptds)).get(30s), std::invalid_argument);

  // A failed submission still settles the completion of a fragmented slice.
  int failures    = 0;
  auto completion = std::make_shared<prepared_io_completion>(
    [&failures](std::span<fragment_handle const>, bool ok) noexcept { failures += ok ? 0 : 1; });
  std::vector<fragment> list{fragment{0, nullptr, 16, &byte}};  // no memory: invalid
  std::vector<prepared_io_slice> bad;
  bad.emplace_back(range{0, 16}, host_buffer{std::move(list)});
  bad.back().on_complete = completion;
  EXPECT_THROW(eng->read(*object, std::move(bad)).get(30s), std::invalid_argument);
  EXPECT_EQ(failures, 1);

  // After shutdown nothing is accepted, and shutting down twice is harmless.
  other->shutdown();
  other->shutdown();
  std::vector<prepared_io_slice> late;
  late.emplace_back(range{0, 1}, host_buffer{&byte});
  EXPECT_THROW(other->read(*object, std::move(late)).get(30s), std::system_error);
}

TEST_F(CufilePosixEngineTest, empty_and_out_of_range_reads_settle_with_zero)
{
  auto eng    = make();
  auto object = eng->open(path, open_options{});
  EXPECT_EQ(eng->read(*object, {}).get(30s), 0u);
  std::uint8_t byte{};
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{file_bytes + 100, 1}, host_buffer{&byte});
  EXPECT_EQ(eng->read(*object, std::move(slices)).get(30s), 0u);
}

TEST_F(CufilePosixEngineTest, align_and_coalesce_merges_touching_ranges_and_drops_invalid_ones)
{
  auto eng = make();
  std::vector<byte_range> const ranges{{5000, 100}, {10, 20}, {4096, 904}, {-5, 10}, {7, 0}};
  // Aligned to pages, the three valid ranges become two adjacent pages, which are one range.
  auto const merged = eng->align_and_coalesce(ranges, 4096);
  ASSERT_EQ(merged.size(), 1u);
  EXPECT_EQ(merged[0].offset(), 0);
  EXPECT_EQ(merged[0].size(), 8192);
  auto const exact = eng->align_and_coalesce(ranges, std::nullopt);
  ASSERT_EQ(exact.size(), 2u);
  EXPECT_EQ(exact[1].offset(), 4096);
  EXPECT_EQ(exact[1].size(), 1004);
}
