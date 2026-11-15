/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The `uring` engine, loaded as a plugin by name. Nothing here links the plugin: the test uses
// only the engine interface and the header-only diagnostics interface.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kvikio/io/engine_registry.hpp>
#include <kvikio/io/errors.hpp>
#include <kvikio/io/uring/uring_engine.hpp>

#include "utils/test_staging.hpp"
#include "utils/utils.hpp"

using namespace kvikio::io;
using namespace std::chrono_literals;
using kvikio::test::test_staging;
using ::testing::HasSubstr;

namespace {

constexpr std::size_t KiB = 1024;
constexpr std::size_t MiB = 1024 * KiB;

std::uint8_t pattern(std::size_t offset)
{
  return static_cast<std::uint8_t>((offset * 131 + 17) % 251);
}

std::vector<std::uint8_t> pattern_bytes(std::size_t offset, std::size_t size)
{
  std::vector<std::uint8_t> result(size);
  for (std::size_t i = 0; i < size; ++i) {
    result[i] = pattern(offset + i);
  }
  return result;
}

class UringEngineTest : public testing::Test {
 protected:
  static constexpr std::size_t file_bytes = 6 * MiB + 321;
  static constexpr std::size_t block_size = 1 * MiB;

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

  /// Select `uring` in a registry of this test. Each reactor leases 64 MiB of staging.
  std::shared_ptr<engine> select(std::size_t n_reactors = 2)
  {
    uring::uring_options options;
    options.n_reactors = n_reactors;
    options.staging    = std::make_shared<test_staging>(block_size, 64 * n_reactors);
    try {
      return registry.select(engine_route::local, uring::uring_engine_name, options);
    } catch (engine_selection_error const& e) {
      // A sandbox may forbid io_uring; the engine then cannot start here.
      skipped = e.what();
      return nullptr;
    }
  }

  std::vector<std::uint8_t> device_bytes(kvikio::test::DevBuffer<std::uint8_t> const& buffer)
  {
    std::vector<std::uint8_t> host(buffer.nbytes);
    KVIKIO_CHECK_CUDA(cudaMemcpy(host.data(), buffer.ptr, buffer.nbytes, cudaMemcpyDeviceToHost));
    return host;
  }

  kvikio::test::TempDir tmpdir{true};
  std::string path;
  std::string skipped;
  engine_registry registry;
};

#define SELECT_OR_SKIP(var, ...)                                               \
  auto var = select(__VA_ARGS__);                                              \
  if (var == nullptr) { GTEST_SKIP() << "uring is unavailable: " << skipped; } \
  static_cast<void>(0)

}  // namespace

TEST_F(UringEngineTest, is_a_plugin_found_next_to_the_library)
{
  EXPECT_FALSE(registry.has_factory("uring")) << "uring is not built into libkvikio.so";
  SELECT_OR_SKIP(eng);
  EXPECT_TRUE(registry.has_factory("uring"));
  EXPECT_EQ(eng->name(), "uring");
  EXPECT_EQ(registry.selected_name(engine_route::local), "uring");

  auto const& caps = eng->capabilities();
  EXPECT_TRUE(caps.device_read);
  EXPECT_TRUE(caps.host_to_device_read);
  EXPECT_TRUE(caps.vector_host_read);
  EXPECT_TRUE(caps.device_range_read);
  EXPECT_TRUE(caps.fragmented_host_buffer);
  EXPECT_TRUE(caps.write);
  EXPECT_TRUE(caps.device_write);
  EXPECT_TRUE(caps.priority);
  EXPECT_FALSE(caps.prefers_bulk_io);
  EXPECT_EQ(caps.min_alignment, 4096u);
  EXPECT_EQ(caps.merge_gap, 4096u);
  EXPECT_EQ(caps.staging_block_size, block_size);
  EXPECT_TRUE(eng->supports(path));
  EXPECT_FALSE(eng->supports("s3://bucket/key"));
}

// Selecting uring without staging fails; the selection in effect before stays.
TEST_F(UringEngineTest, a_failed_selection_keeps_the_default_engine)
{
  auto* const fallback =
    registry.select(engine_route::local, default_engine_name(engine_route::local)).get();
  try {
    registry.select(engine_route::local, "uring");  // plain options: no staging source
    FAIL() << "selecting uring without staging should fail";
  } catch (engine_selection_error const& e) {
    EXPECT_THAT(e.what(), HasSubstr("staging"));
    EXPECT_THAT(e.what(), HasSubstr("still using 'cufile_posix'"));
  }
  EXPECT_EQ(registry.current(engine_route::local), fallback);
}

TEST_F(UringEngineTest, reads_host_slices_and_clamps_to_the_end)
{
  SELECT_OR_SKIP(eng);
  auto object = eng->open(path, open_options{});
  EXPECT_EQ(object->size(), file_bytes);

  std::vector<std::uint8_t> a(3 * MiB), b(777), c(8192);
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{5, a.size()}, host_buffer{a.data()});
  slices.emplace_back(range{4 * MiB + 3, b.size()}, host_buffer{b.data()});
  slices.emplace_back(range{file_bytes - 100, c.size()}, host_buffer{c.data()});
  EXPECT_EQ(eng->read(*object, std::move(slices)).get(30s), a.size() + b.size() + 100);
  EXPECT_EQ(a, pattern_bytes(5, a.size()));
  EXPECT_EQ(b, pattern_bytes(4 * MiB + 3, b.size()));
  c.resize(100);
  EXPECT_EQ(c, pattern_bytes(file_bytes - 100, 100));

  std::vector<std::uint8_t> sync(4096);
  EXPECT_EQ(eng->host_read(*object, 12345, sync.size(), sync.data()), sync.size());
  EXPECT_EQ(sync, pattern_bytes(12345, sync.size()));
}

// Each fragment is filled in place and handed back when its own operation finishes.
TEST_F(UringEngineTest, fragments_are_filled_in_place_and_handed_back)
{
  SELECT_OR_SKIP(eng);
  auto object                       = eng->open(path, open_options{});
  constexpr std::size_t n_fragments = 5;
  // One block per fragment, page-aligned as a cache's blocks are.
  std::unique_ptr<void, void (*)(void*)> memory{std::aligned_alloc(4096, n_fragments * block_size),
                                                &std::free};
  auto* base = static_cast<std::uint8_t*>(memory.get());
  std::vector<fragment> list;
  for (std::size_t i = 0; i < n_fragments; ++i) {
    list.push_back(
      fragment{i * block_size, base + i * block_size, block_size, base + i * block_size});
  }
  std::mutex mutex;
  std::vector<fragment_handle> handed_back;
  auto completion = std::make_shared<prepared_io_completion>(
    [&](std::span<fragment_handle const> handles, bool ok) noexcept {
      EXPECT_TRUE(ok);
      std::lock_guard lock(mutex);
      handed_back.insert(handed_back.end(), handles.begin(), handles.end());
    });
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, n_fragments * block_size}, host_buffer{std::move(list)});
  slices.back().on_complete = completion;

  EXPECT_EQ(eng->read(*object, std::move(slices)).get(30s), n_fragments * block_size);
  ASSERT_EQ(handed_back.size(), n_fragments);
  for (std::size_t i = 0; i < n_fragments; ++i) {
    EXPECT_NE(std::find(handed_back.begin(), handed_back.end(), base + i * block_size),
              handed_back.end());
  }
  std::vector<std::uint8_t> const got(base, base + n_fragments * block_size);
  EXPECT_EQ(got, pattern_bytes(0, n_fragments * block_size));
}

TEST_F(UringEngineTest, reads_into_device_memory_through_staging)
{
  SELECT_OR_SKIP(eng);
  auto object = eng->open(path, open_options{});
  kvikio::test::DevBuffer<std::uint8_t> device(3 * MiB + 5);
  cudaStream_t stream{};
  KVIKIO_CHECK_CUDA(cudaStreamCreate(&stream));
  for (CUstream s : {static_cast<CUstream>(stream), static_cast<CUstream>(nullptr)}) {
    KVIKIO_CHECK_CUDA(cudaMemset(device.ptr, 0, device.nbytes));
    std::vector<prepared_io_slice> slices;
    slices.emplace_back(range{1000, device.nbytes},
                        device_buffer{static_cast<std::uint8_t*>(device.ptr), s});
    EXPECT_EQ(eng->read(*object, std::move(slices)).get(30s), device.nbytes);
    // The bytes are in place when the future settles.
    EXPECT_EQ(device_bytes(device), pattern_bytes(1000, device.nbytes));
  }
  KVIKIO_CHECK_CUDA(cudaStreamDestroy(stream));
}

TEST_F(UringEngineTest, writes_host_and_device_sources_and_reads_them_back)
{
  SELECT_OR_SKIP(eng);
  auto target = (tmpdir.path() / "out.bin").string();
  auto object = eng->open_for_write(target, write_open_options{});
  EXPECT_TRUE(object->writable());

  auto const first  = pattern_bytes(0, 2 * MiB + 100);
  auto const second = pattern_bytes(first.size(), MiB + 7);
  kvikio::test::DevBuffer<std::uint8_t> device(second);
  std::vector<prepared_io_slice> slices;
  slices.push_back(make_host_write_slice(range{0, first.size()}, first.data()));
  slices.push_back(make_device_write_slice(
    range{first.size(), second.size()}, static_cast<std::uint8_t const*>(device.ptr), nullptr));
  EXPECT_EQ(eng->write(*object, std::move(slices)).get(30s), first.size() + second.size());
  EXPECT_EQ(object->size(), first.size() + second.size());
  eng->flush_async(*object, io_priority::automatic).get(30s);

  auto reader = eng->open(target, open_options{});
  std::vector<std::uint8_t> back(first.size() + second.size());
  std::vector<prepared_io_slice> read;
  read.emplace_back(range{0, back.size()}, host_buffer{back.data()});
  EXPECT_EQ(eng->read(*reader, std::move(read)).get(30s), back.size());
  EXPECT_EQ(back, pattern_bytes(0, back.size()));
}

// The diagnostics are reached through a header-only interface.
TEST_F(UringEngineTest, exposes_reactor_gauges)
{
  SELECT_OR_SKIP(eng, 3);
  auto* diagnostics = dynamic_cast<uring::uring_diagnostics*>(eng.get());
  ASSERT_NE(diagnostics, nullptr);
  EXPECT_EQ(diagnostics->reactor_gauges().size(), 3u);
}

TEST_F(UringEngineTest, shutdown_rejects_later_requests)
{
  SELECT_OR_SKIP(eng);
  auto object = eng->open(path, open_options{});
  eng->shutdown();
  std::uint8_t byte{};
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, 1}, host_buffer{&byte});
  EXPECT_ANY_THROW(eng->read(*object, std::move(slices)).get(30s));
}
