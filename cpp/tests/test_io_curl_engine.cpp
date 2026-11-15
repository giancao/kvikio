/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The `curl` engine against a loopback HTTP server. s3:// URLs go through the same code with a
// different endpoint class, which needs a real object store and is not covered here.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kvikio/io/curl_engine.hpp>
#include <kvikio/io/engine_registry.hpp>
#include <kvikio/io/errors.hpp>
#include <kvikio/io/object_listing.hpp>

#include "utils/env.hpp"
#include "utils/loopback_http_server.hpp"
#include "utils/test_staging.hpp"
#include "utils/utils.hpp"

using namespace kvikio::io;
using namespace std::chrono_literals;
using kvikio::test::LoopbackHttpServer;
using kvikio::test::test_staging;
using ::testing::HasSubstr;

namespace {

constexpr std::size_t KiB = 1024;
constexpr std::size_t MiB = 1024 * KiB;

std::vector<std::uint8_t> pattern_bytes(std::size_t offset, std::size_t size)
{
  std::vector<std::uint8_t> result(size);
  for (std::size_t i = 0; i < size; ++i) {
    result[i] = static_cast<std::uint8_t>(((offset + i) * 97 + 5) % 253);
  }
  return result;
}

class CurlEngineTest : public testing::Test {
 protected:
  static constexpr std::size_t object_bytes = 3 * MiB + 333;

  /// An engine pointed at the loopback server as its S3 endpoint, signing with @p access_key.
  std::shared_ptr<engine> select_store(std::string access_key, std::string secret_key = "secret")
  {
    curl_options options;
    options.object_store.endpoint   = server.url("");
    options.object_store.region     = "us-east-1";
    options.object_store.access_key = std::move(access_key);
    options.object_store.secret_key = std::move(secret_key);
    return registry.select(engine_route::s3, curl_engine_name, options);
  }

  std::shared_ptr<engine> select(std::shared_ptr<staging_source> staging = nullptr,
                                 std::size_t max_operation_bytes         = 16 * MiB)
  {
    curl_options options;
    options.staging             = std::move(staging);
    options.max_operation_bytes = max_operation_bytes;
    return registry.select(engine_route::s3, curl_engine_name, options);
  }

  std::vector<std::uint8_t> device_bytes(kvikio::test::DevBuffer<std::uint8_t> const& buffer)
  {
    std::vector<std::uint8_t> host(buffer.nbytes);
    KVIKIO_CHECK_CUDA(cudaMemcpy(host.data(), buffer.ptr, buffer.nbytes, cudaMemcpyDeviceToHost));
    return host;
  }

  LoopbackHttpServer server{pattern_bytes(0, object_bytes)};
  engine_registry registry;
};

}  // namespace

TEST_F(CurlEngineTest, is_built_in_and_declares_what_it_cannot_do)
{
  EXPECT_TRUE(registry.has_factory("curl"));
  EXPECT_EQ(default_engine_name(engine_route::s3), "curl");
  auto eng = select();
  EXPECT_EQ(eng->name(), "curl");
  auto const& caps = eng->capabilities();
  EXPECT_TRUE(caps.device_read);
  EXPECT_TRUE(caps.vector_host_read);
  EXPECT_TRUE(caps.fragmented_host_buffer);
  EXPECT_FALSE(caps.write);
  EXPECT_FALSE(caps.device_write);
  EXPECT_FALSE(caps.priority);
  EXPECT_FALSE(caps.prefers_bulk_io);
  EXPECT_TRUE(eng->supports("s3://bucket/key"));
  EXPECT_TRUE(eng->supports("https://host/key"));
  EXPECT_FALSE(eng->supports("/tmp/local.bin"));
  auto object = eng->open(server.url(), open_options{});
  EXPECT_THROW((void)eng->open_for_write(server.url(), write_open_options{}),
               not_implemented_error);
  std::uint8_t byte{};
  EXPECT_THROW((void)eng->host_write(*object, 0, 1, &byte), not_implemented_error);
}

// One HEAD gives the size and the validator; a known size sends nothing.
TEST_F(CurlEngineTest, open_takes_the_size_and_the_etag_from_one_head)
{
  auto eng    = select();
  auto object = eng->open(server.url(), open_options{});
  EXPECT_EQ(server.head_requests(), 1u);
  EXPECT_EQ(object->size(), object_bytes);
  EXPECT_EQ(object->validation_tag(), "\"v1\"");
  EXPECT_TRUE(object->validation_tag_is_strong());
  EXPECT_EQ(object->identity().kind, identity_kind::object_tag);
  EXPECT_FALSE(object->writable());

  open_options known;
  known.known_size = 4096;
  auto sized       = eng->open(server.url(), known);
  EXPECT_EQ(server.head_requests(), 1u) << "a known size needs no request";
  EXPECT_EQ(sized->size(), 4096u);
  EXPECT_EQ(sized->identity().kind, identity_kind::opened_handle);

  EXPECT_ANY_THROW((void)eng->open(server.url("/missing.bin"), open_options{}));
}

TEST_F(CurlEngineTest, reads_host_slices_and_clamps_to_the_end)
{
  auto eng    = select();
  auto object = eng->open(server.url(), open_options{});

  std::vector<std::uint8_t> a(MiB + 7), b(999), c(8192);
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{3, a.size()}, host_buffer{a.data()});
  slices.emplace_back(range{2 * MiB + 1, b.size()}, host_buffer{b.data()});
  slices.emplace_back(range{object_bytes - 100, c.size()}, host_buffer{c.data()});
  EXPECT_EQ(eng->read(*object, std::move(slices)).get(30s), a.size() + b.size() + 100);
  EXPECT_EQ(a, pattern_bytes(3, a.size()));
  EXPECT_EQ(b, pattern_bytes(2 * MiB + 1, b.size()));
  c.resize(100);
  EXPECT_EQ(c, pattern_bytes(object_bytes - 100, 100));

  std::vector<std::uint8_t> sync(4096);
  EXPECT_EQ(eng->host_read(*object, 12345, sync.size(), sync.data()), sync.size());
  EXPECT_EQ(sync, pattern_bytes(12345, sync.size()));
  EXPECT_EQ(eng->host_read(*object, object_bytes - 10, 100, sync.data()), 10u);

  // A slice entirely past the end reads nothing and still settles.
  std::vector<prepared_io_slice> beyond;
  beyond.emplace_back(range{object_bytes + 10, 100}, host_buffer{sync.data()});
  EXPECT_EQ(eng->read(*object, std::move(beyond)).get(30s), 0u);
}

// A read larger than max_operation_bytes becomes several range requests.
TEST_F(CurlEngineTest, a_large_read_is_split_into_range_requests)
{
  auto eng    = select(nullptr, 256 * KiB);
  auto object = eng->open(server.url(), open_options{});
  std::vector<std::uint8_t> all(MiB);
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, all.size()}, host_buffer{all.data()});
  auto const before = server.get_requests();
  EXPECT_EQ(eng->read(*object, std::move(slices)).get(30s), all.size());
  EXPECT_EQ(server.get_requests() - before, 4u);
  EXPECT_EQ(all, pattern_bytes(0, all.size()));
}

// Each fragment is filled in place and handed back when its own request finishes.
TEST_F(CurlEngineTest, fragments_are_filled_in_place_and_handed_back)
{
  auto eng                            = select();
  auto object                         = eng->open(server.url(), open_options{});
  constexpr std::size_t n_fragments   = 4;
  constexpr std::size_t fragment_size = 128 * KiB;
  std::vector<std::uint8_t> memory(n_fragments * fragment_size);
  std::vector<fragment> list;
  for (std::size_t i = 0; i < n_fragments; ++i) {
    auto* data = memory.data() + i * fragment_size;
    list.push_back(fragment{i * fragment_size, data, fragment_size, data});
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
  slices.emplace_back(range{0, memory.size()}, host_buffer{std::move(list)});
  slices.back().on_complete = completion;
  EXPECT_EQ(eng->read(*object, std::move(slices)).get(30s), memory.size());
  EXPECT_EQ(handed_back.size(), n_fragments);
  EXPECT_EQ(memory, pattern_bytes(0, memory.size()));
}

// A device read is staged through the injected source; the bytes are in place when the future
// settles, with nothing left for the caller to wait on.
TEST_F(CurlEngineTest, reads_into_device_memory_through_the_injected_staging)
{
  // Two blocks for five range requests: the reactor parks the rest until a block comes back.
  auto staging = std::make_shared<test_staging>(256 * KiB, 2);
  auto eng     = select(staging);
  EXPECT_EQ(eng->capabilities().staging_block_size, 256 * KiB);
  auto object = eng->open(server.url(), open_options{});

  kvikio::test::DevBuffer<std::uint8_t> device(MiB + 17);
  cudaStream_t stream{};
  KVIKIO_CHECK_CUDA(cudaStreamCreate(&stream));
  for (CUstream s : {static_cast<CUstream>(stream), static_cast<CUstream>(nullptr)}) {
    KVIKIO_CHECK_CUDA(cudaMemset(device.ptr, 0, device.nbytes));
    std::vector<prepared_io_slice> slices;
    slices.emplace_back(range{5000, device.nbytes},
                        device_buffer{static_cast<std::uint8_t*>(device.ptr), s});
    EXPECT_EQ(eng->read(*object, std::move(slices)).get(30s), device.nbytes);
    EXPECT_EQ(device_bytes(device), pattern_bytes(5000, device.nbytes));
  }

  // Host and device together: the bytes land in host memory and are copied from there.
  std::vector<std::uint8_t> host(300 * KiB);
  kvikio::test::DevBuffer<std::uint8_t> both(host.size());
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{777, host.size()},
                      host_buffer{host.data()},
                      device_buffer{static_cast<std::uint8_t*>(both.ptr), stream});
  EXPECT_EQ(eng->read(*object, std::move(slices)).get(30s), host.size());
  EXPECT_EQ(host, pattern_bytes(777, host.size()));
  EXPECT_EQ(device_bytes(both), pattern_bytes(777, host.size()));
  KVIKIO_CHECK_CUDA(cudaStreamDestroy(stream));
}

TEST_F(CurlEngineTest, device_reads_without_usable_staging_fail)
{
  kvikio::test::DevBuffer<std::uint8_t> device(64 * KiB);
  auto const read_device = [&](engine& eng) {
    auto object = eng.open(server.url(), open_options{});
    std::vector<prepared_io_slice> slices;
    slices.emplace_back(range{0, device.nbytes},
                        device_buffer{static_cast<std::uint8_t*>(device.ptr), nullptr});
    return eng.read(*object, std::move(slices));
  };
  // No staging source at all: rejected before anything is sent.
  auto without = select();
  EXPECT_THROW(read_device(*without).get(30s), std::invalid_argument);
  // The only block is lent to someone else, and nothing of this engine would return one: the
  // read fails with the source's exhaustion error, a bad_alloc, instead of waiting forever.
  auto staging   = std::make_shared<test_staging>(64 * KiB, 1);
  auto held      = staging->acquire(64 * KiB);
  auto exhausted = select(staging);
  ASSERT_EQ(held.status, staging_status::ok);
  EXPECT_THROW(read_device(*exhausted).get(30s), std::bad_alloc);
  // With the block back, the same read works.
  held.lease.reset();
  EXPECT_EQ(read_device(*exhausted).get(30s), device.nbytes);
}

TEST_F(CurlEngineTest, a_failed_request_fails_the_read_and_hands_fragments_back)
{
  auto eng = select();
  open_options known;
  known.known_size = 4096;
  auto missing     = eng->open(server.url("/missing.bin"), known);
  std::vector<std::uint8_t> memory(4096);
  bool handed_back_failed = false;
  auto completion         = std::make_shared<prepared_io_completion>(
    [&](std::span<fragment_handle const>, bool ok) noexcept { handed_back_failed = !ok; });
  std::vector<fragment> list{fragment{0, memory.data(), memory.size(), memory.data()}};
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, memory.size()}, host_buffer{std::move(list)});
  slices.back().on_complete = completion;
  EXPECT_ANY_THROW(eng->read(*missing, std::move(slices)).get(30s));
  EXPECT_TRUE(handed_back_failed);
}

// The options name the object store and its credentials; they win over the environment, and an
// empty field leaves its environment variable in charge.
TEST_F(CurlEngineTest, options_override_the_environment_for_s3)
{
  kvikio::test::EnvVarContext const env{{"AWS_ACCESS_KEY_ID", "ENVKEY"},
                                        {"AWS_SECRET_ACCESS_KEY", "envsecret"},
                                        {"AWS_DEFAULT_REGION", "eu-west-9"}};
  auto eng    = select_store("OPTIONKEY");
  auto object = eng->open("s3://bkt/object.bin", open_options{});
  EXPECT_EQ(object->size(), object_bytes);
  EXPECT_EQ(server.last_target(), "/bkt/object.bin") << "path-style on the configured endpoint";
  EXPECT_THAT(server.last_authorization(), HasSubstr("AWS4-HMAC-SHA256"));
  EXPECT_THAT(server.last_authorization(), HasSubstr("Credential=OPTIONKEY/"));
  EXPECT_THAT(server.last_authorization(), HasSubstr("/us-east-1/s3/"));

  // The reads are signed the same way.
  std::vector<std::uint8_t> bytes(4096);
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{100, bytes.size()}, host_buffer{bytes.data()});
  EXPECT_EQ(eng->read(*object, std::move(slices)).get(30s), bytes.size());
  EXPECT_EQ(bytes, pattern_bytes(100, bytes.size()));
  EXPECT_THAT(server.last_authorization(), HasSubstr("Credential=OPTIONKEY/"));

  // No key in the options: the environment's key, with the options' endpoint and region.
  auto from_env = select_store("", "");
  (void)from_env->open("s3://bkt/object.bin", open_options{});
  EXPECT_THAT(server.last_authorization(), HasSubstr("Credential=ENVKEY/"));
  EXPECT_THAT(server.last_authorization(), HasSubstr("/us-east-1/s3/"));
}

// Listing uses the paging logic shared with the uring_http engine; curl fetches the pages.
TEST_F(CurlEngineTest, lists_a_bucket_page_by_page)
{
  auto eng      = select_store("OPTIONKEY");
  auto* listing = dynamic_cast<object_listing*>(eng.get());
  ASSERT_NE(listing, nullptr);
  EXPECT_GT(listing->list_max_matches(), 0u);

  auto const entries = listing->list_objects("bkt", "data/", 2);
  ASSERT_EQ(entries.size(), 3u);
  EXPECT_EQ(entries[0].key, "data/part.0.parquet");
  EXPECT_EQ(entries[1].size, 200u);
  EXPECT_EQ(entries[2].key, "data/part.2.parquet");
  EXPECT_EQ(entries[2].etag, "\"e2\"");
  EXPECT_EQ(server.list_requests(), 2u);
  EXPECT_THAT(server.last_target(), HasSubstr("/bkt/?continuation-token=page-two&list-type=2"));
  EXPECT_THAT(server.last_target(), HasSubstr("prefix=data%2F"));
  EXPECT_THAT(server.last_authorization(), HasSubstr("Credential=OPTIONKEY/"));

  // A sink that stops after the first page sends no second request.
  std::size_t pages = 0;
  listing->list_objects_paged("bkt", "data/", 2, [&](rest::s3::list_objects_v2_page const& page) {
    ++pages;
    EXPECT_TRUE(page.is_truncated);
    return false;
  });
  EXPECT_EQ(pages, 1u);
  EXPECT_EQ(server.list_requests(), 3u);
}

TEST_F(CurlEngineTest, shutdown_rejects_later_requests)
{
  auto eng    = select();
  auto object = eng->open(server.url(), open_options{});
  eng->shutdown();
  std::uint8_t byte{};
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, 1}, host_buffer{&byte});
  EXPECT_ANY_THROW(eng->read(*object, std::move(slices)).get(30s));
}
