/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The `uring_http` engine, loaded as a plugin by name. Nothing here links the plugin. There is
// no object store to talk to in this test: reads are covered where a loopback server exists.

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kvikio/io/engine_registry.hpp>
#include <kvikio/io/errors.hpp>
#include <kvikio/io/rest/placement.hpp>
#include <kvikio/io/rest/uring_http_api.hpp>

using namespace kvikio::io;
using ::testing::HasSubstr;

namespace {

rest::uring_http_options store_options()
{
  rest::uring_http_options options;
  options.n_reactors              = 2;
  options.object_store.endpoint   = "http://127.0.0.1:9";
  options.object_store.region     = "us-east-1";
  options.object_store.access_key = "test-access-key";
  options.object_store.secret_key = "test-secret-key";
  return options;
}

std::string selection_error(engine_registry& registry, engine_options const& options)
{
  try {
    registry.select(engine_route::s3, "uring_http", options);
  } catch (engine_selection_error const& e) {
    return e.what();
  }
  return {};
}

}  // namespace

TEST(UringHttpEngineTest, is_a_plugin_and_declares_its_capabilities)
{
  engine_registry registry;
  EXPECT_FALSE(registry.has_factory("uring_http")) << "not built into libkvikio.so";
  std::shared_ptr<engine> eng;
  try {
    eng = registry.select(engine_route::s3, rest::uring_http_engine_name, store_options());
  } catch (engine_selection_error const& e) {
    // A sandbox may forbid io_uring; the engine then cannot start here.
    GTEST_SKIP() << "uring_http is unavailable: " << e.what();
  }
  EXPECT_EQ(eng->name(), "uring_http");
  EXPECT_EQ(registry.selected_name(engine_route::s3), "uring_http");

  auto const& caps = eng->capabilities();
  EXPECT_TRUE(caps.device_read);
  EXPECT_TRUE(caps.fragmented_host_buffer);
  EXPECT_TRUE(caps.prefers_bulk_io);
  EXPECT_TRUE(caps.priority);
  EXPECT_FALSE(caps.write);
  EXPECT_FALSE(caps.device_write);
  EXPECT_TRUE(eng->supports("s3://bucket/key.parquet"));
  EXPECT_FALSE(eng->supports("/tmp/local.parquet"));

  // Writes are not implemented, and say so without touching the network.
  EXPECT_THROW((void)eng->open_for_write("s3://bucket/key", write_open_options{}),
               not_implemented_error);

  // What is beyond the engine interface is reached without linking the plugin.
  auto* api = dynamic_cast<rest::uring_http_api*>(eng.get());
  ASSERT_NE(api, nullptr);
  EXPECT_EQ(api->stats().gets, 0u);
  EXPECT_EQ(api->footer_stash_reserved_bytes(), 0u);
  EXPECT_GT(api->list_max_matches(), 0u);
  eng->shutdown();
}

TEST(UringHttpEngineTest, selection_needs_an_authorizer_or_a_configured_store)
{
  engine_registry registry;
  EXPECT_THAT(selection_error(registry, engine_options{}), HasSubstr("needs uring_http_options"));
  rest::uring_http_options unconfigured;
  EXPECT_THAT(selection_error(registry, unconfigured),
              HasSubstr("a request authorizer or a configured object store"));
  EXPECT_EQ(registry.current(engine_route::s3), nullptr);
}

// The placement plan is answered by libkvikio.so, before any engine exists.
TEST(UringHttpPlacementTest, no_nic_or_no_pinning_plans_no_cpus)
{
  EXPECT_TRUE(rest::planned_reactor_cpus(2, true, std::nullopt).empty());
  nic_topology nic;
  nic.node_cpus  = {0, 1, 2, 3};
  nic.queue_irqs = {nic_queue_irq{40, 0, {0}, {0}}};
  nic.irq_cpus   = {0};
  EXPECT_TRUE(rest::planned_reactor_cpus(0, false, nic).empty()) << "pinning is off";
}
