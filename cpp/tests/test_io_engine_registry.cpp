/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kvikio/io/engine_plugin.hpp>
#include <kvikio/io/engine_registry.hpp>

#include "utils/fake_engine.hpp"

#ifndef KVIKIO_TEST_ENGINE_PLUGIN_DIR
#error "KVIKIO_TEST_ENGINE_PLUGIN_DIR must be defined"
#endif

using namespace kvikio::io;
using kvikio::test::fake_engine;
using ::testing::HasSubstr;

// Reading the current engine must not take a lock.
static_assert(std::atomic<engine*>::is_always_lock_free);

namespace {

engine_factory make_factory(std::string name,
                            std::shared_ptr<fake_engine>* created = nullptr,
                            bool fail_start                       = false)
{
  return [name = std::move(name), created, fail_start](engine_options const&) {
    auto result = std::make_shared<fake_engine>(name, fail_start);
    if (created != nullptr) { *created = result; }
    return std::shared_ptr<engine>(result);
  };
}

/// What() of the engine_selection_error @p fn throws.
template <typename Fn>
std::string selection_error(Fn&& fn)
{
  try {
    fn();
  } catch (engine_selection_error const& e) {
    return e.what();
  }
  ADD_FAILURE() << "expected engine_selection_error";
  return {};
}

}  // namespace

TEST(EngineRegistryTest, defaults_are_the_built_in_engines)
{
  EXPECT_EQ(default_engine_name(engine_route::local), "cufile_posix");
  EXPECT_EQ(default_engine_name(engine_route::s3), "curl");
}

TEST(EngineRegistryTest, nothing_is_selected_at_first)
{
  engine_registry registry;
  EXPECT_EQ(registry.current(engine_route::local), nullptr);
  EXPECT_EQ(registry.acquire(engine_route::s3), nullptr);
  EXPECT_TRUE(registry.selected_name(engine_route::local).empty());
}

TEST(EngineRegistryTest, select_creates_starts_and_publishes_the_engine)
{
  engine_registry registry;
  std::shared_ptr<fake_engine> created;
  registry.register_factory("alpha", make_factory("alpha", &created));
  EXPECT_TRUE(registry.has_factory("alpha"));
  EXPECT_EQ(created, nullptr) << "registering must not create the engine";

  auto selected = registry.select(engine_route::local, "alpha");
  ASSERT_NE(created, nullptr);
  EXPECT_TRUE(created->started.load());
  EXPECT_EQ(selected.get(), created.get());
  EXPECT_EQ(registry.current(engine_route::local), created.get());
  EXPECT_EQ(registry.acquire(engine_route::local), selected);
  EXPECT_EQ(registry.selected_name(engine_route::local), "alpha");
  // The other route is untouched.
  EXPECT_EQ(registry.current(engine_route::s3), nullptr);
}

TEST(EngineRegistryTest, routes_are_selected_independently)
{
  engine_registry registry;
  registry.register_factory("alpha", make_factory("alpha"));
  registry.register_factory("beta", make_factory("beta"));
  registry.select(engine_route::local, "alpha");
  registry.select(engine_route::s3, "beta");
  EXPECT_EQ(registry.selected_name(engine_route::local), "alpha");
  EXPECT_EQ(registry.selected_name(engine_route::s3), "beta");
}

TEST(EngineRegistryTest, engine_options_reach_the_factory)
{
  struct my_options : engine_options {
    int depth{0};
  };
  engine_registry registry;
  int seen = 0;
  registry.register_factory("alpha", [&seen](engine_options const& options) {
    seen = dynamic_cast<my_options const&>(options).depth;
    return std::shared_ptr<engine>(std::make_shared<fake_engine>("alpha"));
  });
  my_options options;
  options.depth = 64;
  registry.select(engine_route::local, "alpha", options);
  EXPECT_EQ(seen, 64);
}

// A failed selection is reported and the previous selection stays in effect.
TEST(EngineRegistryTest, unknown_engine_keeps_the_previous_selection)
{
  engine_registry registry;
  registry.add_plugin_search_path(KVIKIO_TEST_ENGINE_PLUGIN_DIR);
  registry.register_factory("alpha", make_factory("alpha"));
  auto* const before = registry.select(engine_route::local, "alpha").get();

  auto const what =
    selection_error([&] { registry.select(engine_route::local, "no_such_engine"); });
  EXPECT_THAT(what, HasSubstr("no_such_engine"));
  EXPECT_THAT(what, HasSubstr("libkvikio_engine_no_such_engine.so"));
  EXPECT_THAT(what, HasSubstr("still using 'alpha'"));

  EXPECT_EQ(registry.current(engine_route::local), before);
  EXPECT_EQ(registry.selected_name(engine_route::local), "alpha");
}

TEST(EngineRegistryTest, engine_that_cannot_start_keeps_the_previous_selection)
{
  engine_registry registry;
  std::shared_ptr<fake_engine> broken;
  registry.register_factory("alpha", make_factory("alpha"));
  registry.register_factory("broken", make_factory("broken", &broken, /*fail_start=*/true));
  auto* const before = registry.select(engine_route::local, "alpha").get();

  auto const what = selection_error([&] { registry.select(engine_route::local, "broken"); });
  EXPECT_THAT(what, HasSubstr("io_uring is disabled"));
  EXPECT_THAT(what, HasSubstr("still using 'alpha'"));
  EXPECT_EQ(registry.current(engine_route::local), before);
  ASSERT_NE(broken, nullptr);
  EXPECT_FALSE(broken->started.load());
}

TEST(EngineRegistryTest, failed_first_selection_leaves_nothing_selected)
{
  engine_registry registry;
  registry.register_factory("broken", make_factory("broken", nullptr, /*fail_start=*/true));
  auto const what = selection_error([&] { registry.select(engine_route::s3, "broken"); });
  EXPECT_THAT(what, HasSubstr("no engine is selected"));
  EXPECT_EQ(registry.current(engine_route::s3), nullptr);
}

TEST(EngineRegistryTest, factory_failures_are_selection_errors)
{
  engine_registry registry;
  registry.register_factory("throws", [](engine_options const&) -> std::shared_ptr<engine> {
    throw std::runtime_error("out of file descriptors");
  });
  registry.register_factory("null",
                            [](engine_options const&) { return std::shared_ptr<engine>{}; });
  EXPECT_THAT(selection_error([&] { registry.select(engine_route::local, "throws"); }),
              HasSubstr("out of file descriptors"));
  EXPECT_THAT(selection_error([&] { registry.select(engine_route::local, "null"); }),
              HasSubstr("returned no engine"));
}

TEST(EngineRegistryTest, names_are_restricted)
{
  engine_registry registry;
  EXPECT_THROW(registry.register_factory("../evil", make_factory("x")), std::invalid_argument);
  EXPECT_THROW(registry.register_factory("", make_factory("x")), std::invalid_argument);
  registry.register_factory("alpha", make_factory("alpha"));
  EXPECT_THROW(registry.register_factory("alpha", make_factory("alpha")), std::invalid_argument);
  // A name that could escape the plugin directory is refused before any file is looked for.
  EXPECT_THAT(selection_error([&] { registry.select(engine_route::local, "../evil"); }),
              HasSubstr("[a-z0-9_]+"));
}

TEST(EngineRegistryTest, replaced_engine_stays_valid_until_shutdown)
{
  std::shared_ptr<fake_engine> first;
  std::shared_ptr<fake_engine> second;
  {
    engine_registry registry;
    registry.register_factory("alpha", make_factory("alpha", &first));
    registry.register_factory("beta", make_factory("beta", &second));
    registry.select(engine_route::local, "alpha");
    registry.select(engine_route::local, "beta");
    EXPECT_EQ(registry.current(engine_route::local), second.get());
    // A holder of the replaced engine can still use it.
    EXPECT_EQ(first->shutdowns.load(), 0);

    registry.shutdown();
    EXPECT_EQ(registry.current(engine_route::local), nullptr);
    EXPECT_TRUE(registry.selected_name(engine_route::local).empty());
    EXPECT_EQ(first->shutdowns.load(), 1);
    EXPECT_EQ(second->shutdowns.load(), 1);
    // Factories survive a shutdown.
    registry.select(engine_route::local, "alpha");
  }
  // Destroying the registry shuts down what it still holds.
  EXPECT_EQ(first->shutdowns.load(), 1);
}

TEST(EngineRegistryTest, process_wide_instance_is_one_object)
{
  EXPECT_EQ(&engine_registry::instance(), &engine_registry::instance());
}

// -- Plugins ---------------------------------------------------------------------

TEST(EnginePluginTest, search_path_ends_with_the_library_directory)
{
  engine_registry registry;
  registry.add_plugin_search_path("/first");
  auto const path = registry.plugin_search_path();
  ASSERT_GE(path.size(), 2u);
  EXPECT_EQ(path.front(), "/first");
  EXPECT_FALSE(path.back().empty());
  EXPECT_EQ(engine_plugin_file_name("uring"), "libkvikio_engine_uring.so");
}

TEST(EnginePluginTest, plugin_is_found_by_name_and_loaded)
{
  engine_registry registry;
  registry.add_plugin_search_path(KVIKIO_TEST_ENGINE_PLUGIN_DIR);
  EXPECT_FALSE(registry.has_factory("test_good"));

  auto selected = registry.select(engine_route::local, "test_good");
  ASSERT_NE(selected, nullptr);
  EXPECT_EQ(selected->name(), "test_good");
  EXPECT_TRUE(registry.has_factory("test_good"));

  // The engine object was created inside the plugin; its type is still known here.
  auto* fake = dynamic_cast<fake_engine*>(selected.get());
  ASSERT_NE(fake, nullptr);
  EXPECT_TRUE(fake->started.load());

  // And it serves a request through the interface.
  auto object = selected->open("fake://object", open_options{});
  std::uint8_t bytes[4]{};
  EXPECT_EQ(selected->host_read(*object, 300, 4, bytes), 4u);
  EXPECT_EQ(bytes[0], kvikio::test::fake_byte(300));

  // An exception thrown inside the plugin is caught by its type here.
  EXPECT_THROW(selected->flush(*object), not_implemented_error);
}

TEST(EnginePluginTest, plugin_built_for_another_abi_is_rejected)
{
  engine_registry registry;
  registry.add_plugin_search_path(KVIKIO_TEST_ENGINE_PLUGIN_DIR);
  auto const what = selection_error([&] { registry.select(engine_route::local, "test_bad_abi"); });
  EXPECT_THAT(what, HasSubstr("rejected"));
  EXPECT_THAT(what, HasSubstr("engine ABI"));
  EXPECT_FALSE(registry.has_factory("test_bad_abi"));
  EXPECT_EQ(registry.current(engine_route::local), nullptr);
}

TEST(EnginePluginTest, plugin_providing_another_engine_is_rejected)
{
  engine_registry registry;
  registry.add_plugin_search_path(KVIKIO_TEST_ENGINE_PLUGIN_DIR);
  auto const what =
    selection_error([&] { registry.select(engine_route::local, "test_wrong_name"); });
  EXPECT_THAT(what, HasSubstr("some_other_engine"));
}

// The plugin loads, but its engine cannot start here: the default stays.
TEST(EnginePluginTest, plugin_engine_that_cannot_start_keeps_the_previous_selection)
{
  engine_registry registry;
  registry.add_plugin_search_path(KVIKIO_TEST_ENGINE_PLUGIN_DIR);
  registry.register_factory("alpha", make_factory("alpha"));
  auto* const before = registry.select(engine_route::local, "alpha").get();

  auto const what =
    selection_error([&] { registry.select(engine_route::local, "test_start_fails"); });
  EXPECT_THAT(what, HasSubstr("io_uring is disabled"));
  EXPECT_EQ(registry.current(engine_route::local), before);
}
