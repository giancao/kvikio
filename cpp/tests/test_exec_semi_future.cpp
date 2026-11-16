/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke tests of kvikio::exec. The full suite of this code lives with its main
// user, cuCascade, and runs through cuCascade's forwarding headers.

#include <stdexcept>
#include <thread>
#include <utility>

#include <gtest/gtest.h>
#include <kvikio/exec/semi_future.hpp>

using namespace kvikio::exec;

TEST(SemiFutureTest, value_set_before_get)
{
  promise<int> p;
  auto f = p.get_semi_future();
  p.set_value(7);
  EXPECT_TRUE(f.is_ready());
  EXPECT_EQ(std::move(f).get(), 7);
}

TEST(SemiFutureTest, get_blocks_until_another_thread_sets_the_value)
{
  promise<int> p;
  auto f = p.get_semi_future();
  std::thread producer([&p] { p.set_value(11); });
  EXPECT_EQ(std::move(f).get(), 11);
  producer.join();
}

TEST(SemiFutureTest, exception_is_rethrown_by_get)
{
  promise<int> p;
  auto f = p.get_semi_future();
  p.set_exception(std::runtime_error("boom"));
  EXPECT_THROW(std::move(f).get(), std::runtime_error);
}

TEST(SemiFutureTest, callback_runs_on_the_thread_that_arrives_second)
{
  promise<int> p;
  auto f   = p.get_semi_future();
  int seen = 0;
  std::move(f).install_callback([&seen](try_t<int>&& t) { seen = std::move(t).value(); });
  EXPECT_EQ(seen, 0);
  p.set_value(5);
  EXPECT_EQ(seen, 5);
}

TEST(SemiFutureTest, dropped_promise_breaks_the_future)
{
  semi_future<int> f;
  {
    promise<int> p;
    f = p.get_semi_future();
  }
  EXPECT_THROW(std::move(f).get(), broken_promise_error);
}

TEST(SemiFutureTest, ready_futures)
{
  EXPECT_EQ(make_semi_future(try_t<int>(3)).get(), 3);
  EXPECT_THROW(make_semi_future_with([]() -> int { throw std::logic_error("no"); }).get(),
               std::logic_error);
  make_semi_future().get();
}

TEST(SemiFutureTest, invocable_is_move_only_and_callable_once)
{
  auto owned = std::make_unique<int>(9);
  invocable<int() &&> f{[owned = std::move(owned)]() { return *owned; }};
  invocable<int() &&> g{std::move(f)};
  EXPECT_EQ(std::move(g)(), 9);
}
