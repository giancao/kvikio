/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <kvikio/io/errors.hpp>
#include <kvikio/io/types.hpp>

#include "utils/fake_engine.hpp"

using namespace kvikio::io;

// A request's lists are allocated once and then only moved.
static_assert(!std::is_copy_constructible_v<host_buffer>);
static_assert(!std::is_copy_assignable_v<host_buffer>);
static_assert(std::is_nothrow_move_constructible_v<host_buffer>);
static_assert(!std::is_copy_constructible_v<prepared_io_slice>);
static_assert(!std::is_copy_assignable_v<prepared_io_slice>);
static_assert(std::is_nothrow_move_constructible_v<prepared_io_slice>);
static_assert(std::is_nothrow_move_assignable_v<prepared_io_slice>);
// A fragment is a plain descriptor.
static_assert(std::is_trivially_copyable_v<fragment>);

TEST(IoTypesTest, range_end_saturates_and_intersect_clips)
{
  EXPECT_EQ((range{10, 5}).end(), 15u);
  EXPECT_EQ((range{std::numeric_limits<std::size_t>::max() - 1, 8}).end(),
            std::numeric_limits<std::size_t>::max());
  EXPECT_TRUE((range{3, 0}).empty());

  auto const both = intersect(range{0, 100}, range{60, 100});
  EXPECT_EQ(both.offset, 60u);
  EXPECT_EQ(both.size, 40u);
  EXPECT_TRUE(intersect(range{0, 10}, range{10, 10}).empty());
}

TEST(IoTypesTest, host_buffer_has_three_forms)
{
  host_buffer staged;
  EXPECT_TRUE(staged.needs_staging());
  EXPECT_TRUE(staged.fragments().empty());

  std::uint8_t byte{};
  host_buffer contiguous{&byte};
  EXPECT_TRUE(contiguous.is_contiguous());
  EXPECT_TRUE(contiguous.fragments().empty());

  std::array<std::uint8_t, 8> memory{};
  std::vector<fragment> list;
  list.push_back(fragment{100, memory.data(), 4, &memory[0]});
  list.push_back(fragment{900, memory.data() + 4, 4, &memory[4]});
  auto const* storage = list.data();
  host_buffer fragmented{std::move(list)};
  EXPECT_TRUE(fragmented.is_fragmented());
  ASSERT_EQ(fragmented.fragments().size(), 2u);
  // The list was moved in, not copied.
  EXPECT_EQ(fragmented.fragments().data(), storage);
  EXPECT_EQ(fragmented.fragments()[1].rng().offset, 900u);

  // Moving the buffer moves the same list along.
  host_buffer moved{std::move(fragmented)};
  EXPECT_EQ(moved.fragments().data(), storage);
}

TEST(IoTypesTest, slice_kinds)
{
  std::uint8_t byte{};
  prepared_io_slice host{range{0, 1}, host_buffer{&byte}};
  EXPECT_TRUE(host.is_host_request());
  EXPECT_FALSE(host.has_device_request());
  EXPECT_FALSE(host.is_write());

  prepared_io_slice device{range{0, 1}, device_buffer{&byte, nullptr, 0}};
  EXPECT_TRUE(device.needs_staging());
  EXPECT_TRUE(device.has_device_request());

  auto const write = make_host_write_slice(range{4096, 1}, &byte, io_priority::low);
  EXPECT_TRUE(write.is_write());
  EXPECT_EQ(write.priority, io_priority::low);
  EXPECT_EQ(write.offset(), 4096u);

  auto const flush = make_flush_slice();
  EXPECT_TRUE(flush.is_flush());
  EXPECT_TRUE(flush.rng.empty());

  EXPECT_EQ(resolve_priority(io_priority::automatic, io_priority::high), io_priority::high);
  EXPECT_EQ(resolve_priority(io_priority::low, io_priority::high), io_priority::low);
}

// The engine hands fragments back per physical operation, not once per request.
TEST(IoTypesTest, fragments_are_handed_back_per_operation)
{
  std::array<std::uint8_t, 12> memory{};
  std::vector<fragment> list;
  list.push_back(fragment{0, memory.data(), 4, &memory[0]});
  list.push_back(fragment{500, memory.data() + 4, 4, &memory[4]});
  list.push_back(fragment{700, memory.data() + 8, 4, &memory[8]});

  std::vector<std::vector<fragment_handle>> calls;
  auto completion = std::make_shared<prepared_io_completion>(
    [&calls](std::span<fragment_handle const> handles, bool ok) noexcept {
      EXPECT_TRUE(ok);
      calls.emplace_back(handles.begin(), handles.end());
    });

  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{0, 704}, host_buffer{std::move(list)});
  slices.back().on_complete = completion;

  kvikio::test::fake_engine engine{"fake"};
  kvikio::test::fake_object object{"fake://object", 4096};
  auto future = engine.read(object, std::move(slices));
  EXPECT_EQ(std::move(future).get(), 12u);

  ASSERT_EQ(calls.size(), 3u);
  EXPECT_EQ(calls[0], std::vector<fragment_handle>{&memory[0]});
  EXPECT_EQ(calls[1], std::vector<fragment_handle>{&memory[4]});
  EXPECT_EQ(calls[2], std::vector<fragment_handle>{&memory[8]});
  // Each fragment holds the bytes of its own file range.
  EXPECT_EQ(memory[0], kvikio::test::fake_byte(0));
  EXPECT_EQ(memory[4], kvikio::test::fake_byte(500));
  EXPECT_EQ(memory[11], kvikio::test::fake_byte(703));
}

TEST(IoTypesTest, object_identity_defaults_to_this_open)
{
  kvikio::test::fake_object first{"fake://object", 10};
  kvikio::test::fake_object second{"fake://object", 10};
  EXPECT_LT(first.open_generation(), second.open_generation());
  EXPECT_EQ(first.identity().kind, identity_kind::opened_handle);
  // Without a strong validator, two opens never share an identity.
  EXPECT_NE(first.identity_cache_key(), second.identity_cache_key());
}

// What an engine does not implement reports not_implemented_error.
TEST(IoTypesTest, unimplemented_entry_points_say_so)
{
  kvikio::test::fake_engine engine{"fake"};
  kvikio::test::fake_object object{"fake://object", 16};
  std::uint8_t byte{};

  EXPECT_FALSE(engine.capabilities().write);
  EXPECT_THROW(engine.host_write(object, 0, 1, &byte), not_implemented_error);
  EXPECT_THROW(engine.flush(object), not_implemented_error);
  EXPECT_THROW((void)engine.open_for_write("fake://new", write_open_options{}),
               not_implemented_error);

  std::vector<prepared_io_slice> slices;
  slices.push_back(make_host_write_slice(range{0, 1}, &byte));
  try {
    (void)engine.write(object, std::move(slices)).get();
    FAIL() << "write should have failed";
  } catch (std::system_error const& e) {
    EXPECT_EQ(e.code(), std::make_error_code(std::errc::not_supported));
  }
  EXPECT_THROW(engine.flush_async(object, io_priority::automatic).get(), not_implemented_error);
}
