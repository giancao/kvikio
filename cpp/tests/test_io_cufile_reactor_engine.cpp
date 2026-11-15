/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The `cufile_reactor` engine, loaded as a plugin by name. Nothing here links the plugin.

#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <kvikio/io/cufile/cufile_options.hpp>
#include <kvikio/io/engine_registry.hpp>
#include <kvikio/io/errors.hpp>

#include "utils/utils.hpp"

using namespace kvikio::io;
using namespace std::chrono_literals;

namespace {

constexpr std::size_t KiB = 1024;
constexpr std::size_t MiB = 1024 * KiB;

std::vector<std::uint8_t> pattern_bytes(std::size_t offset, std::size_t size)
{
  std::vector<std::uint8_t> result(size);
  for (std::size_t i = 0; i < size; ++i) {
    result[i] = static_cast<std::uint8_t>(((offset + i) * 37 + 11) % 249);
  }
  return result;
}

class CufileReactorEngineTest : public testing::Test {
 protected:
  static constexpr std::size_t file_bytes = 3 * MiB + 77;

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

  std::vector<std::uint8_t> device_bytes(kvikio::test::DevBuffer<std::uint8_t> const& buffer)
  {
    std::vector<std::uint8_t> host(buffer.nbytes);
    KVIKIO_CHECK_CUDA(cudaMemcpy(host.data(), buffer.ptr, buffer.nbytes, cudaMemcpyDeviceToHost));
    return host;
  }

  kvikio::test::TempDir tmpdir{true};
  std::string path;
  engine_registry registry;
};

}  // namespace

TEST_F(CufileReactorEngineTest, is_a_plugin_and_declares_what_it_cannot_do)
{
  EXPECT_FALSE(registry.has_factory("cufile_reactor"));
  auto eng = registry.select(engine_route::local, "cufile_reactor");
  EXPECT_EQ(eng->name(), "cufile_reactor");
  auto const& caps = eng->capabilities();
  EXPECT_TRUE(caps.device_read);
  EXPECT_TRUE(caps.device_range_read);
  EXPECT_TRUE(caps.write);
  EXPECT_TRUE(caps.priority);
  // It cannot fill fragmented buffers, so a cache cannot route through it.
  EXPECT_FALSE(caps.fragmented_host_buffer);
  EXPECT_FALSE(caps.vector_host_read);
  EXPECT_FALSE(caps.host_to_device_read);
  EXPECT_EQ(caps.staging_block_size, 0u);
  EXPECT_TRUE(eng->supports(path));
}

TEST_F(CufileReactorEngineTest, reads_host_and_device_destinations)
{
  auto eng    = registry.select(engine_route::local, "cufile_reactor");
  auto object = eng->open(path, open_options{});
  EXPECT_EQ(object->size(), file_bytes);

  std::vector<std::uint8_t> host(MiB + 3);
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{17, host.size()}, host_buffer{host.data()});
  EXPECT_EQ(eng->read(*object, std::move(slices)).get(30s), host.size());
  EXPECT_EQ(host, pattern_bytes(17, host.size()));

  kvikio::test::DevBuffer<std::uint8_t> device(2 * MiB + 9);
  std::vector<prepared_io_slice> device_slices;
  device_slices.emplace_back(range{4096, device.nbytes},
                             device_buffer{static_cast<std::uint8_t*>(device.ptr), nullptr});
  EXPECT_EQ(eng->read(*object, std::move(device_slices)).get(30s), device.nbytes);
  EXPECT_EQ(device_bytes(device), pattern_bytes(4096, device.nbytes));

  std::vector<std::uint8_t> sync(100);
  EXPECT_EQ(eng->host_read(*object, file_bytes - 50, sync.size(), sync.data()), 50u);
}

// Forced onto the POSIX path, a device read stages through the reactor's own pinned scratch.
TEST_F(CufileReactorEngineTest, compat_mode_reads_and_writes_through_posix)
{
  cufile::cufile_reactor_options options;
  options.n_reactors      = 2;
  options.cfg.compat_mode = cufile::compat::on;
  auto eng                = registry.select(engine_route::local, "cufile_reactor", options);

  auto object = eng->open(path, open_options{});
  kvikio::test::DevBuffer<std::uint8_t> device(MiB + 5);
  std::vector<prepared_io_slice> slices;
  slices.emplace_back(range{123, device.nbytes},
                      device_buffer{static_cast<std::uint8_t*>(device.ptr), nullptr});
  EXPECT_EQ(eng->read(*object, std::move(slices)).get(30s), device.nbytes);
  EXPECT_EQ(device_bytes(device), pattern_bytes(123, device.nbytes));

  auto target       = (tmpdir.path() / "out.bin").string();
  auto writer       = eng->open_for_write(target, write_open_options{});
  auto const first  = pattern_bytes(0, MiB);
  auto const second = pattern_bytes(MiB, 300 * KiB);
  kvikio::test::DevBuffer<std::uint8_t> source(second);
  std::vector<prepared_io_slice> writes;
  writes.push_back(make_host_write_slice(range{0, first.size()}, first.data()));
  writes.push_back(make_device_write_slice(
    range{first.size(), second.size()}, static_cast<std::uint8_t const*>(source.ptr), nullptr));
  EXPECT_EQ(eng->write(*writer, std::move(writes)).get(30s), first.size() + second.size());
  eng->flush_async(*writer, io_priority::automatic).get(30s);

  auto reader = eng->open(target, open_options{});
  std::vector<std::uint8_t> back(first.size() + second.size());
  std::vector<prepared_io_slice> read;
  read.emplace_back(range{0, back.size()}, host_buffer{back.data()});
  EXPECT_EQ(eng->read(*reader, std::move(read)).get(30s), back.size());
  EXPECT_EQ(back, pattern_bytes(0, back.size()));
}
