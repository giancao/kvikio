/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <kvikio/io/cufile/cufile_engine.hpp>
#include <kvikio/io/engine_plugin.hpp>

namespace kvikio::io::cufile {

namespace {

auto reactor_factory(std::shared_ptr<cufile_reactor::reactor_context> context)
{
  return [context = std::move(context), i = 0]() mutable {
    return std::make_unique<cufile_reactor>(context, "cufile-" + std::to_string(i++));
  };
}

}  // namespace

cufile_reactor_engine::cufile_reactor_engine(
  std::size_t n_reactors, std::shared_ptr<cufile_reactor::reactor_context> context)
  : kvikio::io::detail::reactor_engine<cufile_reactor>(
      cufile_reactor_engine_name, n_reactors, reactor_factory(std::move(context)))
{
}

cufile_reactor_engine::~cufile_reactor_engine() { shutdown(); }

std::shared_ptr<io_object> cufile_reactor_engine::open(std::string path,
                                                       open_options const& /*options*/)
{
  return std::shared_ptr<io_object>(
    cufile_reactor::create_io_object(std::move(path), _config.use_odirect, _config.compat_mode));
}

std::shared_ptr<io_object> cufile_reactor_engine::open_for_write(std::string path,
                                                                 write_open_options const& options)
{
  return std::shared_ptr<io_object>(cufile_reactor::create_io_object_for_write(
    std::move(path), options, _config.use_odirect, _config.compat_mode));
}

std::vector<cufile_reactor*> cufile_reactor_engine::next_reactor(cufile_io_object const& /*object*/,
                                                                 std::size_t n_slices,
                                                                 io_op_type type,
                                                                 int device_id)
{
  auto const count = _reactors.size();
  if (type == io_op_type::write_vector_async && device_id < 0 && count > 1) {
    // A host-only vectored write (device_id stays unresolved without device
    // slices) goes to one reactor: its page-cache writes with early writeback
    // already keep the device busy, and interleaving writers of one file only
    // contends on its inode.
    auto selected = default_next_reactor();
    if (selected.size() > 1) selected.resize(1);
    return selected;
  }
  if (type != io_op_type::host_vector_async || n_slices <= 2 || count <= 2) {
    return default_next_reactor();
  }
  // A vectored read: every reactor, least backlogged first.
  auto const start = _next.fetch_add(1, std::memory_order_relaxed) % count;
  std::vector<std::pair<std::size_t, std::size_t>> ranked;
  ranked.reserve(count);
  for (std::size_t distance = 0; distance < count; ++distance) {
    ranked.emplace_back(_reactors[(start + distance) % count]->queued_bytes(), distance);
  }
  std::sort(ranked.begin(), ranked.end());
  std::vector<cufile_reactor*> result;
  result.reserve(count);
  for (auto const& [bytes, distance] : ranked) {
    result.push_back(_reactors[(start + distance) % count].get());
  }
  return result;
}

std::shared_ptr<engine> make_cufile_reactor_engine(engine_options const& options)
{
  cufile_reactor_options const defaults;
  auto const* typed  = dynamic_cast<cufile_reactor_options const*>(&options);
  auto const& chosen = typed != nullptr ? *typed : defaults;
  if (chosen.n_reactors == 0) {
    throw std::invalid_argument("the cufile_reactor engine needs at least one reactor");
  }
  return std::make_shared<cufile_reactor_engine>(
    chosen.n_reactors, std::make_shared<cufile_reactor::reactor_context>(chosen.cfg));
}

}  // namespace kvikio::io::cufile

KVIKIO_DEFINE_ENGINE_PLUGIN("cufile_reactor", &kvikio::io::cufile::make_cufile_reactor_engine)
