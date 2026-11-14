/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <kvikio/io/engine.hpp>
#include <kvikio/io/errors.hpp>
#include <kvikio/io/staging_source.hpp>

namespace kvikio::io {

// Defined here, not inline: the vtables and type information of these
// interfaces then live in libkvikio.so, one copy for the library and every
// plugin.
staging_lease::~staging_lease()   = default;
staging_source::~staging_source() = default;
engine_options::~engine_options() = default;
engine::~engine()                 = default;

void engine::start() {}

void engine::warmup(std::string_view /*url*/) noexcept {}

std::shared_ptr<io_object> engine::open_for_write(std::string /*path*/,
                                                  write_open_options const& /*options*/)
{
  throw not_implemented_error(std::string(name()) + ": open_for_write");
}

std::size_t engine::host_write(io_object const& /*obj*/,
                               std::size_t /*offset*/,
                               std::size_t /*size*/,
                               std::uint8_t const* /*src*/)
{
  throw not_implemented_error(std::string(name()) + ": host_write");
}

exec::semi_future<std::size_t> engine::write(io_object const& /*obj*/,
                                             std::vector<prepared_io_slice>&& /*slices*/) noexcept
{
  try {
    throw not_implemented_error(std::string(name()) + ": write");
  } catch (...) {
    return exec::make_semi_future<std::size_t>(std::current_exception());
  }
}

void engine::flush(io_object const& /*obj*/)
{
  throw not_implemented_error(std::string(name()) + ": flush");
}

exec::semi_future<void> engine::flush_async(io_object const& /*obj*/,
                                            io_priority /*priority*/) noexcept
{
  try {
    throw not_implemented_error(std::string(name()) + ": flush_async");
  } catch (...) {
    return exec::make_semi_future<void>(exec::try_t<void>(std::current_exception()));
  }
}

}  // namespace kvikio::io
