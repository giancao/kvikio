/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file reactor_engine.hpp
 * @brief An engine made of a pool of reactors.
 */

#include <kvikio/io/detail/reactor_dispatch.hpp>
#include <kvikio/io/engine.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace kvikio::io::detail {

/**
 * @brief The @ref kvikio::io::engine interface in front of a pool of @p Reactor.
 *
 * Requests go through @ref reactor_dispatch. The capabilities are the
 * reactor's traits and the scheduling hints of its configuration; a reactor
 * engine has request priorities.
 *
 * A backend derives from this class to open objects differently or to route
 * requests (@ref next_reactor).
 */
template <io_reactor_c Reactor>
class reactor_engine : public engine, public reactor_dispatch<Reactor, reactor_engine<Reactor>> {
  using dispatch_type = reactor_dispatch<Reactor, reactor_engine<Reactor>>;
  friend dispatch_type;

 public:
  using reactor_type           = Reactor;
  using io_object_type         = typename Reactor::io_object_type;
  using reactor_traits_t       = reactor_traits<Reactor>;
  using reactor_write_traits_t = reactor_write_traits<Reactor>;
  using io_op_type             = typename dispatch_type::io_op_type;

  /// @p factory is called @p n_reactors times, each call returning one reactor.
  template <class Factory>
    requires std::invocable<Factory&> &&
             std::convertible_to<std::invoke_result_t<Factory&>, std::unique_ptr<Reactor>>
  reactor_engine(std::string name, std::size_t n_reactors, Factory factory)
    : dispatch_type(n_reactors, std::move(factory)), _name(std::move(name))
  {
    _caps.device_read            = reactor_traits_t::supports_device_read;
    _caps.host_to_device_read    = reactor_traits_t::supports_host_to_device_read;
    _caps.vector_host_read       = reactor_traits_t::supports_vector_host_read;
    _caps.device_range_read      = reactor_traits_t::supports_device_range_read;
    _caps.fragmented_host_buffer = reactor_traits_t::supports_vector_host_read ||
                                   reactor_traits_t::supports_host_to_device_read;
    _caps.write                = reactor_write_traits_t::supports_write;
    _caps.device_write         = reactor_write_traits_t::supports_device_write;
    _caps.priority             = true;
    _caps.prefers_bulk_io      = reactor_traits_t::prefers_bulk_io;
    _caps.min_alignment        = this->_config.min_alignment_requirement();
    _caps.merge_gap            = this->_config.merge_gap_size();
    _caps.max_concurrent_scans = this->_config.n_max_concurrent_scans;
    refresh_staging_block_size();
  }

  ~reactor_engine() override { this->shutdown_reactors(); }

  [[nodiscard]] std::string_view name() const noexcept override { return _name; }

  [[nodiscard]] engine_capabilities const& capabilities() const noexcept override { return _caps; }

  [[nodiscard]] bool supports(std::string_view path) const noexcept override
  {
    return Reactor::supports(path);
  }

  void start() override
  {
    this->start_reactors();
    // A reactor may size its staging when it starts.
    refresh_staging_block_size();
  }

  void shutdown() noexcept override { this->shutdown_reactors(); }

  [[nodiscard]] std::vector<byte_range> align_and_coalesce(
    std::span<byte_range const> ranges, std::optional<std::size_t> alignment) const noexcept override
  {
    return Reactor::align_and_coalesce(ranges, alignment);
  }

  /// Opens through the reactor, which takes no open options.
  [[nodiscard]] std::shared_ptr<io_object> open(std::string path,
                                                open_options const& /*options*/) override
  {
    return std::shared_ptr<io_object>(Reactor::create_io_object(std::move(path)));
  }

  [[nodiscard]] std::shared_ptr<io_object> open_for_write(
    std::string path, write_open_options const& options) override
  {
    if constexpr (!reactor_write_traits_t::supports_write) {
      return engine::open_for_write(std::move(path), options);
    } else {
      return std::shared_ptr<io_object>(
        Reactor::create_io_object_for_write(std::move(path), options));
    }
  }

  /// The reactors a request is split across; the least-backlogged ones by default.
  virtual std::vector<Reactor*> next_reactor([[maybe_unused]] io_object_type const& object,
                                             [[maybe_unused]] std::size_t n_slices,
                                             [[maybe_unused]] io_op_type type,
                                             [[maybe_unused]] int device_id = -1)
  {
    return this->default_next_reactor();
  }

  std::size_t host_read(io_object const& object,
                        std::size_t offset,
                        std::size_t size,
                        std::uint8_t* dst) override
  {
    return this->dispatch_host_read(object, offset, size, dst);
  }

  [[nodiscard]] exec::semi_future<std::size_t> read(
    io_object const& object, std::vector<prepared_io_slice>&& slices) noexcept override
  {
    return this->dispatch_read(object, std::move(slices));
  }

  std::size_t host_write(io_object const& object,
                         std::size_t offset,
                         std::size_t size,
                         std::uint8_t const* src) override
  {
    if constexpr (!reactor_write_traits_t::supports_write) {
      return engine::host_write(object, offset, size, src);
    } else {
      return this->dispatch_host_write(object, offset, size, src);
    }
  }

  [[nodiscard]] exec::semi_future<std::size_t> write(
    io_object const& object, std::vector<prepared_io_slice>&& slices) noexcept override
  {
    if constexpr (!reactor_write_traits_t::supports_write) {
      return engine::write(object, std::move(slices));
    } else {
      return this->dispatch_write(object, std::move(slices));
    }
  }

  void flush(io_object const& object) override
  {
    if constexpr (!reactor_write_traits_t::supports_write) {
      engine::flush(object);
    } else {
      this->dispatch_flush(object);
    }
  }

  [[nodiscard]] exec::semi_future<void> flush_async(io_object const& object,
                                                    io_priority priority) noexcept override
  {
    if constexpr (!reactor_write_traits_t::supports_write) {
      return engine::flush_async(object, priority);
    } else {
      return this->dispatch_flush_async(object, priority);
    }
  }

 protected:
  /// Called when a request with a device slice fails before it is queued.
  virtual void on_device_dispatch_failure() noexcept {}

 private:
  void refresh_staging_block_size() noexcept
  {
    _caps.staging_block_size =
      this->_reactors.empty() ? 0 : this->_reactors.front()->staging_block_size();
  }

  std::string _name;
  engine_capabilities _caps;
};

}  // namespace kvikio::io::detail
