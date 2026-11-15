/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file uring_engine.hpp
 * @brief The `uring` engine: local files through io_uring.
 *
 * It lives in the plugin `libkvikio_engine_uring.so`, not in `libkvikio.so`.
 * Select it by name through @c kvikio::io::engine_registry; this header is for
 * passing it options and for reading its diagnostics. Code that constructs
 * @ref kvikio::io::uring::uring_engine or a reactor directly links the plugin.
 *
 * Moved from cuCascade's `io/uring`. The reactor itself is unchanged apart from
 * where it gets staging memory (a @c staging_source), the CUDA calls (driver
 * API) and logging.
 */

#include <kvikio/io/detail/reactor_engine.hpp>
#include <kvikio/io/engine.hpp>
#include <kvikio/io/uring/config.hpp>
#include <kvikio/io/uring/uring_reactor.hpp>
#include <kvikio/shim/utils.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace kvikio::io::uring {

/// Name the engine is registered under.
inline constexpr char const* uring_engine_name = "uring";

/**
 * @brief Options of the `uring` engine.
 *
 * `staging` (inherited) is required: every reactor leases a fixed 64 MiB of it
 * when the engine starts.
 */
struct uring_options : engine_options {
  /// Number of reactors, each one thread with its own ring.
  std::size_t n_reactors{4};
  /// Per-reactor configuration.
  config cfg{};
};

/**
 * @brief Diagnostics of the `uring` engine.
 *
 * Reach it with `dynamic_cast<uring_diagnostics*>(engine)`; that needs this
 * header but not the plugin at link time.
 */
class uring_diagnostics {
 public:
  virtual ~uring_diagnostics() = default;

  /// One gauge snapshot per reactor, in reactor order. Each call restarts the
  /// reactors' max-inflight and queue-delay windows. Nothing samples them in
  /// the background; a caller that wants them polls this.
  [[nodiscard]] virtual std::vector<uring_reactor::gauges> reactor_gauges() noexcept = 0;
};

namespace detail {

/// What the reactors of an engine point at. A base class of the engine so that
/// it is destroyed after them.
struct uring_engine_resources {
  std::shared_ptr<staging_source> staging;
  std::shared_ptr<uring_reactor::reactor_context> context;
};

}  // namespace detail

class KVIKIO_EXPORT uring_engine final : private detail::uring_engine_resources,
                                         public kvikio::io::detail::reactor_engine<uring_reactor>,
                                         public uring_diagnostics {
 public:
  /// @p n_reactors reactors configured by @p cfg, staging through @p staging.
  uring_engine(std::size_t n_reactors, config const& cfg, std::shared_ptr<staging_source> staging);

  /// @p n_reactors reactors sharing @p context, whose staging source the
  /// caller keeps alive for as long as the engine.
  uring_engine(std::size_t n_reactors, std::shared_ptr<uring_reactor::reactor_context> context);

  ~uring_engine() override;

  [[nodiscard]] std::vector<uring_reactor::gauges> reactor_gauges() noexcept override;
};

/// Create the engine. A plain @c engine_options gets the defaults of
/// @ref uring_options with its staging source.
[[nodiscard]] KVIKIO_EXPORT std::shared_ptr<engine> make_uring_engine(
  engine_options const& options);

}  // namespace kvikio::io::uring
