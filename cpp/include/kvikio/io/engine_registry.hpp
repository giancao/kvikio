/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file engine_registry.hpp
 * @brief Which engine serves which kind of path.
 *
 * Most paths have exactly one engine that can serve them. A choice exists only
 * where several engines overlap; each such place is an @c engine_route. Every
 * route has one selected engine at a time.
 *
 * Selection happens when @ref engine_registry::select is called, never while
 * requests run: @c select creates and starts the engine, so an engine that is
 * not usable in this process (plugin not installed, io_uring disabled) is
 * reported there. A failed @c select throws and leaves the previous selection
 * in effect.
 */

#include <kvikio/io/engine.hpp>
#include <kvikio/io/errors.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace kvikio::io {

/// The kinds of path more than one engine can serve.
enum class engine_route : std::uint8_t {
  local,  ///< local files
  s3,     ///< `s3://` objects
};

/// Number of @ref engine_route values.
inline constexpr std::size_t engine_route_count = 2;

/// The engine KvikIO uses for @p route unless told otherwise. Both are built
/// into `libkvikio.so`, so they need no plugin.
[[nodiscard]] constexpr std::string_view default_engine_name(engine_route route) noexcept
{
  return route == engine_route::local ? std::string_view{"cufile_posix"} : std::string_view{"curl"};
}

/**
 * @brief Engine factories, plugin loading and the per-route selection.
 *
 * The process-wide instance is @ref instance. Separate instances are
 * independent of each other (tests use them).
 *
 * Thread-safe. @ref current is lock-free; everything else takes the registry's
 * mutex and belongs at startup, not on a request path. A component that issues
 * requests keeps the pointer it got from @ref select or @ref acquire.
 */
class engine_registry {
 public:
  engine_registry();
  engine_registry(engine_registry const&)            = delete;
  engine_registry& operator=(engine_registry const&) = delete;
  /// Shuts down the selected engines.
  ~engine_registry();

  /// The process-wide registry. It lives in `libkvikio.so`, so every module of
  /// the process sees the same one. It is never destroyed: whoever selected an
  /// engine shuts it down (directly, or with @ref shutdown).
  [[nodiscard]] static engine_registry& instance();

  /**
   * @brief Make engine @p name available without a plugin.
   *
   * @throws std::invalid_argument if @p name is not `[a-z0-9_]+`, or is already
   *         registered.
   */
  void register_factory(std::string name, engine_factory factory);

  /// Whether @p name has a factory, registered or loaded from a plugin.
  [[nodiscard]] bool has_factory(std::string_view name) const;

  /// Search @p directory for plugins, before the default locations.
  void add_plugin_search_path(std::string directory);

  /// The directories searched for `libkvikio_engine_<name>.so`, in order: the
  /// ones added with @ref add_plugin_search_path, the ones in the environment
  /// variable `KVIKIO_ENGINE_PLUGIN_PATH` (colon-separated), and the directory
  /// holding `libkvikio.so`.
  [[nodiscard]] std::vector<std::string> plugin_search_path() const;

  /**
   * @brief Use engine @p name for @p route.
   *
   * Finds the factory of @p name, loading its plugin if needed, creates the
   * engine with @p options and starts it. Only then does it become the
   * selection of @p route.
   *
   * The registry lets go of the engine selected before. Whoever still holds
   * it keeps using it and shuts it down; the registry does neither.
   *
   * @return the selected engine
   * @throws engine_selection_error if any step fails. The previous selection of
   *         @p route stays in effect.
   */
  std::shared_ptr<engine> select(engine_route route,
                                 std::string_view name,
                                 engine_options const& options = engine_options{});

  /**
   * @brief Create and start an engine without selecting it.
   *
   * For a component that wants an engine of its own (its own threads and
   * staging) next to the one selected for a route. The caller owns the engine
   * and shuts it down.
   *
   * @throws engine_selection_error if the name is unknown, its plugin cannot be
   *         loaded, or the engine cannot be created or started.
   */
  [[nodiscard]] std::shared_ptr<engine> create(std::string_view name,
                                               engine_options const& options = {});

  /// The engine selected for @p route, or null if none is. Lock-free. The
  /// pointer is valid until the next @ref select of this route or @ref shutdown;
  /// a component that outlives those takes @ref acquire instead.
  [[nodiscard]] engine* current(engine_route route) const noexcept
  {
    return _current[static_cast<std::size_t>(route)].load(std::memory_order_acquire);
  }

  /// The engine selected for @p route with shared ownership, or null.
  [[nodiscard]] std::shared_ptr<engine> acquire(engine_route route) const;

  /// Name of the engine selected for @p route; empty if none is.
  [[nodiscard]] std::string selected_name(engine_route route) const;

  /// Shut down and release the selected engines, and clear the selections.
  /// Factories and loaded plugins stay.
  void shutdown() noexcept;

 private:
  /// The factory of @p name, loading its plugin if it has none yet. Throws
  /// @c engine_selection_error.
  [[nodiscard]] engine_factory factory_for(std::string_view name);
  [[nodiscard]] std::shared_ptr<engine> make_started(std::string_view name,
                                                     engine_options const& options);

  /// Loads the plugin of @p name and returns its factory. Takes no lock while
  /// the plugin's code runs. Throws @c engine_selection_error.
  [[nodiscard]] engine_factory load_plugin(std::string_view name) const;

  mutable std::mutex _mutex;
  std::unordered_map<std::string, engine_factory> _factories;
  std::vector<std::string> _search_path;
  std::array<std::shared_ptr<engine>, engine_route_count> _selected;
  std::array<std::atomic<engine*>, engine_route_count> _current{};
};

}  // namespace kvikio::io
