/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <kvikio/io/cufile_posix_engine.hpp>
#include <kvikio/io/curl_engine.hpp>
#include <kvikio/io/engine_plugin.hpp>
#include <kvikio/io/engine_registry.hpp>
#include <kvikio/logger.hpp>

namespace kvikio::io {

namespace {

/// Engine names become part of a file name, so only `[a-z0-9_]+` is accepted.
bool valid_engine_name(std::string_view name) noexcept
{
  return !name.empty() && std::all_of(name.begin(), name.end(), [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
  });
}

std::string_view route_name(engine_route route) noexcept
{
  return route == engine_route::local ? "local" : "s3";
}

/// Directory of the shared library this function lives in; empty if unknown.
std::string library_directory()
{
  Dl_info info{};
  if (::dladdr(reinterpret_cast<void const*>(&engine_plugin_file_name), &info) == 0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  std::string path{info.dli_fname};
  auto const slash = path.rfind('/');
  return slash == std::string::npos ? std::string{"."} : path.substr(0, slash);
}

void append_path_list(std::vector<std::string>& out, std::string_view list)
{
  while (!list.empty()) {
    auto const colon = list.find(':');
    auto const item  = list.substr(0, colon);
    if (!item.empty()) { out.emplace_back(item); }
    if (colon == std::string_view::npos) { break; }
    list.remove_prefix(colon + 1);
  }
}

/// Checks the C header of a plugin against this build. Returns an empty string
/// when it matches, otherwise the reason it does not.
std::string header_mismatch(kvikio_engine_plugin_header const& header, std::string_view name)
{
  if (header.abi_version != KVIKIO_ENGINE_PLUGIN_ABI_VERSION) {
    return "it was built for engine ABI " + std::to_string(header.abi_version) +
           ", this KvikIO has " + std::to_string(KVIKIO_ENGINE_PLUGIN_ABI_VERSION);
  }
  if (header.build_flags != KVIKIO_ENGINE_PLUGIN_BUILD_FLAGS) {
    return "it was built with different KvikIO build options (flags " +
           std::to_string(header.build_flags) + ", this KvikIO has " +
           std::to_string(KVIKIO_ENGINE_PLUGIN_BUILD_FLAGS) + ")";
  }
  if (header.kvikio_version == nullptr || std::strcmp(header.kvikio_version, KvikIO_VERSION) != 0) {
    return std::string("it was built for KvikIO ") +
           (header.kvikio_version == nullptr ? "(unknown)" : header.kvikio_version) +
           ", this is KvikIO " + KvikIO_VERSION;
  }
  if (header.engine_name == nullptr || name != header.engine_name) {
    return std::string("it provides engine '") +
           (header.engine_name == nullptr ? "" : header.engine_name) + "'";
  }
  return {};
}

}  // namespace

std::string engine_plugin_file_name(std::string_view name)
{
  return "libkvikio_engine_" + std::string(name) + ".so";
}

engine_registry::engine_registry()
{
  // The engines built into libkvikio.so are available in every registry.
  _factories.emplace(cufile_posix_engine_name, &make_cufile_posix_engine);
#ifdef KVIKIO_LIBCURL_FOUND
  _factories.emplace(curl_engine_name, &make_curl_engine);
#endif
}

engine_registry::~engine_registry() { shutdown(); }

engine_registry& engine_registry::instance()
{
  // Never destroyed: engines own threads and may come from plugins, so tearing
  // them down during static destruction would race with the rest of process
  // exit. Whoever selected an engine shuts it down.
  static auto* registry = new engine_registry();
  return *registry;
}

void engine_registry::register_factory(std::string name, engine_factory factory)
{
  if (!valid_engine_name(name)) {
    throw std::invalid_argument("engine name '" + name + "' is not [a-z0-9_]+");
  }
  if (!factory) { throw std::invalid_argument("engine '" + name + "' has an empty factory"); }
  std::lock_guard lock(_mutex);
  if (!_factories.emplace(std::move(name), std::move(factory)).second) {
    throw std::invalid_argument("an engine of this name is already registered");
  }
}

bool engine_registry::has_factory(std::string_view name) const
{
  std::lock_guard lock(_mutex);
  return _factories.find(std::string(name)) != _factories.end();
}

void engine_registry::add_plugin_search_path(std::string directory)
{
  std::lock_guard lock(_mutex);
  _search_path.push_back(std::move(directory));
}

std::vector<std::string> engine_registry::plugin_search_path() const
{
  std::vector<std::string> result;
  {
    std::lock_guard lock(_mutex);
    result = _search_path;
  }
  if (char const* env = std::getenv("KVIKIO_ENGINE_PLUGIN_PATH"); env != nullptr) {
    append_path_list(result, env);
  }
  if (auto dir = library_directory(); !dir.empty()) { result.push_back(std::move(dir)); }
  return result;
}

engine_factory engine_registry::load_plugin(std::string_view name) const
{
  auto const file_name = engine_plugin_file_name(name);
  auto const search    = plugin_search_path();

  std::string path;
  for (auto const& directory : search) {
    auto candidate = directory + "/" + file_name;
    if (::access(candidate.c_str(), R_OK) == 0) {
      path = std::move(candidate);
      break;
    }
  }
  if (path.empty()) {
    std::string searched;
    for (auto const& directory : search) {
      searched += searched.empty() ? "" : ":";
      searched += directory;
    }
    throw engine_selection_error("it is not built in and no plugin " + file_name +
                                 " was found in " + searched);
  }

  // The handle is never closed: the engine's code lives in the plugin.
  void* handle = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    char const* reason = ::dlerror();
    throw engine_selection_error("cannot load plugin " + path + ": " +
                                 (reason == nullptr ? "unknown error" : reason));
  }
  auto reject = [&](std::string const& why) -> engine_selection_error {
    ::dlclose(handle);
    return engine_selection_error("plugin " + path + " rejected: " + why);
  };

  using entry_fn = engine_plugin_descriptor const* (*)();
  auto* entry    = reinterpret_cast<entry_fn>(::dlsym(handle, KVIKIO_ENGINE_PLUGIN_SYMBOL));
  if (entry == nullptr) {
    throw reject(std::string("it does not export ") + KVIKIO_ENGINE_PLUGIN_SYMBOL);
  }
  auto const* descriptor = entry();
  if (descriptor == nullptr) { throw reject("it returned no descriptor"); }
  // Only the C header is read until it is known to match this build.
  if (auto why = header_mismatch(descriptor->header, name); !why.empty()) { throw reject(why); }
  if (descriptor->create == nullptr) { throw reject("it has no factory"); }
  return engine_factory{descriptor->create};
}

engine_factory engine_registry::factory_for(std::string_view name)
{
  if (!valid_engine_name(name)) { throw engine_selection_error("the name is not [a-z0-9_]+"); }
  std::string key{name};
  {
    std::lock_guard lock(_mutex);
    if (auto it = _factories.find(key); it != _factories.end()) { return it->second; }
  }
  // Loading runs the plugin's initializers, so it happens without the lock.
  auto loaded = load_plugin(name);
  std::lock_guard lock(_mutex);
  return _factories.emplace(std::move(key), std::move(loaded)).first->second;
}

namespace {

/// What() of the exception being handled.
std::string current_reason() noexcept
{
  try {
    throw;
  } catch (std::exception const& e) {
    try {
      return e.what();
    } catch (...) {
    }
  } catch (...) {
  }
  return "unknown error";
}

}  // namespace

std::shared_ptr<engine> engine_registry::make_started(std::string_view name,
                                                      engine_options const& options)
{
  // The factory and start() are foreign code: called without the lock.
  auto const factory = factory_for(name);
  auto created       = factory(options);
  if (created == nullptr) { throw engine_selection_error("its factory returned no engine"); }
  created->start();
  return created;
}

std::shared_ptr<engine> engine_registry::create(std::string_view name,
                                                engine_options const& options)
{
  try {
    return make_started(name, options);
  } catch (...) {
    std::string message = "cannot create engine '" + std::string(name) + "': " + current_reason();
    KVIKIO_LOG_ERROR(message);
    throw engine_selection_error(message);
  }
}

std::shared_ptr<engine> engine_registry::select(engine_route route,
                                                std::string_view name,
                                                engine_options const& options)
{
  auto const index = static_cast<std::size_t>(route);
  try {
    auto created = make_started(name, options);
    std::shared_ptr<engine> previous;
    {
      std::lock_guard lock(_mutex);
      previous = std::exchange(_selected[index], created);
      _current[index].store(created.get(), std::memory_order_release);
    }
    // `previous` is released here, outside the lock: if this was its last
    // owner, its destructor stops its threads.
    return created;
  } catch (...) {
    auto const reason   = current_reason();
    auto const previous = selected_name(route);
    std::string message =
      "cannot select engine '" + std::string(name) + "' for " + std::string(route_name(route)) +
      " paths: " + reason + "; " +
      (previous.empty() ? std::string("no engine is selected") : "still using '" + previous + "'");
    KVIKIO_LOG_ERROR(message);
    throw engine_selection_error(message);
  }
}

std::shared_ptr<engine> engine_registry::acquire(engine_route route) const
{
  std::lock_guard lock(_mutex);
  return _selected[static_cast<std::size_t>(route)];
}

std::string engine_registry::selected_name(engine_route route) const
{
  std::lock_guard lock(_mutex);
  auto const& selected = _selected[static_cast<std::size_t>(route)];
  return selected == nullptr ? std::string{} : std::string(selected->name());
}

void engine_registry::release(engine const* e) noexcept
{
  if (e == nullptr) { return; }
  std::array<std::shared_ptr<engine>, engine_route_count> dropped;
  {
    std::lock_guard lock(_mutex);
    for (std::size_t i = 0; i < engine_route_count; ++i) {
      if (_selected[i].get() != e) { continue; }
      _current[i].store(nullptr, std::memory_order_release);
      dropped[i].swap(_selected[i]);
    }
  }
  // `dropped` is released without the lock: it may be the last reference.
}

void engine_registry::shutdown() noexcept
{
  std::array<std::shared_ptr<engine>, engine_route_count> engines;
  {
    std::lock_guard lock(_mutex);
    for (auto& current : _current) {
      current.store(nullptr, std::memory_order_release);
    }
    engines.swap(_selected);
  }
  // Engines are shut down without the lock: shutdown joins threads.
  for (auto const& e : engines) {
    if (e != nullptr) { e->shutdown(); }
  }
}

}  // namespace kvikio::io
