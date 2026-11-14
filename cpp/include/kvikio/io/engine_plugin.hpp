/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file engine_plugin.hpp
 * @brief What a shared library exports to provide an engine.
 *
 * An engine plugin is a shared library named `libkvikio_engine_<name>.so` that
 * exports one C function, @c kvikio_engine_plugin. The function returns a
 * descriptor that starts with a plain C header. KvikIO checks the header first
 * and touches the C++ part (the factory) only if the header matches its own
 * build.
 *
 * A plugin is built together with the KvikIO it is loaded into: there is no
 * binary compatibility across KvikIO versions or build options. The header
 * check turns a mismatch into a clear error instead of undefined behaviour.
 *
 * A plugin links against `libkvikio.so` and uses its logger, CUDA function
 * table and the types in `kvikio/io/` and `kvikio/exec/`.
 */

#include <kvikio/io/engine.hpp>
#include <kvikio/shim/utils.hpp>
#include <kvikio/version_config.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

/// Bumped whenever the engine interface or the descriptor changes shape.
#define KVIKIO_ENGINE_PLUGIN_ABI_VERSION 1u

/// Build options that change the layout of types crossing the plugin boundary.
#ifdef KVIKIO_USE_ABSEIL_INVOCABLE
#define KVIKIO_ENGINE_PLUGIN_BUILD_FLAGS 1u
#else
#define KVIKIO_ENGINE_PLUGIN_BUILD_FLAGS 0u
#endif

/// Name of the function a plugin exports.
#define KVIKIO_ENGINE_PLUGIN_SYMBOL "kvikio_engine_plugin"

extern "C" {

/// The part of a plugin descriptor that is read before anything else.
struct kvikio_engine_plugin_header {
  std::uint32_t abi_version;   ///< @c KVIKIO_ENGINE_PLUGIN_ABI_VERSION of the plugin's build
  std::uint32_t build_flags;   ///< @c KVIKIO_ENGINE_PLUGIN_BUILD_FLAGS of the plugin's build
  const char* kvikio_version;  ///< @c KvikIO_VERSION of the plugin's build
  const char* engine_name;     ///< the engine this plugin provides
};

}  // extern "C"

namespace kvikio::io {

/// What @c kvikio_engine_plugin returns. @c create is valid only when
/// @c header matches the loading KvikIO.
struct engine_plugin_descriptor {
  kvikio_engine_plugin_header header;
  std::shared_ptr<engine> (*create)(engine_options const&);
};

/// File name of the plugin providing engine @p name.
[[nodiscard]] std::string engine_plugin_file_name(std::string_view name);

}  // namespace kvikio::io

/**
 * @brief Define the exported entry point of an engine plugin.
 *
 * @param NAME      the engine name, a string literal of `[a-z0-9_]+`
 * @param CREATE_FN a function `std::shared_ptr<kvikio::io::engine>(kvikio::io::engine_options
 * const&)`
 */
#define KVIKIO_DEFINE_ENGINE_PLUGIN(NAME, CREATE_FN)                                              \
  extern "C" KVIKIO_EXPORT const ::kvikio::io::engine_plugin_descriptor* kvikio_engine_plugin()   \
  {                                                                                               \
    static const ::kvikio::io::engine_plugin_descriptor descriptor{                               \
      {KVIKIO_ENGINE_PLUGIN_ABI_VERSION, KVIKIO_ENGINE_PLUGIN_BUILD_FLAGS, KvikIO_VERSION, NAME}, \
      CREATE_FN};                                                                                 \
    return &descriptor;                                                                           \
  }
