/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <exception>

#include <cufile.h>
#include <kvikio/io/cufile/cufile_api.hpp>
#include <kvikio/io/cufile/platform.hpp>
#include <kvikio/io/detail/log.hpp>

namespace kvikio::io::cufile {

namespace detail {

bool force_compat_env_on(char const* value) noexcept
{
  if (value == nullptr) { return false; }
  constexpr char upper[] = "TRUE";
  for (int i = 0; i < 4; ++i) {
    // A null terminator cannot match a letter, so short values stop here.
    if (value[i] != upper[i] && value[i] != upper[i] + ('a' - 'A')) { return false; }
  }
  return true;
}

}  // namespace detail

namespace {

[[nodiscard]] platform_info compute_platform() noexcept
{
  platform_info info{};
  info.forced_compat = detail::force_compat_env_on(std::getenv("CUFILE_FORCE_COMPAT_MODE"));
  try {
    auto const* api        = cufile_api::get();
    info.library_available = api != nullptr;
    if (!info.library_available) return info;
    info.version = cufile_api::version();
    if (!cufile_api::open_driver()) return info;
    info.driver_opened = true;
    CUfileDrvProps_t props{};
    if (api->driver_get_properties(&props).err != CU_FILE_SUCCESS) {
      KVIKIO_IO_LOG_WARN("cufile: cuFileDriverGetProperties failed; the GDS mode is unknown");
      return info;
    }
    info.properties_available  = true;
    info.dstatusflags          = props.nvfs.dstatusflags;
    info.compat_only           = info.dstatusflags == 0;
    info.max_direct_io_size_kb = props.nvfs.max_direct_io_size;
  } catch (std::exception const& e) {
    KVIKIO_IO_LOG_WARN("cufile: probing the cuFile library failed: %s", e.what());
  } catch (...) {
    KVIKIO_IO_LOG_WARN("cufile: probing the cuFile library failed with an unknown error");
  }
  return info;
}

}  // namespace

platform_info const& probe_platform() noexcept
{
  static platform_info const info = [] {
    auto result = compute_platform();
    if (!result.driver_opened) {
      KVIKIO_IO_LOG_INFO("cufile: cuFile unavailable (library %s, driver not opened)",
                         result.library_available ? "loaded" : "missing");
    } else if (!result.properties_available) {
      KVIKIO_IO_LOG_INFO("cufile: cuFile %d driver opened, but its mode is unknown",
                         result.version);
    } else if (result.compat_only || result.forced_compat) {
      KVIKIO_IO_LOG_INFO(
        "cufile: cuFile %d runs in COMPATIBILITY mode (%s): reads bounce through host memory "
        "inside cuFile",
        result.version,
        result.forced_compat ? "forced by the environment" : "no storage class reports GDS");
    } else {
      KVIKIO_IO_LOG_INFO("cufile: cuFile %d reports GPUDirect Storage (dstatusflags %#x)",
                         result.version,
                         result.dstatusflags);
    }
    return result;
  }();
  return info;
}

bool gds_available() noexcept
{
  auto const& info = probe_platform();
  return info.driver_opened && info.properties_available && !info.compat_only &&
         !info.forced_compat;
}

// ---------------------------------------------------------------------------
// per-path capability
// ---------------------------------------------------------------------------

}  // namespace kvikio::io::cufile
