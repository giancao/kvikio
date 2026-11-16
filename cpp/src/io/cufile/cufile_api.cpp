/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <kvikio/io/cufile/cufile_api.hpp>
#include <kvikio/io/detail/log.hpp>

#include <dlfcn.h>

#include <optional>
#include <tuple>

namespace kvikio::io::cufile {

namespace {

/// The soname every libcufile since 1.0 ships.
constexpr char const* cufile_soname = "libcufile.so.0";

/// `cuFileGetVersion` is not required for I/O; it is resolved separately.
using get_version_fn = decltype(&cuFileGetVersion);

/// Resolve @p name from @p library into @p out; whether it resolved.
template <class Fn>
bool resolve(Fn& out, void* library, char const* name) noexcept
{
  void* const symbol = ::dlsym(library, name);
  // POSIX guarantees the object-pointer to function-pointer conversion of a dlsym result.
  out = reinterpret_cast<Fn>(symbol);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
  return symbol != nullptr;
}

struct loaded_library {
  std::optional<cufile_api> api;
  get_version_fn get_version{nullptr};
};

[[nodiscard]] loaded_library load() noexcept
{
  loaded_library loaded;
  void* const library = ::dlopen(cufile_soname, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  if (library == nullptr) {
    char const* why = ::dlerror();
    KVIKIO_IO_LOG_INFO("cufile: dlopen(%s) failed: %s; cuFile unavailable",
                       cufile_soname,
                       why != nullptr ? why : "unknown error");
    return loaded;
  }

  cufile_api api{};
  bool complete = true;
  auto need     = [&](auto& member, char const* name) noexcept {
    if (!resolve(member, library, name)) {
      complete = false;
      KVIKIO_IO_LOG_INFO("cufile: required symbol %s not found; cuFile unavailable", name);
    }
  };
  need(api.driver_open, "cuFileDriverOpen");
  need(api.driver_get_properties, "cuFileDriverGetProperties");
  need(api.handle_register, "cuFileHandleRegister");
  need(api.handle_deregister, "cuFileHandleDeregister");
  need(api.buf_register, "cuFileBufRegister");
  need(api.buf_deregister, "cuFileBufDeregister");
  need(api.read, "cuFileRead");
  need(api.write, "cuFileWrite");
  if (!complete) { return loaded; }

  loaded.api  = api;
  std::ignore = resolve(loaded.get_version, library, "cuFileGetVersion");
  return loaded;
}

[[nodiscard]] loaded_library const& library() noexcept
{
  static loaded_library const loaded = load();
  return loaded;
}

}  // namespace

cufile_api const* cufile_api::get() noexcept
{
  auto const& loaded = library();
  return loaded.api.has_value() ? &*loaded.api : nullptr;
}

bool cufile_api::open_driver() noexcept
{
  static bool const opened = [] {
    auto const* api = get();
    if (api == nullptr) { return false; }
    auto const status = api->driver_open();
    if (status.err != CU_FILE_SUCCESS) {
      KVIKIO_IO_LOG_WARN("cufile: cuFileDriverOpen failed: %s (%d); GDS fast path disabled",
                         cufileop_status_error(status.err),
                         static_cast<int>(status.err));
      return false;
    }
    return true;
  }();
  return opened;
}

int cufile_api::version() noexcept
{
  auto const& loaded = library();
  if (!loaded.api.has_value() || loaded.get_version == nullptr) { return 0; }
  int value = 0;
  if (loaded.get_version(&value).err != CU_FILE_SUCCESS) { return 0; }
  return value;
}

}  // namespace kvikio::io::cufile
