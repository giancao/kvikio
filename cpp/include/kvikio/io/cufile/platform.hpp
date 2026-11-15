/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file platform.hpp
 * @brief What the cuFile library of this process can do.
 */

#include <kvikio/shim/utils.hpp>

#include <cstddef>

namespace kvikio::io::cufile {

/**
 * @brief What the process's cuFile library reports about this host.
 *
 * cuFile falls back to a POSIX "compatibility mode" silently whenever no real
 * GPUDirect Storage path (nvidia-fs, PCIe P2PDMA, C2C, an RDMA filesystem client)
 * is active.  Reads stay correct there, but they bounce through host memory
 * inside cuFile, so a caller choosing between the cuFile and io_uring backends
 * needs to know which mode it would get.
 *
 * This is a process-wide snapshot of the library and driver; it says nothing
 * about a particular file (see @ref probe_path for that).
 */
struct platform_info {
  bool library_available{false};  ///< libcufile could be loaded.
  bool driver_opened{false};      ///< @c cuFileDriverOpen succeeded.
  /// @c cuFileDriverGetProperties succeeded, so @c dstatusflags, @c compat_only
  /// and @c max_direct_io_size_kb are meaningful.  False with an open driver
  /// means the mode is unknown, not that it is compatibility-only.
  bool properties_available{false};
  /// No storage class has a real GDS mode: @c dstatusflags is zero (true too
  /// when the properties are unavailable).  The probe never calls
  /// @c cuFileDriverGetP2PFlags, which resets cuFile's running parameters in
  /// libcufile 1.19.
  bool compat_only{true};
  /// Compatibility mode is forced by @c CUFILE_FORCE_COMPAT_MODE: a value
  /// beginning with true (any case), as cuFile reads it, whatever the driver reports.
  bool forced_compat{false};
  int version{0};                        ///< @c cuFileGetVersion, e.g. 1170; 0 when unknown.
  unsigned dstatusflags{0};              ///< Bit per storage class with a real GDS mode.
  std::size_t max_direct_io_size_kb{0};  ///< cuFile's largest single DMA, in KiB.
};

/**
 * @brief Probe the cuFile library once per process and return the result.
 *
 * The first call loads libcufile, opens the cuFile driver (the same process-wide
 * open the cuFile reactor performs; it costs a few hundred milliseconds and lets
 * cuFile's thread pool retain a CUDA context on each visible GPU) and reads
 * @c cuFileDriverGetProperties; the effective mode is logged once.  Later calls
 * return the cached result.  Thread-safe; never throws.
 */
[[nodiscard]] KVIKIO_EXPORT platform_info const& probe_platform() noexcept;

/**
 * @brief Whether a real GPUDirect Storage path is usable on this host.
 *
 * True iff libcufile loaded, the driver opened and reported its properties, at
 * least one storage class reports a real GDS mode and the environment does not
 * force compatibility mode.  A specific file can still end up in compatibility
 * mode (an unsupported filesystem such as tmpfs or btrfs);
 * @c cufile_io_object::cufile_registered() only says the handle registered, not
 * which path cuFile takes for it -- see @ref probe_path.
 */
[[nodiscard]] KVIKIO_EXPORT bool gds_available() noexcept;

namespace detail {

/// Whether @p value, the value of `CUFILE_FORCE_COMPAT_MODE`, forces compatibility mode.
[[nodiscard]] KVIKIO_EXPORT bool force_compat_env_on(char const* value) noexcept;

}  // namespace detail

}  // namespace kvikio::io::cufile
