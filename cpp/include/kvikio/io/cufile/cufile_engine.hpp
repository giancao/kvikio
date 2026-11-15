/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file cufile_engine.hpp
 * @brief The `cufile_reactor` engine: local files through cuFile / GPUDirect Storage.
 *
 * It lives in the plugin `libkvikio_engine_cufile_reactor.so`. Select it by name
 * through @c kvikio::io::engine_registry; this header is for passing it options.
 * Code that constructs the engine or a reactor directly links the plugin.
 *
 * Device destinations are read with `cuFileRead`, everything else through
 * POSIX. It cannot fill fragmented host buffers, so a prefetching cache cannot
 * route through it; its capabilities say so.
 *
 * Moved from cuCascade's `io/cufile`. The reactor loads libcufile itself, apart
 * from KvikIO's own loading of it.
 */

#include <kvikio/io/cufile/config.hpp>
#include <kvikio/io/cufile/cufile_options.hpp>
#include <kvikio/io/cufile/cufile_reactor.hpp>
#include <kvikio/io/detail/reactor_engine.hpp>
#include <kvikio/io/engine.hpp>
#include <kvikio/shim/utils.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace kvikio::io::cufile {

class KVIKIO_EXPORT cufile_reactor_engine final
  : public kvikio::io::detail::reactor_engine<cufile_reactor> {
 public:
  /// @p n_reactors reactors sharing @p context.
  cufile_reactor_engine(std::size_t n_reactors,
                        std::shared_ptr<cufile_reactor::reactor_context> context);

  ~cufile_reactor_engine() override;

  /// Opens with the engine's `use_odirect` and `compat_mode`.
  [[nodiscard]] std::shared_ptr<io_object> open(std::string path,
                                                open_options const& options) override;

  [[nodiscard]] std::shared_ptr<io_object> open_for_write(
    std::string path, write_open_options const& options) override;

  /**
   * @brief Reactor selection tuned to synchronous reactors.
   *
   * A reactor serves one transfer at a time, so a request's queue depth is the
   * number of reactors it is spread over. A vectored read with more than two
   * slices gets every reactor, least backlogged first. A host-only vectored
   * write goes to one reactor: its page-cache writes with early writeback keep
   * the device busy, and interleaving writers of one file only contends on its
   * inode. Other operations keep the default selection.
   */
  std::vector<cufile_reactor*> next_reactor(cufile_io_object const& object,
                                            std::size_t n_slices,
                                            io_op_type type,
                                            int device_id = -1) override;

  /// The configuration the reactors run with.
  [[nodiscard]] config const& configuration() const noexcept { return _config; }
};

/// Create the engine. A plain @c engine_options gets the defaults of
/// @ref cufile_reactor_options.
[[nodiscard]] KVIKIO_EXPORT std::shared_ptr<engine> make_cufile_reactor_engine(
  engine_options const& options);

}  // namespace kvikio::io::cufile
