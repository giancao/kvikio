/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file engine.hpp
 * @brief The interface every I/O engine implements.
 *
 * An engine is one I/O technology (cuFile and POSIX, io_uring, libcurl, ...)
 * with its reads and writes. Queueing, scheduling, the use of staging memory
 * and host-to-device copies all live inside the engine. Engines agree on this
 * contract instead of sharing code:
 *
 *  - **Completion.** When a returned future settles, the engine no longer
 *    touches any destination or source memory of that request, including the
 *    slices that failed or were cancelled. A device read settles only after its
 *    copy finished, so no stream synchronization is needed afterwards.
 *  - **Submission.** @ref engine::read, @ref engine::write and
 *    @ref engine::flush_async only queue the request and return; they do not
 *    block the calling thread. Everything a request needs is allocated before
 *    it is queued, so nothing after that point fails on allocation.
 *  - **Fragments.** For a fragmented host buffer the engine calls the slice's
 *    @c prepared_io_completion once per physical operation, with the handles of
 *    the fragments that operation filled.
 *  - **Memory.** Staging memory comes only from the injected
 *    @c staging_source.
 *  - **Capabilities.** @ref engine::capabilities says truthfully what the
 *    engine can do. An entry point it does not implement reports
 *    @c not_implemented_error; it never falls back to another path silently.
 */

#include <kvikio/exec/semi_future.hpp>
#include <kvikio/io/byte_range.hpp>
#include <kvikio/io/staging_source.hpp>
#include <kvikio/io/types.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace kvikio::io {

/**
 * @brief What an engine can do, and how the layer above should drive it.
 *
 * Constant for the lifetime of an engine instance.
 */
struct engine_capabilities {
  // -- What the engine can do ---------------------------------------------------

  /// Reads straight into device memory (e.g. GDS), without a host stage.
  bool device_read{false};
  /// Serves a slice that has both a host and a device destination: the bytes
  /// land in the host buffer and are copied to the device from there.
  bool host_to_device_read{false};
  /// Serves a batch of host reads in one request.
  bool vector_host_read{false};
  /// Serves a batch of device reads efficiently. An engine may still process
  /// mixed slices serially when this is false.
  bool device_range_read{false};
  /// Accepts a fragmented @c host_buffer and reports per physical operation
  /// through @c prepared_io_completion.
  bool fragmented_host_buffer{false};
  /// Implements @c open_for_write, the write entry points and the flushes.
  bool write{false};
  /// Accepts device-memory write sources. Implies @ref write.
  bool device_write{false};
  /// Distinguishes @c io_priority::high from @c io_priority::low.
  bool priority{false};

  // -- Scheduling hints ---------------------------------------------------------

  /// Would rather receive one batched request covering everything a reader
  /// needs than a stream of small reads. A preference, not a capability: for an
  /// object store a read is a round trip, for a local file it is a syscall.
  bool prefers_bulk_io{false};
  /// The smallest unit the engine can address, in bytes. A read is widened to a
  /// multiple of it before being issued.
  std::size_t min_alignment{1};
  /// Above 1 when the engine writes with O_DIRECT: a write must then start on a
  /// multiple of it and cover whole multiples, or the engine has to write the
  /// rest through the page cache. 1 when writes have no such requirement.
  std::size_t write_alignment{1};
  /// The largest gap between two ranges still worth bridging into one request,
  /// in bytes.
  std::size_t merge_gap{0};
  /// How many scans a readahead scheduler may keep in flight against this
  /// engine at once. Zero opts out of readahead scheduling.
  std::size_t max_concurrent_scans{0};
  /// Size of one staging block of this engine, in bytes; 0 if it does not stage.
  std::size_t staging_block_size{0};
};

/**
 * @brief What an engine is created with.
 *
 * An engine with its own settings derives from this struct; its factory
 * receives the base and casts to the type it expects.
 */
struct engine_options {
  engine_options()                                 = default;
  engine_options(engine_options const&)            = default;
  engine_options& operator=(engine_options const&) = default;
  virtual ~engine_options();

  /// Where the engine gets staging memory. May be null for an engine, or a use
  /// of it, that never stages.
  std::shared_ptr<staging_source> staging;
};

/**
 * @brief One I/O technology behind a common request interface.
 *
 * See the file comment for the contract.
 */
class engine {
 public:
  engine()                         = default;
  engine(engine const&)            = delete;
  engine& operator=(engine const&) = delete;
  virtual ~engine();

  /// The engine's registered name, e.g. "uring".
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;

  /// See @ref engine_capabilities.
  [[nodiscard]] virtual engine_capabilities const& capabilities() const noexcept = 0;

  /// Whether this engine can serve @p path: scheme, and any precondition of its
  /// own (e.g. that a local file exists).
  [[nodiscard]] virtual bool supports(std::string_view path) const noexcept = 0;

  /// Start the engine's threads. Throws if the technology is not usable in this
  /// process (e.g. io_uring is disabled).
  virtual void start();

  /// Stop the engine. Every future it returned has settled when this returns.
  /// Calling it again does nothing.
  virtual void shutdown() noexcept = 0;

  /// Prepare for a burst of requests to @p url (a hint that saves latency, e.g.
  /// opening connections). Never fails a later read. The default does nothing.
  virtual void warmup(std::string_view url) noexcept;

  /// Align each range's ends outward to the engine's I/O alignment and coalesce
  /// overlapping or adjacent results into sorted, non-overlapping ranges.
  /// @p alignment is a lower bound; unset means the engine's own alignment.
  [[nodiscard]] virtual std::vector<byte_range> align_and_coalesce(
    std::span<byte_range const> ranges, std::optional<std::size_t> alignment) const noexcept = 0;

  // -- Objects ------------------------------------------------------------------

  /// Open @p path for reading. Throws on unsupported or unreachable paths.
  [[nodiscard]] virtual std::shared_ptr<io_object> open(std::string path,
                                                        open_options const& options) = 0;

  /// Open @p path for reading and writing. The default throws
  /// @c not_implemented_error.
  [[nodiscard]] virtual std::shared_ptr<io_object> open_for_write(
    std::string path, write_open_options const& options);

  // -- Reads --------------------------------------------------------------------

  /// Read on the calling thread, without queueing. For small reads whose caller
  /// waits anyway (file metadata). Returns the bytes read.
  virtual std::size_t host_read(io_object const& obj,
                                std::size_t offset,
                                std::size_t size,
                                std::uint8_t* dst) = 0;

  /// Queue the read of @p slices and return. The future resolves to the total
  /// bytes read.
  [[nodiscard]] virtual exec::semi_future<std::size_t> read(
    io_object const& obj, std::vector<prepared_io_slice>&& slices) noexcept = 0;

  // -- Writes (defaults report not_implemented_error) -----------------------------

  /// Write on the calling thread. Returns the bytes written.
  virtual std::size_t host_write(io_object const& obj,
                                 std::size_t offset,
                                 std::size_t size,
                                 std::uint8_t const* src);

  /// Queue the write of @p slices (see @c make_host_write_slice and
  /// @c make_device_write_slice) and return. The future resolves to the total
  /// bytes written.
  [[nodiscard]] virtual exec::semi_future<std::size_t> write(
    io_object const& obj, std::vector<prepared_io_slice>&& slices) noexcept;

  /// Make every write whose future resolved before this call durable, on the
  /// calling thread.
  virtual void flush(io_object const& obj);

  /// Queue a flush and return.
  [[nodiscard]] virtual exec::semi_future<void> flush_async(io_object const& obj,
                                                            io_priority priority) noexcept;
};

/// Creates an engine from its options. May throw; the engine is not started.
using engine_factory = std::function<std::shared_ptr<engine>(engine_options const&)>;

}  // namespace kvikio::io
