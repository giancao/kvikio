/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file reactor_dispatch.hpp
 * @brief Splitting a request over a pool of reactors.
 *
 * A reactor is a worker thread that owns the physical I/O of the requests
 * queued on it. @ref kvikio::io::detail::reactor_dispatch is the part in front
 * of a pool of them, shared by every reactor-based backend: it validates and
 * clamps the slices of a request, resolves device ids and default streams on
 * the submitting thread, picks the least-backlogged reactors, creates the
 * request's coordinator and queues one entry per reactor.
 *
 * It is a mix-in: the class that owns it (an engine, or cuCascade's
 * `templated_ioctx`) derives from it and provides two hooks,
 * `next_reactor(object, n_slices, operation, device_id)` and
 * `on_device_dispatch_failure()`.
 *
 * Moved from cuCascade's `templated_ioctx.hpp`; the CUDA calls use the driver
 * API.
 */

#include <kvikio/error.hpp>
#include <kvikio/exec/semi_future.hpp>
#include <kvikio/io/byte_range.hpp>
#include <kvikio/io/detail/cuda_context.hpp>
#include <kvikio/io/detail/io_request.hpp>
#include <kvikio/io/detail/log.hpp>
#include <kvikio/io/types.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace kvikio::io::detail {

/// The device a device slice without a device id belongs to: the device of the
/// submitting thread's current context, or, when the thread has none, the
/// device that owns the slice's memory.
[[nodiscard]] inline int slice_device(prepared_io_slice const& slice)
{
  auto const current = current_device_ordinal();
  if (current >= 0) { return current; }
  return get_device_ordinal_from_pointer(convert_void2deviceptr(slice.d_buffer.data));
}

/**
 * @brief A non-blocking stream ordered after the work the calling thread had
 *        queued on its per-thread default stream.
 *
 * The per-thread default stream handle names the per-thread default stream of
 * whichever thread uses it, so a reactor worker given it would order the
 * transfer after its own (empty) per-thread stream instead of the caller's.
 * This stream is created on the submitting thread, waits there on an event
 * recorded on that thread's per-thread stream, and can then be used from any
 * thread.  Destroyed (asynchronously, per CUDA) when the last slice holding it
 * goes.
 */
class per_thread_stream_stand_in {
 public:
  explicit per_thread_stream_stand_in(int device) : _device(device), _ctx(primary_context(device))
  {
    scoped_context const guard{_ctx};
    auto const& cuda = cudaAPI::instance();
    KVIKIO_CUDA_DRIVER_TRY(cuda.StreamCreate(&_stream, CU_STREAM_NON_BLOCKING), std::runtime_error);
    CUevent ready{};
    auto status = cuda.EventCreate(&ready, CU_EVENT_DISABLE_TIMING);
    if (status == CUDA_SUCCESS) {
      // Resolved here, on the submitting thread: its own per-thread stream.
      status = cuda.EventRecord(ready, CU_STREAM_PER_THREAD);
      if (status == CUDA_SUCCESS) status = cuda.StreamWaitEvent(_stream, ready, 0);
      std::ignore = cuda.EventDestroy(ready);
    }
    if (status != CUDA_SUCCESS) {
      std::ignore = cuda.StreamDestroy(_stream);
      KVIKIO_CUDA_DRIVER_TRY(status, std::runtime_error);
    }
  }

  ~per_thread_stream_stand_in()
  {
    try {
      // Destroyed on whichever thread drops the last slice, usually a reactor.
      scoped_context const guard{_ctx, scoped_context::policy::adopt_if_none};
      std::ignore = cudaAPI::instance().StreamDestroy(_stream);
    } catch (...) {  // NOLINT(bugprone-empty-catch): nothing to do about a stream that cannot go
    }
  }

  per_thread_stream_stand_in(per_thread_stream_stand_in const&)            = delete;
  per_thread_stream_stand_in& operator=(per_thread_stream_stand_in const&) = delete;

  [[nodiscard]] CUstream get() const noexcept { return _stream; }
  [[nodiscard]] int device() const noexcept { return _device; }

 private:
  int _device;
  CUcontext _ctx;
  CUstream _stream{};
};

/**
 * @brief Resolve default streams on the submitting thread before any slice
 *        reaches a reactor. Device ids must be resolved.
 *
 * Null handles become the legacy default stream, independent of how any
 * translation unit was compiled. A caller built with per-thread default streams
 * is covered too: per-thread default streams are blocking streams, so legacy
 * work orders after them.
 *
 * The per-thread default stream is replaced with a stand-in ordered after the
 * submitting thread's per-thread default stream (one per device).
 */
inline void resolve_default_streams(std::vector<prepared_io_slice>& slices)
{
  std::vector<std::shared_ptr<per_thread_stream_stand_in>> stand_ins;
  for (auto& slice : slices) {
    if (!slice.has_device_request()) continue;
    if (slice.d_buffer.stream == nullptr) {
      slice.d_buffer.stream = CU_STREAM_LEGACY;
      continue;
    }
    if (slice.d_buffer.stream != CU_STREAM_PER_THREAD) continue;
    auto const device =
      slice.d_buffer.device_id >= 0 ? slice.d_buffer.device_id : slice_device(slice);
    std::shared_ptr<per_thread_stream_stand_in> stand_in;
    for (auto const& existing : stand_ins) {
      if (existing->device() == device) stand_in = existing;
    }
    if (stand_in == nullptr) {
      stand_in = std::make_shared<per_thread_stream_stand_in>(device);
      stand_ins.push_back(stand_in);
    }
    slice.d_buffer.stream       = stand_in->get();
    slice.d_buffer.stream_owner = stand_in;
  }
}

/**
 * @brief Contract implemented by worker-driven reactors.
 *
 * Logical prepared slices are the only asynchronous input. A reactor owns all
 * backend chunking, alignment, staging allocation, and physical-operation
 * lifecycle after enqueue.
 */
template <class R>
concept io_reactor_c = requires(R reactor,
                                typename R::io_object_type object,
                                std::unique_ptr<grouped_io_request> request,
                                std::size_t offset,
                                std::size_t size,
                                std::uint8_t* destination,
                                std::string path,
                                std::string_view path_view)
{
  typename R::io_object_type;
  typename R::reactor_config_type;

  {reactor.get_config()}->std::same_as<typename R::reactor_config_type const&>;
  {
    reactor.enqueue(std::move(request))
  } noexcept;
  {
    reactor.queued_bytes()
  } noexcept -> std::convertible_to<std::size_t>;
  {
    reactor.staging_block_size()
  } noexcept -> std::convertible_to<std::size_t>;
  {reactor.host_read(object, offset, size, destination)}->std::same_as<std::size_t>;
  {reactor.start()};
  {reactor.shutdown()};
  {reactor.interrupt()};
  {R::create_io_object(std::move(path))}->std::same_as<std::unique_ptr<typename R::io_object_type>>;
  {R::supports(path_view)}->std::same_as<bool>;
  {R::align_and_coalesce(std::span<byte_range const>{}, std::optional<std::size_t>{})}
    ->std::same_as<std::vector<byte_range>>;
};

/**
 * @brief Optional write contract of a reactor.  A reactor that does not model
 *        it serves reads only, and
 *        the dispatcher's owner reports @c not_implemented_error (a
 *        @c std::system_error with @c std::errc::not_supported) for writes.
 *
 * Synchronous members (caller thread, may block, throw on error):
 *  - @c create_io_object_for_write(path, options): open @p path read+write per
 *    @c options.mode / @c permissions (best-effort @c size_hint), returning an
 *    object whose @c writable() is true and whose size is the size at open.
 *  - @c host_write(object, offset, size, src): write every byte (retrying
 *    EINTR and short writes) or throw; @c note_written(offset + size) on
 *    success; returns @p size.
 *  - @c flush(object): make every write completed before the call durable
 *    (@c fdatasync); throw on error.
 *
 * Asynchronous work arrives through the same @c enqueue as reads.  A request is
 * homogeneous: all slices @c io_op_kind::read (unchanged), all
 * @c io_op_kind::write, or exactly one @c io_op_kind::flush slice.
 *  - Write slice: @c rng.size > 0, never clamped (may lie past @c size());
 *    exactly one source, a contiguous host buffer (only read) XOR a device
 *    buffer with a resolved @c device_id; @c on_complete is null; slices of a
 *    request never overlap.  The source stays valid until the request's
 *    coordinator resolves.
 *  - Flush slice: empty range, no buffers, a coordinator of 0 bytes.
 *
 * Obligations: one credit per slice; @c add_tasks(N - 1) before publishing N
 * physical ops for a slice; every op settles exactly one credit through
 * @c io_op_request::finish_success / @c finish_error with @c kind = write and
 * @c io_rng = the exact file range written (finish_success then raises the
 * object's size); short writes are resubmitted, a 0-byte completion is EIO.
 * A device source is read in stream order after the work queued on
 * @c d_buffer.stream before the call (enqueue the D2H copy or
 * @c cuFileWriteAsync on that stream, or synchronize it first), with every
 * argument block / staging buffer kept alive until the stream passed it.
 * Ops of a request whose coordinator stopped (@c should_continue() false) are
 * cancelled with @c operation_canceled, and a request never resolves while one
 * of its ops can still touch the source or the file.  A flush slice is an
 * @c fdatasync of the object.  Shutdown and fatal errors are handled as for
 * reads.  A reactor may declare @c static constexpr bool supports_device_write
 * = false to accept host sources only.
 */
template <class R>
concept io_write_reactor_c = io_reactor_c<R> && requires(R reactor,
                                                         typename R::io_object_type const& object,
                                                         std::size_t offset,
                                                         std::size_t size,
                                                         std::uint8_t const* source,
                                                         std::string path,
                                                         write_open_options options)
{
  {R::create_io_object_for_write(std::move(path), options)}
    ->std::same_as<std::unique_ptr<typename R::io_object_type>>;
  {reactor.host_write(object, offset, size, source)}->std::same_as<std::size_t>;
  {R::flush(object)}->std::same_as<void>;
};

template <class R>
concept reactor_declares_bulk_io_preference = requires
{
  {R::prefers_bulk_io}->std::convertible_to<bool>;
};

template <class R>
concept reactor_declares_device_write_support = requires
{
  {R::supports_device_write}->std::convertible_to<bool>;
};

/// Write capabilities of a reactor.  Kept apart from @ref reactor_traits so a
/// reactor's existing full specialization of that template stays complete.
template <class R>
struct reactor_write_traits {
  static constexpr bool supports_write        = io_write_reactor_c<R>;
  static constexpr bool supports_device_write = [] {
    if constexpr (!io_write_reactor_c<R>) {
      return false;
    } else if constexpr (reactor_declares_device_write_support<R>) {
      return static_cast<bool>(R::supports_device_write);
    } else {
      return true;
    }
  }();
};

template <class R>
struct reactor_traits {
  // Every reactor satisfying the new contract accepts mixed prepared slices.
  static constexpr bool supports_device_read         = true;
  static constexpr bool supports_host_to_device_read = true;
  static constexpr bool supports_vector_host_read    = true;
  static constexpr bool supports_device_range_read   = true;
  static constexpr bool prefers_bulk_io              = [] {
    if constexpr (reactor_declares_bulk_io_preference<R>) {
      return static_cast<bool>(R::prefers_bulk_io);
    } else {
      return false;
    }
  }();
};

/**
 * @brief The dispatch logic in front of a pool of @p Reactor.
 *
 * @tparam Owner the class deriving from this one. It provides
 *   - `std::vector<Reactor*> next_reactor(io_object_type const&, std::size_t n_slices,
 *     io_op_type, int device_id)`: the reactors a request is split across
 *     (@ref default_next_reactor is the usual answer), and
 *   - `void on_device_dispatch_failure() noexcept`: called when a request with a
 *     device slice fails before it is queued.
 */
template <io_reactor_c Reactor, class Owner>
class reactor_dispatch {
 public:
  using reactor_type           = Reactor;
  using io_object_type         = typename Reactor::io_object_type;
  using reactor_config_type    = typename Reactor::reactor_config_type;
  using reactor_traits_t       = reactor_traits<Reactor>;
  using reactor_write_traits_t = reactor_write_traits<Reactor>;

  enum class io_op_type {
    host,
    host_async,
    device_async,
    host_vector_async,
    host_write,
    write_vector_async,
    flush_async
  };

  explicit reactor_dispatch(std::vector<std::unique_ptr<Reactor>> reactors)
    : _reactors(std::move(reactors))
  {
    if (!_reactors.empty()) _config = _reactors.front()->get_config();
  }

  template <class Factory>
  requires std::invocable<Factory&>&&
    std::convertible_to<std::invoke_result_t<Factory&>, std::unique_ptr<Reactor>>
    reactor_dispatch(std::size_t n_reactors, Factory factory)
  {
    _reactors.reserve(n_reactors);
    for (std::size_t i = 0; i < n_reactors; ++i) {
      _reactors.emplace_back(factory());
    }
    if (!_reactors.empty()) _config = _reactors.front()->get_config();
  }

  /// Start every reactor. Idempotent.
  void start_reactors()
  {
    if (_started) return;
    for (auto& reactor : _reactors) {
      reactor->start();
    }
    _started = true;
  }

  /// Shut every reactor down. A failing reactor is logged and the rest still shut down.
  void shutdown_reactors() noexcept
  {
    for (auto& reactor : _reactors) {
      try {
        reactor->shutdown();
      } catch (std::exception const& error) {
        KVIKIO_IO_LOG_ERROR("reactor_dispatch: reactor shutdown failed: %s", error.what());
      } catch (...) {
        KVIKIO_IO_LOG_ERROR("reactor_dispatch: reactor shutdown failed: unknown error");
      }
    }
  }

  /**
   * @brief Select the least-backlogged reactors a request is split across. The
   *        default behind the owner's @c next_reactor hook.
   *
   * At most @ref dispatch_fanout reactors: the reactor config's @c fan_out when it has one
   * (REST), else 2.  Rotation breaks load ties and keeps synchronous reads from sticking to
   * reactor zero.  queued_bytes is advisory; slot limits remain authoritative.  Demand reads
   * are kept ahead of readahead by each reactor's priority tiers (see @ref io_priority), not by
   * reserving reactors.
   */
  [[nodiscard]] std::vector<Reactor*> default_next_reactor()
  {
    auto const count = _reactors.size();
    if (count == 0) return {};

    auto const start = _next.fetch_add(1, std::memory_order_relaxed) % count;
    std::vector<std::pair<std::size_t, std::size_t>> ranked;
    ranked.reserve(count);
    for (std::size_t distance = 0; distance < count; ++distance) {
      ranked.emplace_back(_reactors[(start + distance) % count]->queued_bytes(), distance);
    }

    auto const selected = std::min(count, dispatch_fanout());
    std::partial_sort(
      ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(selected), ranked.end());

    std::vector<Reactor*> result;
    result.reserve(selected);
    for (std::size_t i = 0; i < selected; ++i) {
      result.push_back(_reactors[(start + ranked[i].second) % count].get());
    }
    return result;
  }

  /// Reactors one read is split across: the config's @c fan_out when it has one, else 2.
  [[nodiscard]] std::size_t dispatch_fanout() const noexcept
  {
    if constexpr (requires { _config.fan_out; }) {
      return std::max<std::size_t>(1, static_cast<std::size_t>(_config.fan_out));
    } else {
      return 2;
    }
  }

  /// Most slices one queue entry of a host-only range read may hold when the
  /// backend does not prefer bulk I/O (see dispatch_read); 0 = one entry
  /// per selected reactor.  Read from the reactor config's @c range_batch_slices
  /// when it has one (uring); other backends never split.
  [[nodiscard]] std::size_t range_batch_slices() const noexcept
  {
    if constexpr (requires { _config.range_batch_slices; }) {
      return static_cast<std::size_t>(_config.range_batch_slices);
    } else {
      return 0;
    }
  }

  std::size_t dispatch_host_read(io_object const& object,
                                 std::size_t offset,
                                 std::size_t size,
                                 std::uint8_t* destination)
  {
    auto const& typed = as_typed(object);
    size = std::min(size, typed.size() > offset ? typed.size() - offset : std::size_t{0});
    if (size == 0) return 0;

    auto reactors = owner().next_reactor(typed, 1, io_op_type::host, -1);
    if (reactors.empty()) throw std::runtime_error("host_read_io: no available reactors");
    return reactors.front()->host_read(typed, offset, size, destination);
  }

  exec::semi_future<std::size_t> dispatch_read(
    io_object const& object, std::vector<prepared_io_slice>&& input_slices) noexcept
  {
    if (input_slices.empty()) return exec::make_semi_future<std::size_t>(0);

    std::vector<prepared_io_slice> slices;
    bool has_device_slice = false;
    try {
      auto const& typed = as_typed(object);
      auto owner        = object.shared_from_this();

      slices.reserve(input_slices.size());
      std::size_t total_bytes = 0;
      int device_id           = -1;

      for (auto& slice : input_slices) {
        // A dropped slice reads nothing, so its completion always reports failure: reporting
        // success would publish never-filled cache chunks as `cached`.
        if (slice.rng.size == 0 || slice.rng.offset >= typed.size()) {
          if (slice.is_fragmented()) {
            KVIKIO_IO_LOG_WARN(
              "reactor_dispatch: dropping fragmented slice [%zu, +%zu) outside object '%s' of "
              "%zu bytes; caller must clamp to EOF",
              slice.rng.offset,
              slice.rng.size,
              typed.object_path(),
              typed.size());
          }
          if (slice.on_complete != nullptr) {
            (*slice.on_complete)(slice.h_buffer.fragments(), false);
            slice.on_complete.reset();
          }
          continue;
        }

        slice.rng.size = std::min(slice.rng.size, typed.size() - slice.rng.offset);
        if (slice.rng.size > std::numeric_limits<std::size_t>::max() - total_bytes) {
          throw std::overflow_error("mixed read byte count overflow");
        }
        total_bytes += slice.rng.size;

        if (slice.has_device_request()) {
          has_device_slice = true;
          if (device_id < 0) device_id = slice_device(slice);
          if (slice.d_buffer.device_id < 0) slice.d_buffer.device_id = device_id;
        }
        slices.push_back(std::move(slice));
      }

      if (slices.empty()) return exec::make_semi_future<std::size_t>(0);

      return dispatch_slices(typed,
                             owner,
                             slices,
                             total_bytes,
                             has_device_slice,
                             device_id,
                             io_op_type::host_vector_async);
    } catch (...) {
      auto const error = std::current_exception();
      if (has_device_slice) owner().on_device_dispatch_failure();
      auto fail_unsubmitted = [](auto& pending) noexcept {
        for (auto& slice : pending) {
          if (slice.on_complete != nullptr) {
            (*slice.on_complete)(slice.h_buffer.fragments(), false);
            slice.on_complete.reset();
          }
        }
      };
      fail_unsubmitted(slices);
      fail_unsubmitted(input_slices);
      return exec::make_semi_future<std::size_t>(error);
    }
  }

  // -- Writes (only for reactors modelling io_write_reactor_c) -----------------

  /// Largest write slice dispatched as one queue entry; a larger one is cut at
  /// granule-aligned file offsets so its bytes spread over the selected reactors
  /// (see dispatch_write).  Read from the reactor config's
  /// @c write_split_bytes when it has one; 0 (or no such member) = no split.
  [[nodiscard]] std::size_t write_split_bytes() const noexcept
  {
    if constexpr (requires { _config.write_split_bytes; }) {
      return static_cast<std::size_t>(_config.write_split_bytes);
    } else {
      return 0;
    }
  }

  std::size_t dispatch_host_write(io_object const& object,
                                  std::size_t offset,
                                  std::size_t size,
                                  std::uint8_t const* source) requires
    reactor_write_traits_t::supports_write
  {
    auto const& typed = as_writable(object);
    if (size == 0) return 0;
    auto reactors = owner().next_reactor(typed, 1, io_op_type::host_write, -1);
    if (reactors.empty()) throw std::runtime_error("host_write_io: no available reactors");
    return reactors.front()->host_write(typed, offset, size, source);
  }

  /**
   * @brief Dispatch write slices to the reactors.
   *
   * Empty slices are dropped and the rest validated (@c validate_write_slices);
   * nothing is clamped to EOF.  Device slices get their device id resolved.  A
   * slice larger than @ref write_split_bytes is cut into sub-slices at
   * multiples of that size (rounded down to the write granule), each with its
   * own coordinator credit, before the slices are spread over the selected
   * reactors exactly like a read.
   */
  exec::semi_future<std::size_t> dispatch_write(
    io_object const& object, std::vector<prepared_io_slice>&& input_slices) noexcept requires
    reactor_write_traits_t::supports_write
  {
    {
      bool has_device_slice = false;
      try {
        auto const& typed = as_writable(object);
        auto owner        = object.shared_from_this();

        std::erase_if(input_slices, [](prepared_io_slice const& slice) {
          return slice.is_write() && slice.rng.size == 0;
        });
        if (input_slices.empty()) return exec::make_semi_future<std::size_t>(0);
        auto const total_bytes = validate_write_slices(input_slices);

        int device_id      = -1;  // routes the request: GPU of its first device slice
        int current_device = -1;
        for (auto& slice : input_slices) {
          if (!slice.has_device_request()) continue;
          if constexpr (!reactor_write_traits_t::supports_device_write) {
            throw std::system_error(std::make_error_code(std::errc::not_supported),
                                    "mixed_writev_async_io: backend cannot write device memory");
          }
          has_device_slice = true;
          if (slice.d_buffer.device_id < 0) {
            if (current_device < 0) current_device = slice_device(slice);
            slice.d_buffer.device_id = current_device;
          }
          if (device_id < 0) device_id = slice.d_buffer.device_id;
        }

        auto slices = split_write_slices(std::move(input_slices), write_split_bytes());
        return dispatch_slices(typed,
                               owner,
                               slices,
                               total_bytes,
                               has_device_slice,
                               device_id,
                               io_op_type::write_vector_async);
      } catch (...) {
        auto const error = std::current_exception();
        if (has_device_slice) owner().on_device_dispatch_failure();
        return exec::make_semi_future<std::size_t>(error);
      }
    }
  }

  void dispatch_flush(io_object const& object) requires reactor_write_traits_t::supports_write
  {
    Reactor::flush(as_writable(object));
  }

  /// One flush slice, queued on the least-backlogged reactor.
  exec::semi_future<void> dispatch_flush_async(io_object const& object,
                                               io_priority priority) noexcept requires
    reactor_write_traits_t::supports_write
  {
    {
      try {
        auto const& typed = as_writable(object);
        auto owner        = object.shared_from_this();
        std::vector<prepared_io_slice> slices;
        slices.push_back(make_flush_slice(resolve_priority(priority, io_priority::high)));
        auto future = dispatch_slices(typed, owner, slices, 0, false, -1, io_op_type::flush_async);
        return std::move(future).defer_value([](std::size_t) {});
      } catch (...) {
        return exec::make_semi_future<void>(exec::try_t<void>(std::current_exception()));
      }
    }
  }

 protected:
  /**
   * @brief Publish prepared slices to the reactors as grouped requests.
   *
   * Shared by reads and writes: stamps one priority on every slice, selects
   * reactors with @ref next_reactor, creates the request's coordinator (one
   * credit per slice), spreads the slices over the reactors by bytes, cuts a
   * host-only request into entries of at most @ref range_batch_slices slices
   * when the backend does not prefer bulk I/O, and enqueues.  @p slices must be
   * non-empty.  Throws only before the coordinator exists (the caller then
   * settles @p slices, which are left intact); after that every failure is
   * settled through the returned future.
   */
  exec::semi_future<std::size_t> dispatch_slices(io_object_type const& typed,
                                                 std::shared_ptr<io_object const> const& owner,
                                                 std::vector<prepared_io_slice>& slices,
                                                 std::size_t total_bytes,
                                                 bool has_device_slice,
                                                 int device_id,
                                                 io_op_type operation)
  {
    // A device slice on the per-thread default stream must be ordered against
    // THIS thread's stream before a reactor worker sees it.
    if (has_device_slice) resolve_default_streams(slices);

    // One call site issues one priority, so the first slice speaks for the
    // request; stamp the resolved value on every slice so each queue entry
    // built below carries it.
    auto const priority = request_priority(slices);
    for (auto& slice : slices) {
      slice.priority = priority;
    }

    auto reactors    = owner().next_reactor(typed, slices.size(), operation, device_id);
    auto coordinator = std::make_shared<grouped_coordinator>(total_bytes, slices.size());
    auto future      = coordinator->get_future();

    if (reactors.empty()) {
      auto error = std::make_exception_ptr(
        std::runtime_error(operation == io_op_type::host_vector_async
                             ? "mixed_readv_async_io: no available reactors"
                             : "mixed_writev_async_io: no available reactors"));
      for (auto& slice : slices) {
        if (slice.on_complete != nullptr) {
          (*slice.on_complete)(slice.h_buffer.fragments(), false);
        }
        coordinator->report_error(error);
      }
      return future;
    }

    try {
      auto const partition_count = std::min(reactors.size(), slices.size());
      std::vector<std::vector<prepared_io_slice>> partitions(partition_count);
      std::vector<std::size_t> partition_bytes(partition_count, 0);

      // Keep the originals until every queue entry has been allocated. If an
      // allocation fails, their callbacks can still release all claimed cache
      // chunks and every coordinator credit can be settled.
      for (auto const& slice : slices) {
        auto const smallest = static_cast<std::size_t>(
          std::min_element(partition_bytes.begin(), partition_bytes.end()) -
          partition_bytes.begin());
        partition_bytes[smallest] += slice.size();
        partitions[smallest].push_back(slice.clone());
      }

      // A backend that does not prefer bulk I/O gets a host-only range read as
      // several queue entries of at most range_batch_slices() slices each, not
      // one entry holding every slice.  Coordinator credits are per slice, so
      // the split does not change how the read settles.
      auto const batch = !reactor_traits_t::prefers_bulk_io && !has_device_slice
                           ? range_batch_slices()
                           : std::size_t{0};

      std::vector<std::pair<Reactor*, std::unique_ptr<grouped_io_request>>> requests;
      requests.reserve(partition_count);
      for (std::size_t i = 0; i < partition_count; ++i) {
        auto& partition = partitions[i];
        if (batch == 0 || partition.size() <= batch) {
          requests.emplace_back(
            reactors[i], grouped_io_request::create(owner, std::move(partition), coordinator));
          continue;
        }
        for (std::size_t first = 0; first < partition.size(); first += batch) {
          auto const last = std::min(partition.size(), first + batch);
          std::vector<prepared_io_slice> part(
            std::make_move_iterator(partition.begin() + static_cast<std::ptrdiff_t>(first)),
            std::make_move_iterator(partition.begin() + static_cast<std::ptrdiff_t>(last)));
          requests.emplace_back(reactors[i],
                                grouped_io_request::create(owner, std::move(part), coordinator));
        }
      }

      // enqueue is noexcept by reactor contract, so once publication starts
      // ownership cannot be stranded between reactors.
      for (auto& [reactor, request] : requests) {
        reactor->enqueue(std::move(request));
      }
      return future;
    } catch (...) {
      auto const error = std::current_exception();
      if (has_device_slice) owner().on_device_dispatch_failure();
      for (auto& slice : slices) {
        if (slice.on_complete != nullptr) {
          (*slice.on_complete)(slice.h_buffer.fragments(), false);
        }
        coordinator->report_error(error);
      }
      return future;
    }
  }
  /// Resolve the queue tier for a request whose slices are @p slices.  An
  /// explicit priority on the first slice wins (a cache prefetch stamps
  /// @c low).  @c automatic always resolves to @c high, whatever the call
  /// shape: a caller that did not opt into the low tier is treated as demand
  /// work it may block on (a multi-range host read from a native decoder is
  /// exactly that), so only explicitly-stamped background work runs low.
  [[nodiscard]] static io_priority request_priority(
    std::vector<prepared_io_slice> const& slices) noexcept
  {
    return resolve_priority(slices.front().priority, io_priority::high);
  }

  /**
   * @brief Cut every write slice larger than @p split_bytes into sub-slices.
   *
   * Cut points are file offsets that are multiples of @p split_bytes rounded
   * down to the write granule, so an O_DIRECT-aligned slice stays aligned piece
   * by piece.  Each piece keeps the slice's source (advanced to its offset),
   * stream, device and priority.  @p split_bytes == 0 returns the input.
   */
  [[nodiscard]] static std::vector<prepared_io_slice> split_write_slices(
    std::vector<prepared_io_slice>&& slices, std::size_t split_bytes)
  {
    if (split_bytes == 0) return std::move(slices);
    auto const granule = write_granule();
    auto const piece   = std::max(granule, split_bytes - split_bytes % granule);

    std::vector<prepared_io_slice> result;
    result.reserve(slices.size());
    for (auto& slice : slices) {
      if (slice.size() <= piece) {
        result.push_back(std::move(slice));
        continue;
      }
      auto const begin = slice.offset();
      auto const end   = slice.rng.end();
      auto cursor      = begin;
      while (cursor < end) {
        auto const boundary = cursor - cursor % piece + piece;
        auto const stop     = std::min(end, boundary);
        auto part           = slice.clone();
        part.rng            = range{cursor, stop - cursor};
        auto const delta    = cursor - begin;
        if (part.has_device_request()) {
          part.d_buffer.data += delta;
        } else {
          part.h_buffer.buffer = std::get<std::uint8_t*>(slice.h_buffer.buffer) + delta;
        }
        result.push_back(std::move(part));
        cursor = stop;
      }
    }
    return result;
  }

  reactor_config_type _config{};
  std::vector<std::unique_ptr<Reactor>> _reactors;
  std::atomic<std::size_t> _next{0};
  bool _started{false};

  static io_object_type const& as_typed(io_object const& object)
  {
    auto const* typed = dynamic_cast<io_object_type const*>(&object);
    if (typed == nullptr) throw std::invalid_argument("I/O object belongs to another backend");
    return *typed;
  }

  /// @ref as_typed, plus @c bad_file_descriptor for an object opened read-only.
  static io_object_type const& as_writable(io_object const& object)
  {
    auto const& typed = as_typed(object);
    if (!typed.writable()) {
      throw std::system_error(
        std::make_error_code(std::errc::bad_file_descriptor),
        "templated_ioctx: '" + typed.object_path() + "' was not opened for writing");
    }
    return typed;
  }

 private:
  [[nodiscard]] Owner& owner() noexcept { return static_cast<Owner&>(*this); }
};

}  // namespace kvikio::io::detail
