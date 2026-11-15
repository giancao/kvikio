/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file types.hpp
 * @brief Vocabulary of the engine interface: ranges, slices, buffers and objects.
 *
 * These types describe one request to an @c engine (see engine.hpp). They carry
 * no cache state: a destination made of several pieces of memory is described
 * by @c fragment, a plain (file offset, memory, opaque handle) triple that the
 * layer above fills in.
 *
 * The types that own a list (@c host_buffer, @c prepared_io_slice) are
 * move-only, so a request's lists are allocated once and then only moved.
 */

#include <kvikio/exec/invocable.hpp>
#include <kvikio/io/byte_range.hpp>
#include <kvikio/io/object_identity.hpp>

#include <cuda.h>
#include <sys/types.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace kvikio::io {

class engine;

// ---------------------------------------------------------------------------
// Ranges
// ---------------------------------------------------------------------------

/// The block size local I/O is aligned to.
inline constexpr std::size_t IO_BLOCK_SIZE = 4096;

/// The granule writes are split at: @c max(IO_BLOCK_SIZE, page size).  The
/// aligned body of a write goes O_DIRECT and its unaligned head / tail through
/// the page cache; splitting at a page boundary keeps the two from ever sharing
/// a page (64 KiB pages on some aarch64 systems).
[[nodiscard]] inline std::size_t write_granule() noexcept
{
  static std::size_t const granule = [] {
    auto const page = ::sysconf(_SC_PAGESIZE);
    return page > 0 ? std::max(IO_BLOCK_SIZE, static_cast<std::size_t>(page)) : IO_BLOCK_SIZE;
  }();
  return granule;
}

/// The half-open byte range [offset, offset + size) of a file or object.
struct range {
  std::size_t offset{0};
  std::size_t size{0};

  /// One past the last byte, saturating at the largest @c std::size_t.
  [[nodiscard]] std::size_t end() const noexcept
  {
    return size > std::numeric_limits<std::size_t>::max() - offset
             ? std::numeric_limits<std::size_t>::max()
             : offset + size;
  }

  [[nodiscard]] bool empty() const noexcept { return size == 0; }
};

/// The overlap of @p lhs and @p rhs; empty when they do not overlap.
[[nodiscard]] inline range intersect(range lhs, range rhs) noexcept
{
  auto const begin = std::max(lhs.offset, rhs.offset);
  auto const end   = std::min(lhs.end(), rhs.end());
  return end > begin ? range{begin, end - begin} : range{begin, 0};
}

/// One segment of a vectored host read: the file range and where its bytes go.
struct slice {
  slice() noexcept = default;

  explicit slice(std::size_t offset, std::size_t size, std::uint8_t* dst) noexcept
    : rng{offset, size}, dst{dst}
  {
    assert(dst != nullptr);
    assert(size > 0);
  }

  [[nodiscard]] std::size_t size() const noexcept { return rng.size; }

  [[nodiscard]] std::size_t offset() const noexcept { return rng.offset; }

  range rng;
  std::uint8_t* dst{nullptr};
};

// ---------------------------------------------------------------------------
// Write vocabulary
// ---------------------------------------------------------------------------

/// How @c engine::open_for_write opens a local file. Every mode opens the file
/// for reading and writing.
enum class write_mode : std::uint8_t {
  create_or_truncate,  ///< create if missing, truncate to zero bytes otherwise
  create_or_open,      ///< create if missing, keep the existing content otherwise
  open_existing,       ///< fail with @c ENOENT if missing; keep the existing content
  create_new,          ///< fail with @c EEXIST if present
};

/// Options of @c engine::open_for_write.
struct write_open_options {
  write_mode mode{write_mode::create_or_truncate};  ///< open / create behaviour
  ::mode_t permissions{0644};                       ///< mode of a newly created file
  /// Best-effort preallocation hint in bytes; 0 = none. Never changes @c size();
  /// an engine or filesystem that cannot preallocate ignores it.
  std::uint64_t size_hint{0};
};

/**
 * @brief One segment of a vectored write: the file range
 *        [offset, offset + size) and the bytes to write there.
 *
 * @c src is host or device memory depending on the API it is passed to. It must
 * stay valid and unmodified until the write's future resolves.
 */
struct write_slice {
  write_slice() noexcept = default;

  write_slice(std::size_t offset, std::size_t size, std::uint8_t const* src) noexcept
    : rng{offset, size}, src{src}
  {
  }

  [[nodiscard]] std::size_t size() const noexcept { return rng.size; }

  [[nodiscard]] std::size_t offset() const noexcept { return rng.offset; }

  range rng;
  std::uint8_t const* src{nullptr};
};

// ---------------------------------------------------------------------------
// Destinations
// ---------------------------------------------------------------------------

/// What the layer above uses to recognize a fragment when the engine hands it
/// back. The engine never dereferences it.
using fragment_handle = void*;

/**
 * @brief One piece of a fragmented host destination.
 *
 * The engine reads the file range [offset, offset + size) into @c data. The
 * range is fixed when the request is submitted. Neighbouring fragments of one
 * buffer need not be contiguous in the file or in memory.
 */
struct fragment {
  std::size_t offset{0};            ///< file offset of @c data[0]
  std::uint8_t* data{nullptr};      ///< where the bytes go
  std::size_t size{0};              ///< bytes to read
  fragment_handle handle{nullptr};  ///< handed back when this fragment is filled

  [[nodiscard]] range rng() const noexcept { return range{offset, size}; }
};

/**
 * @brief The host destination of a read, or the host source of a write.
 *
 * One of three forms:
 *  - empty: the engine stages the bytes in memory from its staging source;
 *  - contiguous: one block of caller memory;
 *  - fragmented: a list of @ref fragment.
 *
 * Move-only: the fragment list is allocated once per request.
 */
struct host_buffer {
  host_buffer() noexcept = default;

  explicit host_buffer(std::uint8_t* dst) noexcept : buffer(dst) { assert(dst != nullptr); }

  explicit host_buffer(std::vector<fragment>&& fragments) noexcept : buffer(std::move(fragments))
  {
    assert(!std::get<std::vector<fragment>>(buffer).empty());
  }

  host_buffer(host_buffer const&)                = delete;
  host_buffer& operator=(host_buffer const&)     = delete;
  host_buffer(host_buffer&&) noexcept            = default;
  host_buffer& operator=(host_buffer&&) noexcept = default;

  [[nodiscard]] bool needs_staging() const noexcept
  {
    return std::holds_alternative<std::monostate>(buffer);
  }

  [[nodiscard]] bool is_fragmented() const noexcept
  {
    return std::holds_alternative<std::vector<fragment>>(buffer);
  }

  [[nodiscard]] bool is_contiguous() const noexcept
  {
    return std::holds_alternative<std::uint8_t*>(buffer);
  }

  /// The fragments of a fragmented buffer; empty for the other two forms.
  [[nodiscard]] std::span<fragment const> fragments() const noexcept
  {
    if (!is_fragmented()) return {};
    return std::get<std::vector<fragment>>(buffer);
  }

  /// An explicit copy. Copies the fragment list, which allocates.
  [[nodiscard]] host_buffer clone() const
  {
    host_buffer result;
    result.buffer = buffer;
    return result;
  }

  std::variant<std::monostate, std::uint8_t*, std::vector<fragment>> buffer;
};

/**
 * @brief The device destination of a read, or the device source of a write.
 *
 * @c stream is the CUDA stream the copy is ordered on; the handle is the same
 * object the CUDA runtime calls @c cudaStream_t.
 */
struct device_buffer {
  device_buffer() noexcept = default;

  explicit device_buffer(std::uint8_t* dst, CUstream stream, int device_id = -1) noexcept
    : data(dst), stream(stream), device_id(device_id)
  {
    assert(dst != nullptr);
  }

  std::uint8_t* data{nullptr};
  /// Null is the legacy default stream.
  CUstream stream{nullptr};
  /// The device owning @c data; negative means the calling thread's device.
  int device_id{-1};
  /// Keeps alive a stream created in place of the caller's; null when
  /// @c stream is the caller's own.
  std::shared_ptr<void> stream_owner;
};

/**
 * @brief Tells the layer above which fragments an engine finished.
 *
 * An engine calls it once per physical operation, with the handles of the
 * fragments that operation filled and whether it succeeded. It does not wait
 * for the rest of the request, so a reader of an early fragment is not held up
 * by a late one. Calls for one request may come from several engine threads;
 * they are serialized here.
 *
 * The callback runs on an engine thread: it must not block.
 */
class prepared_io_completion final {
 public:
  using callback_type = exec::invocable<void(std::span<fragment_handle const>, bool) noexcept>;

  template <typename Callback>
  explicit prepared_io_completion(Callback&& callback) : _callback(std::forward<Callback>(callback))
  {
  }

  prepared_io_completion(prepared_io_completion const&)            = delete;
  prepared_io_completion& operator=(prepared_io_completion const&) = delete;

  void operator()(std::span<fragment_handle const> handles, bool success) noexcept
  {
    std::lock_guard lock(_mutex);
    _callback(handles, success);
  }

  /// Report every fragment of @p fragments (a whole slice given back, e.g. on a
  /// failure path). The callback is called at least once, also for an empty
  /// list, and exactly once for up to @ref report_batch fragments; a longer
  /// list is reported in batches. Allocates nothing.
  void operator()(std::span<fragment const> fragments, bool success) noexcept
  {
    std::lock_guard lock(_mutex);
    std::array<fragment_handle, report_batch> handles;
    do {
      auto const n = std::min(fragments.size(), report_batch);
      for (std::size_t i = 0; i < n; ++i) {
        handles[i] = fragments[i].handle;
      }
      _callback(std::span<fragment_handle const>{handles.data(), n}, success);
      fragments = fragments.subspan(n);
    } while (!fragments.empty());
  }

  /// Fragments reported per callback call by the fragment-list overload.
  static constexpr std::size_t report_batch = 64;

 private:
  std::mutex _mutex;
  callback_type _callback;
};

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

/// Which of an engine's queues a request waits in. @c high is a read its caller
/// is waiting for; @c low is background work such as prefetching. An engine
/// that has no priorities (see @c engine_capabilities::priority) treats both
/// alike. @c automatic resolves to @c high.
enum class io_priority : std::uint8_t { automatic, high, low };

/// @p requested, or @p fallback when it is @c io_priority::automatic.
[[nodiscard]] constexpr io_priority resolve_priority(io_priority requested,
                                                     io_priority fallback) noexcept
{
  return requested == io_priority::automatic ? fallback : requested;
}

/// What a prepared slice asks an engine to do.
enum class io_op_kind : std::uint8_t { read, write, flush };

/**
 * @brief One unit of a request: a file range plus where its bytes go or come from.
 *
 * Move-only (it owns a @ref host_buffer).
 */
struct prepared_io_slice {
  /// The logical range the caller asked for. An engine may widen the physical
  /// I/O for alignment, but device copies and returned byte counts stay limited
  /// to this range. A write slice's range is the exact file range to write; a
  /// flush slice's range is empty.
  range rng;
  host_buffer h_buffer;  ///< empty if the engine stages the bytes itself
  device_buffer d_buffer;
  /// Called per physical operation for fragmented host buffers; may be null.
  std::shared_ptr<prepared_io_completion> on_complete;
  /// A request takes its first slice's value.
  io_priority priority{io_priority::automatic};
  /// A write slice carries exactly one source: a contiguous host buffer or a
  /// device buffer, which the engine only reads.
  io_op_kind kind{io_op_kind::read};

  prepared_io_slice() noexcept = default;
  explicit prepared_io_slice(range r, host_buffer h) noexcept : rng(r), h_buffer(std::move(h)) {}
  explicit prepared_io_slice(range r, device_buffer d) noexcept : rng(r), d_buffer(std::move(d)) {}
  explicit prepared_io_slice(range r, host_buffer h, device_buffer d) noexcept
    : rng(r), h_buffer(std::move(h)), d_buffer(std::move(d))
  {
  }

  prepared_io_slice(prepared_io_slice const&)                = delete;
  prepared_io_slice& operator=(prepared_io_slice const&)     = delete;
  prepared_io_slice(prepared_io_slice&&) noexcept            = default;
  prepared_io_slice& operator=(prepared_io_slice&&) noexcept = default;

  /// An explicit copy, sharing @c on_complete. Copies a fragment list, which
  /// allocates.
  [[nodiscard]] prepared_io_slice clone() const
  {
    prepared_io_slice result;
    result.rng         = rng;
    result.h_buffer    = h_buffer.clone();
    result.d_buffer    = d_buffer;
    result.on_complete = on_complete;
    result.priority    = priority;
    result.kind        = kind;
    return result;
  }

  [[nodiscard]] bool needs_staging() const noexcept { return h_buffer.needs_staging(); }

  [[nodiscard]] bool is_fragmented() const noexcept { return h_buffer.is_fragmented(); }

  [[nodiscard]] bool is_contiguous() const noexcept { return h_buffer.is_contiguous(); }

  [[nodiscard]] bool has_host_request() const noexcept { return !h_buffer.needs_staging(); }

  [[nodiscard]] bool has_device_request() const noexcept { return d_buffer.data != nullptr; }

  [[nodiscard]] bool is_host_request() const noexcept
  {
    return !h_buffer.needs_staging() && !has_device_request();
  }

  [[nodiscard]] std::size_t size() const noexcept { return rng.size; }

  [[nodiscard]] std::size_t offset() const noexcept { return rng.offset; }

  [[nodiscard]] bool is_write() const noexcept { return kind == io_op_kind::write; }

  [[nodiscard]] bool is_flush() const noexcept { return kind == io_op_kind::flush; }
};

/// A write slice of @p rng whose bytes come from host memory @p src (non-null).
/// The engine only reads @p src; the cast merely fits the shared buffer type.
[[nodiscard]] inline prepared_io_slice make_host_write_slice(
  range rng, std::uint8_t const* src, io_priority priority = io_priority::automatic) noexcept
{
  prepared_io_slice result{rng, host_buffer{const_cast<std::uint8_t*>(src)}};
  result.kind     = io_op_kind::write;
  result.priority = priority;
  return result;
}

/// A write slice of @p rng whose bytes come from device memory @p src
/// (non-null), read in stream order on @p stream.
[[nodiscard]] inline prepared_io_slice make_device_write_slice(
  range rng,
  std::uint8_t const* src,
  CUstream stream,
  int device_id        = -1,
  io_priority priority = io_priority::automatic) noexcept
{
  prepared_io_slice result{rng, device_buffer{const_cast<std::uint8_t*>(src), stream, device_id}};
  result.kind     = io_op_kind::write;
  result.priority = priority;
  return result;
}

/// A flush slice: make every write that completed before it durable.
[[nodiscard]] inline prepared_io_slice make_flush_slice(
  io_priority priority = io_priority::automatic) noexcept
{
  prepared_io_slice result;
  result.kind     = io_op_kind::flush;
  result.priority = priority;
  return result;
}

// ---------------------------------------------------------------------------
// Objects
// ---------------------------------------------------------------------------

/// What the caller already knows it will read first, so an engine can fetch it
/// while opening.
enum class open_hint : std::uint8_t {
  generic,               ///< nothing in particular
  parquet_footer_probe,  ///< the tail of the object, where a Parquet footer lives
};

/// Options of @c engine::open.
struct open_options {
  open_hint hint{open_hint::generic};
  /// The object's size, when the caller already knows it; an engine whose size
  /// discovery costs a round trip then skips it.
  std::optional<std::uint64_t> known_size;
};

/// Whether an engine opened an @c io_object for reading only or for reading and
/// writing.
enum class io_access : std::uint8_t { read_only, read_write };

/**
 * @brief An opened file or object.
 *
 * Holds the engine's native handles (file descriptors, or an object store's
 * bucket, key and validator). Performs no I/O of its own. An engine accepts
 * only the objects it opened itself.
 *
 * Always owned by a @c std::shared_ptr, so the layer above can extend its
 * lifetime with @c shared_from_this().
 */
class io_object : public std::enable_shared_from_this<io_object> {
 public:
  virtual ~io_object() = default;

  /// This open's process-wide order (see @ref next_open_generation): a later
  /// open of a path has a larger generation than an earlier one.
  [[nodiscard]] std::uint64_t open_generation() const noexcept { return _open_generation; }

  /// The evidence that another open of this path sees the same version of the
  /// object. An engine overrides it with what it can prove (a local file's stat
  /// tuple, a remote object's strong validator). The default uses
  /// @ref validation_tag as an @c object_tag only when the engine declares it
  /// strong via @ref validation_tag_is_strong. Otherwise the identity is scoped
  /// to this open (@c opened_handle, versioned by @ref open_generation), never
  /// to the path. A writable object always has an @c opened_handle identity of
  /// size 0: its bytes and size change as it is written.
  [[nodiscard]] virtual object_identity identity() const
  {
    object_identity result;
    auto const tag = validation_tag();
    if (writable() || tag.empty() || !validation_tag_is_strong()) {
      result.kind    = identity_kind::opened_handle;
      result.size    = writable() ? 0 : size();
      result.version = std::to_string(_open_generation);
    } else {
      result.kind    = identity_kind::object_tag;
      result.size    = size();
      result.version = std::string(tag);
    }
    return result;
  }

  /// The key of this object's version: @ref identity framed with
  /// @ref raw_file_cache_id. Two opens with equal keys may share what was
  /// derived from either.
  [[nodiscard]] std::string identity_cache_key() const
  {
    return identity().cache_key(raw_file_cache_id());
  }

  /// Engine locator, part of the identity-aware key (see
  /// @ref identity_cache_key). Often equal to @c object_path() but may differ
  /// for engines that need to distinguish otherwise-equal paths.
  [[nodiscard]] virtual const std::string& raw_file_cache_id() const noexcept = 0;

  /// The path / URL / key the caller used to open this object.
  [[nodiscard]] virtual const std::string& object_path() const noexcept = 0;

  /// Total size of the underlying object, recorded when it was opened.
  [[nodiscard]] virtual std::size_t size() const noexcept = 0;

  /// Opaque validator observed when the object was opened (an HTTP ETag, with
  /// its quotes and any weak W/ prefix); empty when unavailable. The view is
  /// valid for this object's lifetime.
  [[nodiscard]] virtual std::string_view validation_tag() const noexcept { return {}; }

  /// Whether a nonempty validation_tag() is a strong validator: equal tags of
  /// the same size at the same raw_file_cache_id() name byte-identical
  /// versions. Default false.
  [[nodiscard]] virtual bool validation_tag_is_strong() const noexcept { return false; }

  /// True for objects returned by @c engine::open_for_write.
  [[nodiscard]] virtual bool writable() const noexcept { return false; }

  /// A physical write ending at byte @p end completed through this object. A
  /// writable object grows @ref size to cover it; a no-op for read-only objects.
  virtual void note_written(std::size_t /*end*/) const noexcept {}

  /// A value the layer above the engine attaches to this object right after
  /// opening it, before the object is shared. Engines neither read nor write
  /// it. Not synchronized.
  [[nodiscard]] std::uint64_t client_tag() const noexcept { return _client_tag; }

  /// See @ref client_tag.
  void set_client_tag(std::uint64_t tag) noexcept { _client_tag = tag; }

 private:
  std::uint64_t _client_tag{0};
  std::uint64_t const _open_generation = next_open_generation();
};

}  // namespace kvikio::io
