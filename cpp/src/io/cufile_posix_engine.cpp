/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <kvikio/error.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/io/cufile_posix_engine.hpp>
#include <kvikio/io/errors.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/threadpool_wrapper.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::io {

namespace {

constexpr std::size_t no_fragment = std::numeric_limits<std::size_t>::max();

[[noreturn]] void throw_errno(char const* what)
{
  throw std::system_error(errno, std::generic_category(), what);
}

/// A file opened by this engine.
class local_object final : public io_object {
 public:
  local_object(std::string path, FileHandle file, bool writable)
    : _path(std::move(path)), _file(std::move(file)), _writable(writable)
  {
    _size.store(_file.nbytes(), std::memory_order_relaxed);
    if (!_writable) { _has_identity = capture_local_identity(_file.fd(), _identity); }
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override
  {
    return _size.load(std::memory_order_acquire);
  }
  [[nodiscard]] bool writable() const noexcept override { return _writable; }

  [[nodiscard]] object_identity identity() const override
  {
    return _has_identity ? _identity : io_object::identity();
  }

  void note_written(std::size_t end) const noexcept override
  {
    auto current = _size.load(std::memory_order_relaxed);
    while (current < end && !_size.compare_exchange_weak(
                              current, end, std::memory_order_acq_rel, std::memory_order_relaxed)) {
    }
  }

  [[nodiscard]] int fd() const noexcept { return _file.fd(); }

  /// Whether transfers to and from device memory go through cuFile.
  [[nodiscard]] bool uses_cufile() const noexcept
  {
    return !_file.get_compat_mode_manager().is_compat_mode_preferred();
  }

  [[nodiscard]] FileHandle& file() const noexcept { return _file; }

 private:
  std::string _path;
  mutable FileHandle _file;
  bool _writable;
  mutable std::atomic<std::size_t> _size{0};
  object_identity _identity;
  bool _has_identity{false};
};

/// Read exactly @p size bytes at @p offset; a short read is an error.
void read_exact(int fd, std::uint8_t* dst, std::size_t size, std::size_t offset)
{
  std::size_t done = 0;
  while (done < size) {
    auto const n = ::pread(fd, dst + done, size - done, static_cast<off_t>(offset + done));
    if (n < 0) {
      if (errno == EINTR) { continue; }
      throw_errno("cufile_posix: pread");
    }
    if (n == 0) {
      throw std::system_error(std::make_error_code(std::errc::io_error),
                              "cufile_posix: the file is shorter than when it was opened");
    }
    done += static_cast<std::size_t>(n);
  }
}

void write_exact(int fd, std::uint8_t const* src, std::size_t size, std::size_t offset)
{
  std::size_t done = 0;
  while (done < size) {
    auto const n = ::pwrite(fd, src + done, size - done, static_cast<off_t>(offset + done));
    if (n < 0) {
      if (errno == EINTR) { continue; }
      throw_errno("cufile_posix: pwrite");
    }
    done += static_cast<std::size_t>(n);
  }
}

/// The stream a copy is queued on. Null means the legacy default stream; the
/// per-thread default stream cannot be honoured from a worker thread.
CUstream resolve_stream(CUstream stream)
{
  if (stream == nullptr) { return CU_STREAM_LEGACY; }
  if (stream == CU_STREAM_PER_THREAD) {
    throw std::invalid_argument(
      "cufile_posix: the per-thread default stream is not supported; pass an explicit stream");
  }
  return stream;
}

/// Wait until everything queued on @p stream so far has run.
void wait_for_stream(CUstream stream)
{
  CUevent event{};
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().EventCreate(&event, CU_EVENT_DISABLE_TIMING),
                         std::runtime_error);
  auto status = cudaAPI::instance().EventRecord(event, stream);
  if (status == CUDA_SUCCESS) { status = cudaAPI::instance().EventSynchronize(event); }
  cudaAPI::instance().EventDestroy(event);
  KVIKIO_CUDA_DRIVER_TRY(status, std::runtime_error);
}

struct plan_op {
  std::size_t slice{0};
  std::size_t fragment{no_fragment};
  std::size_t offset{0};  ///< file offset
  std::size_t size{0};    ///< bytes to transfer; already clamped to the end of the file
};

class cufile_posix_engine;

/// Everything one read(), write() or flush_async() call needs until it settles.
struct request {
  std::shared_ptr<io_object const> owner;
  local_object const* object{nullptr};
  std::vector<prepared_io_slice> slices;
  std::vector<plan_op> ops;
  exec::promise<std::size_t> promise;
  std::size_t total_bytes{0};
  std::atomic<std::size_t> remaining{0};
  std::mutex error_mutex;
  std::exception_ptr error;

  void fail(std::exception_ptr e)
  {
    std::lock_guard lock(error_mutex);
    if (error == nullptr) { error = std::move(e); }
  }

  /// One operation finished. Settles the future after the last one.
  void finish_one()
  {
    if (remaining.fetch_sub(1, std::memory_order_acq_rel) != 1) { return; }
    if (error != nullptr) {
      promise.set_exception(error);
    } else {
      promise.set_value(total_bytes);
    }
  }
};

class cufile_posix_engine final : public engine {
 public:
  explicit cufile_posix_engine(cufile_posix_options options)
    : _options(std::move(options)),
      _pool(std::max(1U, _options.num_threads), make_thread_pool_init_task("kvikio-eng"))
  {
    _caps.device_read            = true;
    _caps.host_to_device_read    = true;
    _caps.vector_host_read       = true;
    _caps.device_range_read      = true;
    _caps.fragmented_host_buffer = true;
    _caps.write                  = true;
    _caps.device_write           = true;
    _caps.priority               = false;
    _caps.prefers_bulk_io        = false;
    _caps.min_alignment          = 1;
    _caps.merge_gap              = 4096;
    _caps.max_concurrent_scans   = std::max(1U, _options.num_threads);
    if (_options.max_operation_bytes == 0) { _options.max_operation_bytes = 16UL << 20; }
  }

  ~cufile_posix_engine() override { shutdown(); }

  [[nodiscard]] std::string_view name() const noexcept override { return cufile_posix_engine_name; }

  [[nodiscard]] engine_capabilities const& capabilities() const noexcept override { return _caps; }

  [[nodiscard]] bool supports(std::string_view path) const noexcept override
  {
    std::error_code ec;
    return std::filesystem::is_regular_file(std::filesystem::path{path}, ec) && !ec;
  }

  void shutdown() noexcept override
  {
    if (_stopped.exchange(true, std::memory_order_acq_rel)) { return; }
    _pool.wait();
  }

  [[nodiscard]] std::vector<byte_range> align_and_coalesce(
    std::span<byte_range const> ranges,
    std::optional<std::size_t> alignment) const noexcept override
  {
    std::vector<byte_range> result;
    try {
      auto const align = static_cast<std::int64_t>(std::max<std::size_t>(1, alignment.value_or(1)));
      auto const limit = std::numeric_limits<std::int64_t>::max();
      result.reserve(ranges.size());
      for (auto const& r : ranges) {
        if (r.offset() < 0 || r.size() <= 0) { continue; }
        auto const begin = r.offset() - r.offset() % align;
        auto end         = r.size() > limit - r.offset() ? limit : r.offset() + r.size();
        if (end % align != 0) {
          end = end > limit - (align - end % align) ? limit : end + (align - end % align);
        }
        result.emplace_back(begin, end - begin);
      }
      std::sort(result.begin(), result.end(), [](byte_range const& a, byte_range const& b) {
        return a.offset() < b.offset();
      });
      std::size_t out = 0;
      for (std::size_t i = 0; i < result.size(); ++i) {
        if (out != 0 && result[i].offset() <= result[out - 1].offset() + result[out - 1].size()) {
          auto const end  = std::max(result[out - 1].offset() + result[out - 1].size(),
                                    result[i].offset() + result[i].size());
          result[out - 1] = byte_range{result[out - 1].offset(), end - result[out - 1].offset()};
        } else {
          result[out++] = result[i];
        }
      }
      result.resize(out);
    } catch (...) {
      result.clear();
    }
    return result;
  }

  [[nodiscard]] std::shared_ptr<io_object> open(std::string path, open_options const&) override
  {
    FileHandle file{path, "r", FileHandle::m644, _options.compat_mode};
    return std::make_shared<local_object>(std::move(path), std::move(file), false);
  }

  [[nodiscard]] std::shared_ptr<io_object> open_for_write(
    std::string path, write_open_options const& options) override
  {
    // FileHandle's flag strings cannot express "create but keep" or "must not
    // exist", so the file is created here first where needed.
    char const* flags = "r+";
    switch (options.mode) {
      case write_mode::create_or_truncate: flags = "w+"; break;
      case write_mode::open_existing: break;
      case write_mode::create_or_open:
      case write_mode::create_new: {
        auto const extra = options.mode == write_mode::create_new ? O_EXCL : 0;
        auto const fd =
          ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | extra, options.permissions);
        if (fd < 0) { throw_errno("cufile_posix: open_for_write"); }
        ::close(fd);
        break;
      }
    }
    if (options.mode == write_mode::open_existing && ::access(path.c_str(), F_OK) != 0) {
      throw_errno("cufile_posix: open_for_write");
    }
    FileHandle file{path, flags, options.permissions, _options.compat_mode};
    if (options.size_hint != 0) {
      // Best effort: a filesystem that cannot preallocate is not an error.
      std::ignore =
        ::fallocate(file.fd(), FALLOC_FL_KEEP_SIZE, 0, static_cast<off_t>(options.size_hint));
    }
    return std::make_shared<local_object>(std::move(path), std::move(file), true);
  }

  std::size_t host_read(io_object const& obj,
                        std::size_t offset,
                        std::size_t size,
                        std::uint8_t* dst) override
  {
    auto const& file = typed(obj);
    auto const end   = file.size();
    if (offset >= end || size == 0) { return 0; }
    auto const n = std::min(size, end - offset);
    read_exact(file.fd(), dst, n, offset);
    return n;
  }

  std::size_t host_write(io_object const& obj,
                         std::size_t offset,
                         std::size_t size,
                         std::uint8_t const* src) override
  {
    auto const& file = writable(obj);
    write_exact(file.fd(), src, size, offset);
    file.note_written(offset + size);
    return size;
  }

  void flush(io_object const& obj) override
  {
    if (::fdatasync(writable(obj).fd()) != 0) { throw_errno("cufile_posix: fdatasync"); }
  }

  [[nodiscard]] exec::semi_future<std::size_t> read(
    io_object const& obj, std::vector<prepared_io_slice>&& slices) noexcept override
  {
    return submit(obj, std::move(slices), false);
  }

  [[nodiscard]] exec::semi_future<std::size_t> write(
    io_object const& obj, std::vector<prepared_io_slice>&& slices) noexcept override
  {
    return submit(obj, std::move(slices), true);
  }

  [[nodiscard]] exec::semi_future<void> flush_async(io_object const& obj,
                                                    io_priority priority) noexcept override
  {
    try {
      std::vector<prepared_io_slice> slices;
      slices.push_back(make_flush_slice(priority));
      return submit(obj, std::move(slices), true).defer_value([](std::size_t) {});
    } catch (...) {
      return exec::make_semi_future<void>(exec::try_t<void>(std::current_exception()));
    }
  }

 private:
  [[nodiscard]] static local_object const& typed(io_object const& obj)
  {
    auto const* file = dynamic_cast<local_object const*>(&obj);
    if (file == nullptr) {
      throw std::invalid_argument("cufile_posix: the object was not opened by this engine");
    }
    return *file;
  }

  [[nodiscard]] static local_object const& writable(io_object const& obj)
  {
    auto const& file = typed(obj);
    if (!file.writable()) {
      throw std::invalid_argument("cufile_posix: the object was not opened for writing");
    }
    return file;
  }

  /// Give every fragment of @p slices back as failed.
  static void release(std::vector<prepared_io_slice>& slices) noexcept
  {
    for (auto& slice : slices) {
      if (slice.on_complete != nullptr) { (*slice.on_complete)(slice.h_buffer.fragments(), false); }
    }
  }

  /// Build the request and its operations. Throws before anything is queued.
  [[nodiscard]] std::shared_ptr<request> plan(io_object const& obj,
                                              std::vector<prepared_io_slice>&& slices,
                                              bool write_api)
  {
    auto const& file = write_api ? writable(obj) : typed(obj);
    auto req         = std::make_shared<request>();
    req->owner       = obj.shared_from_this();
    req->object      = &file;
    auto const eof   = file.size();

    std::size_t n_ops = 0;
    for (auto const& slice : slices) {
      if (slice.is_write() != write_api && !slice.is_flush()) {
        throw std::invalid_argument("cufile_posix: read and write slices cannot be mixed");
      }
      if (slice.has_device_request()) { std::ignore = resolve_stream(slice.d_buffer.stream); }
      if (slice.is_fragmented()) {
        n_ops += slice.h_buffer.fragments().size();
      } else {
        n_ops += 1 + slice.rng.size / _options.max_operation_bytes;
      }
    }
    req->ops.reserve(n_ops);

    for (std::size_t i = 0; i < slices.size(); ++i) {
      auto const& slice = slices[i];
      if (slice.is_flush()) {
        req->ops.push_back(plan_op{i, no_fragment, 0, 0});
        continue;
      }
      if (slice.is_write()) {
        if (slice.rng.size == 0) { throw std::invalid_argument("cufile_posix: empty write slice"); }
        if (slice.has_host_request() == slice.has_device_request() || slice.is_fragmented()) {
          throw std::invalid_argument("cufile_posix: a write slice has exactly one source");
        }
        req->total_bytes += slice.rng.size;
        for (std::size_t done = 0; done < slice.rng.size;) {
          auto const n = std::min(_options.max_operation_bytes, slice.rng.size - done);
          req->ops.push_back(plan_op{i, no_fragment, slice.rng.offset + done, n});
          done += n;
        }
        continue;
      }
      // A read: clamp to the size recorded when the file was opened.
      auto const clamped = intersect(slice.rng, range{0, eof});
      req->total_bytes += clamped.size;
      if (slice.is_fragmented()) {
        auto const fragments = slice.h_buffer.fragments();
        for (std::size_t f = 0; f < fragments.size(); ++f) {
          if (fragments[f].data == nullptr) {
            throw std::invalid_argument("cufile_posix: a fragment has no memory");
          }
          req->ops.push_back(
            plan_op{i, f, fragments[f].offset, intersect(fragments[f].rng(), range{0, eof}).size});
        }
        continue;
      }
      if (clamped.empty()) {
        // Nothing to read, but the slice still settles through one operation.
        req->ops.push_back(plan_op{i, no_fragment, slice.rng.offset, 0});
        continue;
      }
      for (std::size_t done = 0; done < clamped.size;) {
        auto const n = std::min(_options.max_operation_bytes, clamped.size - done);
        req->ops.push_back(plan_op{i, no_fragment, clamped.offset + done, n});
        done += n;
      }
    }
    req->slices = std::move(slices);
    req->remaining.store(req->ops.size(), std::memory_order_release);
    return req;
  }

  exec::semi_future<std::size_t> submit(io_object const& obj,
                                        std::vector<prepared_io_slice>&& slices,
                                        bool write_api) noexcept
  {
    std::shared_ptr<request> req;
    try {
      if (_stopped.load(std::memory_order_acquire)) {
        throw std::system_error(std::make_error_code(std::errc::operation_canceled),
                                "cufile_posix: the engine is shut down");
      }
      req = plan(obj, std::move(slices), write_api);
    } catch (...) {
      release(slices);
      return exec::make_semi_future<std::size_t>(std::current_exception());
    }
    auto future = req->promise.get_semi_future();
    if (req->ops.empty()) {
      req->promise.set_value(0);
      return future;
    }
    // From here on nothing is reported by throwing: an operation that cannot be
    // queued is failed in place, so the future still settles.
    for (std::size_t i = 0; i < req->ops.size(); ++i) {
      try {
        _pool.detach_task([this, req, i] { run(req, i); });
      } catch (...) {
        finish(*req, i, std::current_exception(), false);
      }
    }
    return future;
  }

  void finish(request& req, std::size_t index, std::exception_ptr error, bool host_ok) noexcept
  {
    auto const& op = req.ops[index];
    auto& slice    = req.slices[op.slice];
    if (slice.on_complete != nullptr) {
      if (op.fragment != no_fragment) {
        fragment_handle const handle = slice.h_buffer.fragments()[op.fragment].handle;
        (*slice.on_complete)(std::span<fragment_handle const>{&handle, 1}, host_ok);
      } else {
        (*slice.on_complete)(std::span<fragment_handle const>{}, host_ok);
      }
    }
    if (error != nullptr) { req.fail(std::move(error)); }
    req.finish_one();
  }

  void run(std::shared_ptr<request> const& req, std::size_t index) noexcept
  {
    bool host_ok = false;
    try {
      auto const& op    = req->ops[index];
      auto const& slice = req->slices[op.slice];
      if (slice.is_flush()) {
        if (::fdatasync(req->object->fd()) != 0) { throw_errno("cufile_posix: fdatasync"); }
      } else if (slice.is_write()) {
        write_op(*req, op, slice);
      } else if (!read_op(req, index, op, slice, host_ok)) {
        return;  // parked until staging memory comes back
      }
      host_ok = true;
    } catch (...) {
      finish(*req, index, std::current_exception(), host_ok);
      return;
    }
    finish(*req, index, nullptr, true);
  }

  /// Returns false if the operation was parked; it runs again later.
  bool read_op(std::shared_ptr<request> const& req,
               std::size_t index,
               plan_op const& op,
               prepared_io_slice const& slice,
               bool& host_ok)
  {
    auto const& file = *req->object;
    if (op.size == 0) { return true; }

    std::uint8_t* host = nullptr;
    if (op.fragment != no_fragment) {
      host = slice.h_buffer.fragments()[op.fragment].data;
    } else if (slice.is_contiguous()) {
      host = std::get<std::uint8_t*>(slice.h_buffer.buffer) + (op.offset - slice.rng.offset);
    }

    if (host != nullptr) {
      read_exact(file.fd(), host, op.size, op.offset);
      host_ok = true;
      if (slice.has_device_request()) {
        // Only the part of this operation inside the slice's range goes to the device.
        auto const part = intersect(range{op.offset, op.size}, slice.rng);
        if (!part.empty()) {
          copy_to_device(slice.d_buffer,
                         part.offset - slice.rng.offset,
                         host + (part.offset - op.offset),
                         part.size);
        }
      }
      return true;
    }

    auto* device      = slice.d_buffer.data + (op.offset - slice.rng.offset);
    auto const stream = resolve_stream(slice.d_buffer.stream);
    if (file.uses_cufile()) {
      CUcontext ctx{};
      auto const [base, base_size, device_offset] = get_alloc_info(device, &ctx);
      std::ignore                                 = base_size;
      PushAndPopContext scope(ctx);
      // cuFile does not order against streams: let the work queued before this read finish.
      wait_for_stream(stream);
      auto const n = file.file().read(base, op.size, op.offset, device_offset, false);
      if (n != op.size) {
        throw std::system_error(std::make_error_code(std::errc::io_error),
                                "cufile_posix: short cuFile read");
      }
      return true;
    }

    auto lease = acquire_staging(req, index, op.size);
    if (lease == nullptr) { return false; }
    staging_release releaser{*this};
    PushAndPopContext scope(get_context_from_pointer(device));
    std::size_t done = 0;
    for (auto* block : lease->blocks()) {
      if (done == op.size) { break; }
      auto const n = std::min(lease->block_size(), op.size - done);
      read_exact(file.fd(), reinterpret_cast<std::uint8_t*>(block), n, op.offset + done);
      KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
                               convert_void2deviceptr(device + done), block, n, stream),
                             std::runtime_error);
      done += n;
    }
    // The staging blocks go back only after the copies read them.
    wait_for_stream(stream);
    return true;
  }

  void write_op(request& req, plan_op const& op, prepared_io_slice const& slice)
  {
    auto const& file = *req.object;
    if (slice.is_contiguous()) {
      auto const* src =
        std::get<std::uint8_t*>(slice.h_buffer.buffer) + (op.offset - slice.rng.offset);
      write_exact(file.fd(), src, op.size, op.offset);
      file.note_written(op.offset + op.size);
      return;
    }

    auto* device      = slice.d_buffer.data + (op.offset - slice.rng.offset);
    auto const stream = resolve_stream(slice.d_buffer.stream);
    if (file.uses_cufile()) {
      CUcontext ctx{};
      auto const [base, base_size, device_offset] = get_alloc_info(device, &ctx);
      std::ignore                                 = base_size;
      PushAndPopContext scope(ctx);
      wait_for_stream(stream);
      auto const n = file.file().write(base, op.size, op.offset, device_offset, false);
      if (n != op.size) {
        throw std::system_error(std::make_error_code(std::errc::io_error),
                                "cufile_posix: short cuFile write");
      }
      file.note_written(op.offset + op.size);
      return;
    }

    // A device write is not parked: it waits here, on this engine's own thread.
    if (_options.staging == nullptr) {
      throw std::runtime_error("cufile_posix: a device write needs a staging source");
    }
    PushAndPopContext scope(get_context_from_pointer(device));
    for (std::size_t done = 0; done < op.size;) {
      auto result =
        _options.staging->acquire(std::min(op.size - done, _options.staging->block_size()));
      if (result.status != staging_status::ok) {
        throw std::runtime_error("cufile_posix: no staging memory for a device write");
      }
      auto* block  = result.lease->blocks()[0];
      auto const n = std::min(result.lease->block_size(), op.size - done);
      KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyDtoHAsync(
                               block, convert_void2deviceptr(device + done), n, stream),
                             std::runtime_error);
      wait_for_stream(stream);
      write_exact(file.fd(), reinterpret_cast<std::uint8_t const*>(block), n, op.offset + done);
      done += n;
    }
    file.note_written(op.offset + op.size);
  }

  void copy_to_device(device_buffer const& d_buffer,
                      std::size_t device_offset,
                      std::uint8_t const* src,
                      std::size_t size)
  {
    auto const stream = resolve_stream(d_buffer.stream);
    auto* dst         = d_buffer.data + device_offset;
    PushAndPopContext scope(get_context_from_pointer(dst));
    KVIKIO_CUDA_DRIVER_TRY(
      cudaAPI::instance().MemcpyHtoDAsync(convert_void2deviceptr(dst), src, size, stream),
      std::runtime_error);
    wait_for_stream(stream);
  }

  // -- Staging: an operation that finds no block waits for one of this engine's
  // own operations to give blocks back, and fails if none holds any. -------------

  struct parked {
    std::shared_ptr<request> req;
    std::size_t index;
  };

  struct staging_release {
    cufile_posix_engine& engine;
    ~staging_release() { engine.staging_returned(); }
  };

  [[nodiscard]] std::unique_ptr<staging_lease> acquire_staging(std::shared_ptr<request> const& req,
                                                               std::size_t index,
                                                               std::size_t bytes)
  {
    if (_options.staging == nullptr) {
      throw std::runtime_error("cufile_posix: a device read of this file needs a staging source");
    }
    _staging_holders.fetch_add(1, std::memory_order_acq_rel);
    // The source is called without holding any lock of this engine.
    auto result = _options.staging->acquire(bytes);
    if (result.status == staging_status::ok) { return std::move(result.lease); }

    std::unique_lock lock(_parked_mutex);
    auto const others = _staging_holders.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (result.status == staging_status::unavailable && others > 0) {
      _parked.push_back(parked{req, index});
      return nullptr;
    }
    lock.unlock();
    throw std::runtime_error(result.status == staging_status::unavailable
                               ? "cufile_posix: staging memory is exhausted"
                               : "cufile_posix: the staging source failed");
  }

  void staging_returned() noexcept
  {
    std::optional<parked> next;
    {
      std::lock_guard lock(_parked_mutex);
      _staging_holders.fetch_sub(1, std::memory_order_acq_rel);
      if (!_parked.empty()) {
        next = std::move(_parked.front());
        _parked.pop_front();
      }
    }
    if (!next) { return; }
    try {
      _pool.detach_task([this, req = next->req, index = next->index] { run(req, index); });
    } catch (...) {
      finish(*next->req, next->index, std::current_exception(), false);
    }
  }

  cufile_posix_options _options;
  engine_capabilities _caps;
  std::atomic<bool> _stopped{false};
  std::atomic<std::size_t> _staging_holders{0};
  std::mutex _parked_mutex;
  std::deque<parked> _parked;
  ThreadPool _pool;  ///< last: its threads use everything above
};

}  // namespace

std::shared_ptr<engine> make_cufile_posix_engine(engine_options const& options)
{
  if (auto const* typed = dynamic_cast<cufile_posix_options const*>(&options); typed != nullptr) {
    return std::make_shared<cufile_posix_engine>(*typed);
  }
  cufile_posix_options defaults;
  defaults.staging = options.staging;
  return std::make_shared<cufile_posix_engine>(std::move(defaults));
}

}  // namespace kvikio::io
