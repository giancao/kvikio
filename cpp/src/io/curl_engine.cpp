/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <curl/curl.h>
#include <kvikio/detail/http_retry.hpp>
#include <kvikio/detail/multi_poll_reactor.hpp>
#include <kvikio/detail/remote_callback.hpp>
#include <kvikio/detail/utils.hpp>
#include <kvikio/error.hpp>
#include <kvikio/exec/semi_future.hpp>
#include <kvikio/io/curl_engine.hpp>
#include <kvikio/io/detail/io_utils.hpp>
#include <kvikio/io/staging_source.hpp>
#include <kvikio/io/types.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::io {

namespace {

constexpr std::size_t no_fragment = std::numeric_limits<std::size_t>::max();

/// The stream a copy is queued on. Null means the legacy default stream; the
/// per-thread default stream cannot be honoured from a reactor thread.
CUstream resolve_stream(CUstream stream)
{
  if (stream == nullptr) { return CU_STREAM_LEGACY; }
  if (stream == CU_STREAM_PER_THREAD) {
    throw std::invalid_argument(
      "curl: the per-thread default stream is not supported; pass an explicit stream");
  }
  return stream;
}

/// The context that owns the device memory of @p d_buffer. Called on the
/// submitting thread: a reactor thread has no current context to find it with.
CUcontext context_of(device_buffer const& d_buffer)
{
  if (d_buffer.device_id >= 0) { return get_primary_cuda_context(d_buffer.device_id); }
  return get_context_from_pointer(d_buffer.data);
}

bool has_scheme(std::string_view path, std::string_view scheme) noexcept
{
  if (path.size() <= scheme.size()) { return false; }
  for (std::size_t i = 0; i < scheme.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(path[i])) != scheme[i]) { return false; }
  }
  return true;
}

/// Collects the last `ETag` response header.
std::size_t collect_etag(char* buffer, std::size_t size, std::size_t nitems, void* userdata)
{
  auto const bytes = size * nitems;
  std::string_view line{buffer, bytes};
  constexpr std::string_view name{"etag:"};
  if (has_scheme(line, name)) {
    line.remove_prefix(name.size());
    while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) {
      line.remove_prefix(1);
    }
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
      line.remove_suffix(1);
    }
    try {
      *static_cast<std::string*>(userdata) = std::string{line};
    } catch (...) {
      return 0;  // aborts the transfer
    }
  }
  return bytes;
}

/// A remote object opened by this engine.
class remote_object final : public io_object {
 public:
  remote_object(std::string path, RemoteHandle handle, std::size_t size, std::string etag)
    : _path(std::move(path)), _handle(std::move(handle)), _size(size), _etag(std::move(etag))
  {
  }

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return _size; }
  [[nodiscard]] std::string_view validation_tag() const noexcept override { return _etag; }
  [[nodiscard]] bool validation_tag_is_strong() const noexcept override
  {
    // RFC 9110: a strong validator is a quoted string without the weak prefix.
    return _etag.size() >= 2 && _etag.front() == '"' && _etag.back() == '"';
  }

  /// The endpoint configures a curl handle for this object. It keeps no
  /// per-request state, and `RemoteHandle::pread` uses it the same way.
  [[nodiscard]] RemoteEndpoint& endpoint() const noexcept
  {
    return const_cast<RemoteEndpoint&>(_handle.endpoint());
  }

 private:
  std::string _path;
  RemoteHandle _handle;
  std::size_t _size;
  std::string _etag;
};

/// What the engine and its in-flight requests share. A request may outlive the
/// engine object, so it holds this, not the engine.
struct shared_state {
  std::shared_ptr<staging_source> staging;
  /// Staged operations of this engine that hold a block: when there are none,
  /// an exhausted staging source will not recover by waiting.
  std::atomic<std::size_t> staging_holders{0};
  std::atomic<bool> stopped{false};
};

/// One range request of a read.
struct plan_op {
  std::size_t slice{0};
  std::size_t fragment{no_fragment};
  std::size_t offset{0};  ///< object offset
  std::size_t size{0};    ///< bytes to transfer; already clamped to the end of the object
  /// Where the bytes land in host memory; null when the operation is staged.
  std::uint8_t* host{nullptr};
};

/// Everything one read() call needs until it settles.
struct request {
  std::shared_ptr<shared_state> state;
  std::shared_ptr<io_object const> owner;
  std::vector<prepared_io_slice> slices;
  std::vector<plan_op> ops;
  /// The CUDA context of each slice's device memory (null without one).
  std::vector<CUcontext> contexts;
  exec::promise<std::size_t> promise;
  std::size_t total_bytes{0};
  std::atomic<std::size_t> remaining{0};
  std::mutex error_mutex;
  std::exception_ptr error;

  /// Operation @p index is done: hand its fragment back and settle the future
  /// after the last one. @p host_ok says whether the host bytes are valid.
  void finish(std::size_t index, std::exception_ptr e, bool host_ok) noexcept
  {
    auto const& op = ops[index];
    auto& slice    = slices[op.slice];
    if (slice.on_complete != nullptr) {
      if (op.fragment != no_fragment) {
        fragment_handle const handle = slice.h_buffer.fragments()[op.fragment].handle;
        (*slice.on_complete)(std::span<fragment_handle const>{&handle, 1}, host_ok);
      } else {
        (*slice.on_complete)(std::span<fragment_handle const>{}, host_ok);
      }
    }
    if (e != nullptr) {
      std::lock_guard lock(error_mutex);
      if (error == nullptr) { error = std::move(e); }
    }
    if (remaining.fetch_sub(1, std::memory_order_acq_rel) != 1) { return; }
    if (error != nullptr) {
      promise.set_exception(error);
    } else {
      promise.set_value(total_bytes);
    }
  }
};

/// Runs when the host-to-device copy of one operation has finished.
struct copy_done {
  std::shared_ptr<request> req;
  std::size_t index{0};
  std::unique_ptr<staging_lease> lease;
  CURLM* multi{nullptr};

  /// Called by the CUDA driver on one of its threads; must not call CUDA.
  static void CUDA_CB run(void* user_data)
  {
    std::unique_ptr<copy_done> self{static_cast<copy_done*>(user_data)};
    self->settle(nullptr);
  }

  void settle(std::exception_ptr e) noexcept
  {
    if (lease != nullptr) {
      lease.reset();
      req->state->staging_holders.fetch_sub(1, std::memory_order_acq_rel);
      // A block is back: let the reactor admit an operation that was waiting for one.
      if (multi != nullptr) { std::ignore = curl_multi_wakeup(multi); }
    }
    req->finish(index, std::move(e), true);
  }
};

/// Settles one operation's transfer for the multi-poll reactor.
class op_hooks final : public detail::RemoteMultiTransferHooks {
 public:
  op_hooks(std::shared_ptr<request> req, std::size_t index) : _req(std::move(req)), _index(index) {}

  [[nodiscard]] Admission admit(detail::RemoteMultiTransfer& transfer) noexcept override
  {
    auto const& op = _req->ops[_index];
    if (op.host != nullptr) { return Admission::ADMITTED; }
    // A staged operation: one staging block receives the response body.
    auto& state = *_req->state;
    auto got    = state.staging->acquire(op.size);
    if (got.status == staging_status::unavailable) {
      if (state.staging_holders.load(std::memory_order_acquire) != 0) {
        return Admission::DEFERRED;  // one of this engine's operations will give a block back
      }
      std::exception_ptr e;
      try {
        auto const what = "curl: staging exhausted for " + _req->owner->object_path() + " (" +
                          std::to_string(op.size) + " bytes)";
        e = state.staging->exhausted_error(what.c_str());
      } catch (...) {
        e = std::current_exception();
      }
      _req->finish(_index, std::move(e), false);
      return Admission::FAILED;
    }
    if (got.status != staging_status::ok || got.lease == nullptr || got.lease->blocks().empty() ||
        got.lease->block_size() < op.size) {
      _req->finish(_index,
                   std::make_exception_ptr(
                     std::runtime_error("curl: the staging source did not lend a usable block")),
                   false);
      return Admission::FAILED;
    }
    _lease = std::move(got.lease);
    state.staging_holders.fetch_add(1, std::memory_order_acq_rel);
    transfer.ctx.pinned_buffer = _lease->blocks().front();
    return Admission::ADMITTED;
  }

  void reset_for_retry(detail::RemoteMultiTransfer& transfer) noexcept override
  {
    release_lease();
    transfer.ctx.pinned_buffer = nullptr;
  }

  void complete(detail::RemoteMultiTransfer& /*transfer*/, CURLM* multi) override
  {
    auto const& op    = _req->ops[_index];
    auto const& slice = _req->slices[op.slice];
    if (!slice.has_device_request()) {
      _req->finish(_index, nullptr, true);
      return;
    }
    // Only the part of this operation inside the slice's range goes to the device.
    auto const part = intersect(range{op.offset, op.size}, slice.rng);
    if (part.empty()) {
      release_lease();
      _req->finish(_index, nullptr, true);
      return;
    }
    auto const* source =
      (op.host != nullptr ? op.host : reinterpret_cast<std::uint8_t*>(_lease->blocks().front())) +
      (part.offset - op.offset);
    auto* target      = slice.d_buffer.data + (part.offset - slice.rng.offset);
    auto const stream = resolve_stream(slice.d_buffer.stream);

    auto done   = std::make_unique<copy_done>();
    done->req   = _req;
    done->index = _index;
    done->multi = multi;

    PushAndPopContext const scope(_req->contexts[op.slice]);
    auto status = cudaAPI::instance().MemcpyHtoDAsync(
      convert_void2deviceptr(target), source, part.size, stream);
    if (status == CUDA_SUCCESS) {
      // The operation settles, and its staging block goes back, when the copy has run. Nobody
      // waits for it: the driver calls back.
      done->lease = std::move(_lease);
      status      = cudaAPI::instance().LaunchHostFunc(stream, &copy_done::run, done.get());
      if (status == CUDA_SUCCESS) {
        std::ignore = done.release();
        return;
      }
      // The copy is queued but there is no callback: let it finish reading the block.
      std::ignore = cudaAPI::instance().StreamSynchronize(stream);
      _lease      = std::move(done->lease);
    }
    release_lease();
    char const* text = nullptr;
    std::ignore      = cudaAPI::instance().GetErrorString(status, &text);
    _req->finish(_index,
                 std::make_exception_ptr(
                   std::runtime_error(std::string{"curl: copying to the device failed: "} +
                                      (text != nullptr ? text : "unknown error"))),
                 true);
  }

  void fail(std::exception_ptr eptr) noexcept override
  {
    release_lease();
    _req->finish(_index, std::move(eptr), false);
  }

 private:
  void release_lease() noexcept
  {
    if (_lease == nullptr) { return; }
    _lease.reset();
    _req->state->staging_holders.fetch_sub(1, std::memory_order_acq_rel);
  }

  std::shared_ptr<request> _req;
  std::size_t _index;
  std::unique_ptr<staging_lease> _lease;
};

class curl_engine final : public engine {
 public:
  explicit curl_engine(curl_options options)
    : _options(std::move(options)), _state(std::make_shared<shared_state>())
  {
    if (_options.max_operation_bytes == 0) { _options.max_operation_bytes = 16UL << 20; }
    _state->staging              = _options.staging;
    _caps.device_read            = true;
    _caps.host_to_device_read    = true;
    _caps.vector_host_read       = true;
    _caps.device_range_read      = true;
    _caps.fragmented_host_buffer = true;
    _caps.write                  = false;
    _caps.device_write           = false;
    _caps.priority               = false;
    _caps.prefers_bulk_io        = false;
    _caps.min_alignment          = 1;
    // Nearby ranges are worth one request: a round trip costs more than the gap.
    _caps.merge_gap            = 512UL << 10;
    _caps.max_concurrent_scans = 8;
    _caps.staging_block_size   = _state->staging != nullptr ? _state->staging->block_size() : 0;
  }

  ~curl_engine() override { shutdown(); }

  [[nodiscard]] std::string_view name() const noexcept override { return curl_engine_name; }

  [[nodiscard]] engine_capabilities const& capabilities() const noexcept override { return _caps; }

  [[nodiscard]] bool supports(std::string_view path) const noexcept override
  {
    return has_scheme(path, "s3://") || has_scheme(path, "http://") || has_scheme(path, "https://");
  }

  /// Later requests are rejected. Requests in flight belong to the process-wide
  /// reactors and settle on their own; they share nothing with this object.
  void shutdown() noexcept override { _state->stopped.store(true, std::memory_order_release); }

  [[nodiscard]] std::vector<byte_range> align_and_coalesce(
    std::span<byte_range const> ranges,
    std::optional<std::size_t> alignment) const noexcept override
  {
    try {
      return io_utils::align_and_coalesce(ranges, std::max<std::size_t>(1, alignment.value_or(1)));
    } catch (...) {
      return {};
    }
  }

  /**
   * Opens with one HEAD request, which gives the size and the ETag. With a
   * known size nothing is sent, and the object then has no validator.
   */
  [[nodiscard]] std::shared_ptr<io_object> open(std::string path,
                                                open_options const& options) override
  {
    // Naming the type keeps RemoteHandle from probing the object itself; the
    // placeholder size keeps it from sending its own HEAD.
    auto const type = infer_remote_endpoint_type(path);
    auto handle     = RemoteHandle::open(path, type, std::nullopt, std::size_t{0});
    if (options.known_size.has_value()) {
      return std::make_shared<remote_object>(
        std::move(path), std::move(handle), static_cast<std::size_t>(*options.known_size), "");
    }
    auto& endpoint = const_cast<RemoteEndpoint&>(handle.endpoint());
    if (type == RemoteEndpointType::S3_PRESIGNED_URL) {
      // A presigned URL is signed for GET only; the endpoint knows how to size it.
      auto const size = endpoint.get_file_size();
      return std::make_shared<remote_object>(std::move(path), std::move(handle), size, "");
    }
    std::string etag;
    auto curl = create_curl_handle();
    endpoint.setopt(curl);
    curl.setopt(CURLOPT_NOBODY, 1L);
    curl.setopt(CURLOPT_FOLLOWLOCATION, 1L);
    curl.setopt(CURLOPT_HEADERFUNCTION, &collect_etag);
    curl.setopt(CURLOPT_HEADERDATA, static_cast<void*>(&etag));
    curl.perform([&etag] { etag.clear(); });
    curl_off_t length = -1;
    curl.getinfo(CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &length);
    if (length < 0) {
      throw std::runtime_error("curl: cannot get the size of " + path +
                               ": the server sent no content length");
    }
    return std::make_shared<remote_object>(
      std::move(path), std::move(handle), static_cast<std::size_t>(length), std::move(etag));
  }

  /// One range request on the calling thread.
  std::size_t host_read(io_object const& obj,
                        std::size_t offset,
                        std::size_t size,
                        std::uint8_t* dst) override
  {
    auto const& object = typed(obj);
    auto const end     = object.size();
    if (offset >= end || size == 0) { return 0; }
    auto const n = std::min(size, end - offset);
    auto curl    = create_curl_handle();
    object.endpoint().setopt(curl);
    object.endpoint().setup_range_request(curl, offset, n);
    detail::CallbackContext ctx{dst, n};
    curl.setopt(CURLOPT_WRITEFUNCTION, &detail::callback_host_memory);
    curl.setopt(CURLOPT_WRITEDATA, static_cast<void*>(&ctx));
    try {
      curl.perform([&ctx] { ctx.reset_for_retry(); });
    } catch (std::runtime_error const& e) {
      if (ctx.overflow_error) {
        throw std::overflow_error(std::string{"curl: the server returned more than the range "
                                              "asked for; maybe it does not support ranges? ["} +
                                  e.what() + "]");
      }
      throw;
    }
    return n;
  }

  [[nodiscard]] exec::semi_future<std::size_t> read(
    io_object const& obj, std::vector<prepared_io_slice>&& slices) noexcept override
  {
    std::shared_ptr<request> req;
    std::vector<std::unique_ptr<detail::RemoteMultiTransfer>> transfers;
    try {
      if (_state->stopped.load(std::memory_order_acquire)) {
        throw std::system_error(std::make_error_code(std::errc::operation_canceled),
                                "curl: the engine is shut down");
      }
      req       = plan(obj, slices);
      transfers = make_transfers(req);
    } catch (...) {
      // Nothing was queued. If the slices already moved into the request, they are there.
      release(req != nullptr && !req->slices.empty() ? req->slices : slices);
      return exec::make_semi_future<std::size_t>(std::current_exception());
    }
    auto future = req->promise.get_semi_future();
    if (transfers.empty()) {
      req->promise.set_value(req->total_bytes);
      return future;
    }
    // From here on nothing is reported by throwing: every transfer settles
    // through its hooks, also when the reactors are dead.
    try {
      detail::MultiReactorPool::instance().submit_pread(std::move(transfers));
    } catch (...) {
      // The pool could not take the batch. Transfers it did not consume are
      // failed here; consumed ones are settled by the pool.
      auto const error = std::current_exception();
      for (auto& transfer : transfers) {
        if (transfer != nullptr) { transfer->report_failed(error); }
      }
    }
    return future;
  }

 private:
  [[nodiscard]] static remote_object const& typed(io_object const& obj)
  {
    auto const* object = dynamic_cast<remote_object const*>(&obj);
    if (object == nullptr) {
      throw std::invalid_argument("curl: the object was not opened by this engine");
    }
    return *object;
  }

  /// Give every fragment of @p slices back as failed.
  static void release(std::vector<prepared_io_slice>& slices) noexcept
  {
    for (auto& slice : slices) {
      if (slice.on_complete != nullptr) { (*slice.on_complete)(slice.h_buffer.fragments(), false); }
    }
  }

  /// Build the request and its operations. Throws before anything is queued;
  /// the slices are moved into the request last.
  [[nodiscard]] std::shared_ptr<request> plan(io_object const& obj,
                                              std::vector<prepared_io_slice>& slices) const
  {
    auto const& object = typed(obj);
    auto req           = std::make_shared<request>();
    req->state         = _state;
    req->owner         = obj.shared_from_this();
    auto const eof     = object.size();
    req->contexts.resize(slices.size(), nullptr);

    for (std::size_t i = 0; i < slices.size(); ++i) {
      auto const& slice = slices[i];
      if (slice.is_write() || slice.is_flush()) {
        throw std::invalid_argument("curl: a read request holds a write or flush slice");
      }
      if (slice.has_device_request()) {
        std::ignore      = resolve_stream(slice.d_buffer.stream);
        req->contexts[i] = context_of(slice.d_buffer);
      }
      auto const clamped = intersect(slice.rng, range{0, eof});
      req->total_bytes += clamped.size;

      if (slice.is_fragmented()) {
        auto const fragments = slice.h_buffer.fragments();
        for (std::size_t f = 0; f < fragments.size(); ++f) {
          if (fragments[f].data == nullptr) {
            throw std::invalid_argument("curl: a fragment has no memory");
          }
          req->ops.push_back(plan_op{i,
                                     f,
                                     fragments[f].offset,
                                     intersect(fragments[f].rng(), range{0, eof}).size,
                                     fragments[f].data});
        }
        continue;
      }
      if (clamped.empty()) {
        // Nothing to read, but the slice still settles through one operation.
        req->ops.push_back(plan_op{i, no_fragment, slice.rng.offset, 0, nullptr});
        continue;
      }
      if (slice.is_contiguous()) {
        auto* const base = std::get<std::uint8_t*>(slice.h_buffer.buffer);
        for (std::size_t done = 0; done < clamped.size;) {
          auto const n = std::min(_options.max_operation_bytes, clamped.size - done);
          req->ops.push_back(plan_op{i,
                                     no_fragment,
                                     clamped.offset + done,
                                     n,
                                     base + (clamped.offset + done - slice.rng.offset)});
          done += n;
        }
        continue;
      }
      // Device only: staged, one staging block per range request.
      if (_state->staging == nullptr || _state->staging->block_size() == 0) {
        throw std::invalid_argument(
          "curl: a device read needs a staging source (engine_options::staging)");
      }
      auto const block = _state->staging->block_size();
      for (std::size_t done = 0; done < clamped.size;) {
        auto const n = std::min(block, clamped.size - done);
        req->ops.push_back(plan_op{i, no_fragment, clamped.offset + done, n, nullptr});
        done += n;
      }
    }
    req->remaining.store(req->ops.size(), std::memory_order_release);
    req->slices = std::move(slices);
    return req;
  }

  /// One transfer per operation that has bytes to fetch. An operation without
  /// bytes is finished here, which may already settle the request.
  [[nodiscard]] static std::vector<std::unique_ptr<detail::RemoteMultiTransfer>> make_transfers(
    std::shared_ptr<request> const& req)
  {
    auto const& object      = static_cast<remote_object const&>(*req->owner);
    auto const retry_policy = std::make_shared<detail::HttpRetryPolicy const>();
    std::vector<std::unique_ptr<detail::RemoteMultiTransfer>> transfers;
    transfers.reserve(req->ops.size());
    for (std::size_t i = 0; i < req->ops.size(); ++i) {
      auto const& op = req->ops[i];
      if (op.size == 0) { continue; }
      auto transfer  = std::make_unique<detail::RemoteMultiTransfer>();
      transfer->curl = std::make_unique<CurlHandle>(LibCurl::instance().get_handle(),
                                                    detail::fix_conda_file_path_hack(__FILE__),
                                                    KVIKIO_STRINGIFY(__LINE__));
      object.endpoint().setopt(*transfer->curl);
      object.endpoint().setup_range_request(*transfer->curl, op.offset, op.size);
      transfer->ctx.size     = op.size;
      transfer->retry_policy = retry_policy;
      if (op.host != nullptr) {
        transfer->ctx.buf = reinterpret_cast<char*>(op.host);
        transfer->curl->setopt(CURLOPT_WRITEFUNCTION, &detail::callback_host_memory);
      } else {
        // The staging block is attached when the reactor admits the transfer.
        transfer->curl->setopt(CURLOPT_WRITEFUNCTION, &detail::callback_pinned_buffer);
      }
      transfer->curl->setopt(CURLOPT_WRITEDATA, static_cast<void*>(&transfer->ctx));
      transfer->hooks = std::make_unique<op_hooks>(req, i);
      transfers.push_back(std::move(transfer));
    }
    // Only now, with every allocation behind us, finish the operations without bytes.
    for (std::size_t i = 0; i < req->ops.size(); ++i) {
      if (req->ops[i].size == 0 && !transfers.empty()) { req->finish(i, nullptr, true); }
    }
    if (transfers.empty()) {
      // Every operation is empty: hand the fragments back; read() settles the future.
      for (std::size_t i = 0; i < req->ops.size(); ++i) {
        auto const& op = req->ops[i];
        auto& slice    = req->slices[op.slice];
        if (slice.on_complete == nullptr) { continue; }
        if (op.fragment != no_fragment) {
          fragment_handle const handle = slice.h_buffer.fragments()[op.fragment].handle;
          (*slice.on_complete)(std::span<fragment_handle const>{&handle, 1}, true);
        } else {
          (*slice.on_complete)(std::span<fragment_handle const>{}, true);
        }
      }
    }
    return transfers;
  }

  curl_options _options;
  std::shared_ptr<shared_state> _state;
  engine_capabilities _caps;
};

}  // namespace

std::shared_ptr<engine> make_curl_engine(engine_options const& options)
{
  if (auto const* typed = dynamic_cast<curl_options const*>(&options); typed != nullptr) {
    return std::make_shared<curl_engine>(*typed);
  }
  curl_options defaults;
  defaults.staging = options.staging;
  return std::make_shared<curl_engine>(std::move(defaults));
}

}  // namespace kvikio::io
