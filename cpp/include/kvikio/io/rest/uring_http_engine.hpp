/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <kvikio/exec/admission_control.hpp>
#include <kvikio/io/detail/reactor_engine.hpp>
#include <kvikio/io/engine.hpp>
#include <kvikio/io/nic_topology.hpp>
#include <kvikio/io/rest/object_store_lister.hpp>
#include <kvikio/io/rest/rest_reactor.hpp>
#include <kvikio/io/rest/s3/list_parser.hpp>
#include <kvikio/io/rest/uring_http_api.hpp>
#include <kvikio/shim/utils.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace kvikio::io::rest {

// ---------------------------------------------------------------------------
// uring_http_engine
// ---------------------------------------------------------------------------

/**
 * @brief The `uring_http` engine: object stores (s3://) over HTTP/1.1 on io_uring.
 *
 * It lives in the plugin `libkvikio_engine_uring_http.so`. Select it by name
 * through @c kvikio::io::engine_registry with @ref uring_http_options, and reach
 * what is beyond the engine interface (LIST, batched footer resolution, transport
 * counters) through @ref uring_http_api. Moved from cuCascade's `rest_ioctx`;
 * error messages still name it that way.
 *
 * Owns a pool of @c rest_reactor workers (load-balanced by the base) that share
 * one @p authorizer.  Overrides @c create_io_object to resolve an object's size
 * via a blocking HEAD before constructing the @c rest_io_object — the static
 * reactor factory cannot do this since it needs the authorizer + a round-trip.
 *
 * Control-plane calls (the HEAD or footer probe of an open, LIST pages, a
 * @ref resolve_footer_objects batch) block the caller and run on one reactor
 * picked per call: the least backlogged (@c rest_reactor::queued_bytes), ties
 * rotated across calls, so any reactor of a multi-reactor pool may serve one.
 * The exchanges use that reactor's connections while its worker accepts work.
 * Before the pool is started (@c start), after it stopped, or when the call is
 * made from a reactor thread (e.g. a completion callback, which must not wait on
 * a reactor loop that may be its own), they run instead on a short-lived
 * private helper thread with its own io_uring ring and new connections — far
 * heavier, so start the pool before issuing control-plane calls.  io_uring
 * never runs on the caller's thread.
 */
namespace detail {

/// What the reactors of an engine point at. A base class of the engine so that
/// it is destroyed after them.
struct uring_http_resources {
  std::shared_ptr<staging_source> staging;
};

}  // namespace detail

class KVIKIO_EXPORT uring_http_engine final
  : private detail::uring_http_resources,
    public kvikio::io::detail::reactor_engine<rest_reactor>,
    public uring_http_api {
 public:
  /// Build a pool of @p n_reactors reactors, all sharing @p ctx (one context per
  /// pool: it carries the per-reactor @c config, the presigning authorizer, and
  /// the pinned bounce-staging source).  @p nic is the egress interface as the
  /// host discovered it; without one the pool has the default size and is not
  /// pinned.  @p staging, when given, is kept alive for as long as the engine;
  /// otherwise the caller keeps the context's staging source alive.
  ///
  /// @p n_reactors is a total; 0 sizes the pool from the egress NIC (one reactor per IRQ CPU,
  /// pinned to it, when the NIC's queue IRQs are steered to a strict subset of its NUMA node's
  /// cores; the default 2 otherwise).  A pinned reactor's priority is raised only with
  /// @c config::raise_reactor_priority (off by default).
  /// @throws std::invalid_argument when the config is invalid (@c config::validate) or
  /// @c rest.fan_out exceeds a positive @p n_reactors (the default fan-out of 2 is clamped to a
  /// one-reactor pool instead, as the former hard-coded fan-out was).
  uring_http_engine(std::size_t n_reactors,
                    std::shared_ptr<rest_reactor::reactor_context> ctx,
                    std::optional<nic_topology> nic         = std::nullopt,
                    std::shared_ptr<staging_source> staging = nullptr);

  ~uring_http_engine() override;

  /// Number of reactors in the pool, and the counters and queued bytes of one
  /// of them (diagnostics; @p i must be less than the count).
  [[nodiscard]] std::size_t reactor_count() const noexcept { return _reactors.size(); }
  [[nodiscard]] transport_stats reactor_stats(std::size_t i) const { return _reactors.at(i)->stats(); }
  [[nodiscard]] std::size_t reactor_queued_bytes(std::size_t i) const
  {
    return _reactors.at(i)->queued_bytes();
  }

  /// This pool's placement (plan and interface), computed at construction.
  [[nodiscard]] detail::pool_placement const& placement() const noexcept { return *_placement; }

  /// Counters of the whole pool: the reactors' @c transport_stats summed (extremes for the MSS
  /// and in-flight peaks).  A snapshot; safe while the pool runs.
  [[nodiscard]] transport_stats stats() const noexcept override;

  /// Stream a bucket's ListObjectsV2 pages under @p prefix to @p sink, one call
  /// per page (a page holds at most 1000 entries, so peak memory is one page
  /// regardless of bucket population).  @p sink returns false to stop early —
  /// no further LIST requests are issued.  @p page_size is clamped to [1,1000]
  /// (0 and >1000 mean 1000).  Throws (never truncates) on a truncated page
  /// without a continuation token, and once more than @p max_scanned entries
  /// have been scanned across pages (bounds time / request count on a prefix
  /// whose population dwarfs the caller's matches).
  void list_objects_paged(std::string_view bucket,
                          std::string_view prefix,
                          std::size_t page_size,
                          std::function<bool(s3::list_objects_v2_page const&)> const& sink,
                          std::optional<std::size_t> max_scanned = std::nullopt) override;

  /// Whole-listing convenience over @c list_objects_paged: every object under
  /// @p prefix, in document order, with sizes.  Throws (never truncates) when
  /// the accumulated entries would exceed @p max_keys — a partial key set would
  /// resolve a glob to a silently incomplete table.
  [[nodiscard]] std::vector<s3::list_entry> list_objects(
    std::string_view bucket,
    std::string_view prefix,
    std::size_t page_size               = 1000,
    std::optional<std::size_t> max_keys = std::nullopt) override;

  /// The configured matched cap (@c config.list_max_matches) — exposed so a
  /// glob layer one level up can bound its match set without a reactor handle.
  /// Falls back to the built-in default when the pool is empty (never in
  /// practice).
  [[nodiscard]] std::size_t list_max_matches() const override;

  /// Resolve many objects' footers concurrently.  Per-entry semantics are
  /// IDENTICAL to @c open_io_object(path, parquet_footer_probe): one verified
  /// suffix GET; 200/416/unverifiable-206 fall back to a HEAD supplying
  /// size+tag; the same retry policy per entry, never stalling siblings.  The
  /// whole batch runs on one reactor, picked like every control-plane call: the
  /// least-backlogged reactor of the pool, ties rotated across calls (not
  /// necessarily reactor 0).  Its exchanges use that reactor's connections
  /// while its worker accepts work; before the pool is started, after it
  /// stopped, or when called from a reactor thread, they run on a short-lived
  /// private helper thread with its own io_uring ring and connections (see the
  /// class notes).  @p on_result is invoked ON THE CALLER'S THREAD, SERIALLY,
  /// as each entry lands — completion order, no all-entries barrier.
  ///
  /// Every input occurrence is delivered exactly once (duplicates delivered
  /// per occurrence, disambiguated by index); no callback runs after the
  /// call returns.  On @p stop, in-flight transfers abort and every
  /// undelivered entry receives one std::system_error(operation_canceled) —
  /// including a batch cancelled while queued behind another batch (one
  /// active batch per ioctx; concurrent calls FIFO-serialize).  If
  /// @p on_result throws, the remaining entries are cancelled (delivered as
  /// canceled, their callback throws suppressed) and the first exception is
  /// rethrown after the sweep.  Throws directly only on submission errors:
  /// an empty batch, the API disabled via
  /// @c config::footer_resolve_max_inflight == 0, or an explicit
  /// @c footer_resolve_stash_budget smaller than @c footer_probe_bytes.  An
  /// unparsable or non-s3 path is a per-entry error, not a batch error.
  /// Should the transfer driver itself fail (its helper thread cannot be
  /// started, or it fails mid-batch), every undelivered entry is delivered as
  /// canceled before that failure is rethrown — exactly-once holds on every
  /// exit path.  This ioctx must outlive the call.
  void resolve_footer_objects(std::span<std::string const> paths,
                              std::function<void(footer_resolve_result)> const& on_result,
                              std::stop_token stop = {}) override;

  /// Live / high-water bytes reserved from the footer-resolve stash budget.
  /// Both are 0 when the batched footer API is disabled or has never run on
  /// this ioctx.  Safe to call while the pool is running.
  [[nodiscard]] std::size_t footer_stash_reserved_bytes() const noexcept override;
  [[nodiscard]] std::size_t footer_stash_reserved_peak_bytes() const noexcept override;

  /// Open every reactor's connection pool against @p bucket_url's bucket, so the
  /// query's first reads find pooled connections instead of paying TCP+TLS on
  /// the hot path.  Fans the work out to the reactors and returns immediately:
  /// each reactor's connection cache is thread-confined, so only its own worker
  /// can fill it.
  ///
  /// Rate-limited rather than run once, because what goes stale is the
  /// connection, not the bucket.  @c conn_max_age is a hard cap on reusing a
  /// pooled connection (its idle cutoff), so a pool warmed before an idle
  /// gap longer than that is cold again by the next query -- a warm-once flag
  /// would serve the first query and no other.  Re-warms when the endpoint
  /// changes or the last warm-up is at least @c conn_max_age old, which is free
  /// for back-to-back queries and self-correcting for spaced-out ones.
  void warmup(std::string_view bucket_url) noexcept override;

  // -- Writes: not yet implemented ------------------------------------------
  //
  // Unified IO POC: upstream PR #221 (S3 PUT / multipart upload) is not merged
  // into this branch.  Take it when S3 writes are needed.
  //
  // REST writes (S3 PUT / multipart upload) are not implemented yet.  Every
  // write and flush hook reports @c not_implemented_error
  // (@c std::errc::not_supported) without touching the network: the
  // synchronous ones throw, the asynchronous ones return a failed future, and
  // @c supports_write() / @c supports_device_write() stay false.

  std::size_t host_write(io_object const& obj,
                         std::size_t offset,
                         std::size_t size,
                         std::uint8_t const* src) override;

  [[nodiscard]] exec::semi_future<std::size_t> write(
    io_object const& obj, std::vector<prepared_io_slice>&& slices) noexcept override;

  void flush(io_object const& obj) override;

  [[nodiscard]] exec::semi_future<void> flush_async(io_object const& obj,
                                                    io_priority priority) noexcept override;

  /// Always throws @c not_implemented_error: no object is created, no HEAD is
  /// sent.
  [[nodiscard]] std::shared_ptr<io_object> open_for_write(
    std::string path, write_open_options const& options) override;

  /// Open @p path (s3://bucket/key).  A known size builds the object with no
  /// network; @c open_hint::parquet_footer_probe resolves the size and stashes
  /// the object's tail in one suffix-range GET; otherwise a HEAD resolves the
  /// size.  Throws on a non-s3 scheme or a failed request.
  [[nodiscard]] std::shared_ptr<io_object> open(std::string path,
                                                open_options const& options) override;

 private:
  /// This pool's placement (plan and NIC), computed at construction and shared with its
  /// reactors; never stored on the reactor_context, which several pools may share.
  std::shared_ptr<detail::pool_placement const> _placement;

  /// Plain open: parse @p path
  /// (s3://bucket/key), HEAD it for the size, and build a @c rest_io_object.
  /// Throws on a non-s3 scheme or a failed HEAD.
  std::shared_ptr<io_object> create_io_object(std::string path);

  /// Hinted open: @c open_hint::parquet_footer_probe resolves the size AND stashes
  /// the object's trailing bytes in one suffix-range GET, stashed on the
  /// returned io_object; every other hint falls back to the plain HEAD path above.
  std::shared_ptr<io_object> create_io_object(std::string path, open_hint hint);

  /// Known-size open: the caller already learned the object's size (e.g. from a
  /// ListObjectsV2 response), so the io_object is built with ZERO network — no
  /// HEAD, no probe.
  std::shared_ptr<io_object> create_io_object(std::string path, std::uint64_t known_size);

  uring_http_engine(std::shared_ptr<detail::pool_placement const> placement,
                    std::shared_ptr<rest_reactor::reactor_context> ctx,
                    std::shared_ptr<staging_source> staging);

  /// Resolve @p path with a single suffix-range GET: it discovers the size and
  /// stashes the object's trailing bytes on the returned io_object so cuDF's
  /// footer reads are served locally.  Falls back to a plain HEAD (no stash)
  /// when the response is unusable.
  std::shared_ptr<io_object> create_footer_probe_object(std::string path);

  object_store_lister _lister;
  std::atomic<std::size_t> _control_next{0};

  /// The effective resolve_footer_objects concurrency cap: the configured knob, or one
  /// reactor's max_connections under footer_resolve_auto (a batch runs on one reactor).  0 =
  /// the API is disabled.
  [[nodiscard]] std::size_t footer_resolve_inflight_cap() const;

  /// The reactor a control-plane call (HEAD, footer probe, LIST, footer batch) runs on: the
  /// least backlogged, ties rotated, so control traffic does not pile onto reactor 0.  The call
  /// itself then runs on that reactor's worker, or on a private helper when the worker does not
  /// accept work or the caller is a reactor thread.
  [[nodiscard]] rest_reactor& control_reactor();

  // Batched-footer-resolve coordination: one active batch per ioctx, later
  // calls FIFO-parked on the ticket queue (stop-aware — a queued batch whose
  // token fires is removed without ever becoming active).  _footer_budget is
  // created in the constructor and never reassigned, so the stash gauges may
  // read it without the mutex.
  mutable std::mutex _footer_resolve_mutex;
  std::condition_variable_any _footer_resolve_cv;
  std::deque<std::uint64_t> _footer_resolve_queue;
  std::uint64_t _footer_resolve_next_ticket{0};
  bool _footer_resolve_active{false};
  std::shared_ptr<exec::admission_control> _footer_budget;

  /// Warn once per path when @p obj has no strong ETag (none, weak or malformed).
  /// Such an object cannot be revalidated across opens, so its bytes and
  /// metadata are cached only within this open.  Returns @p obj unchanged.
  std::shared_ptr<io_object> warn_if_no_strong_etag(std::shared_ptr<io_object> obj);

  /// Guards the warm-up rate limiter.  Contended once per query at most, and
  /// never on a read path.
  std::mutex _warm_mtx;
  std::mutex _weak_etag_mtx;
  std::vector<std::string> _weak_etag_paths;  ///< paths already warned about, bounded
  /// Bucket the pools were last warmed against, and when.  An unset time means
  /// "never warmed", which no elapsed comparison can express.
  std::string _warmed_bucket;
  std::optional<std::chrono::steady_clock::time_point> _warmed_at;
};

/// Create the engine from @ref uring_http_options, which must carry an authorizer.
[[nodiscard]] KVIKIO_EXPORT std::shared_ptr<engine> make_uring_http_engine(
  engine_options const& options);

}  // namespace kvikio::io::rest
