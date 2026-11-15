/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>

/**
 * @file reactor_test_hooks.hpp
 * @brief TESTING ONLY: observation points of the REST reactor's worker loop, so tests can
 *        assert scheduling invariants deterministically (e.g. that work queued during a pass
 *        starts in the same pass, without an extra blocking wait).
 */

namespace kvikio::io::rest::testing {

struct reactor_events {
  /// A data GET attempt was handed to the transport (worker thread): the object key, the
  /// attempt number (0 = first) and the worker's blocking-wait count at that moment.
  std::function<void(std::string_view key, std::size_t attempt, std::uint64_t waits)>
    on_get_started;
  /// A data GET attempt completed (worker thread), before any retry or follow-up is scheduled:
  /// key, attempt, HTTP status (0 = transport failure), wait count.
  std::function<void(
    std::string_view key, std::size_t attempt, unsigned status, std::uint64_t waits)>
    on_get_finished;
  /// Called right before a data GET is handed to the transport (worker thread); a throw is
  /// treated as a failure of the transport's submit.
  std::function<void(std::string_view key)> before_get_submit;
};

/// Install @p events (nullptr removes them).  Not synchronized with running reactors: install
/// before starting the reactor under test and remove after shutting it down.
void set_reactor_events(reactor_events const* events) noexcept;

}  // namespace kvikio::io::rest::testing
