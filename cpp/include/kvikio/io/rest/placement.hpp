/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file placement.hpp
 * @brief Which CPUs the reactors of the `uring_http` engine would be pinned to.
 *
 * In `libkvikio.so`, not in the engine plugin: a host asks before it creates
 * any engine, to keep its own thread pools off those CPUs.
 */

#include <kvikio/io/nic_topology.hpp>
#include <kvikio/shim/utils.hpp>

#include <cstddef>
#include <optional>
#include <vector>

namespace kvikio::io::rest {

/**
 * @brief The CPUs a pool of @p n_reactors reactors would pin to; ascending, distinct.
 *
 * Computed from the calling thread's affinity and @p nic exactly as the engine
 * does when it starts. Empty when the pool would not pin: @p pin_to_irq_cpus is
 * off, there is no @p nic, or its interrupts are not steered.
 *
 * @param n_reactors      configured reactor count; 0 sizes the pool from @p nic
 * @param pin_to_irq_cpus the engine's `pin_to_irq_cpus` setting
 * @param nic             the egress interface, as the host discovered it
 */
[[nodiscard]] KVIKIO_EXPORT std::vector<int> planned_reactor_cpus(
  std::size_t n_reactors, bool pin_to_irq_cpus, std::optional<nic_topology> const& nic);

}  // namespace kvikio::io::rest
