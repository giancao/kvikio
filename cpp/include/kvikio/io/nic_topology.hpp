/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file nic_topology.hpp
 * @brief What an engine needs to know about the network interface it sends through.
 *
 * A plain description, filled in by the host: KvikIO does not discover it. An
 * engine that talks to an object store uses it to size its reactor pool and to
 * pin its reactor threads to the CPUs the interface's interrupts are delivered
 * to. Without one, such an engine uses its default pool size and pins nothing.
 *
 * The fields mirror cuCascade's `memory::nic_topology`, which discovers them
 * from sysfs.
 */

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace kvikio::io {

/// One queue interrupt of an interface.
struct nic_queue_irq {
  int irq{-1};
  int queue{-1};
  /// CPUs the interrupt is delivered to (the effective mask).
  std::vector<int> cpus;
  /// CPUs the interrupt was asked to go to (the requested mask), when known.
  std::vector<int> requested_cpus;
};

struct nic_topology {
  std::string interface;
  std::string pci_address;
  int numa_node{-1};  ///< -1 = unknown
  int mtu{0};
  unsigned rx_queues{0};
  unsigned tx_queues{0};
  std::vector<nic_queue_irq> queue_irqs;
  /// Union of the delivery CPUs of the queue interrupts; ascending, distinct.
  std::vector<int> irq_cpus;
  /// CPUs of the interface's NUMA node; ascending, distinct.
  std::vector<int> node_cpus;

  static constexpr std::size_t default_steered_max_cpus = 1;

  /**
   * @brief Whether the queue interrupts are steered rather than spread.
   *
   * Steered iff there is at least one queue interrupt, every one is delivered
   * to between 1 and @p max_cpus_per_irq CPUs and its requested mask (when
   * known) is no wider, and the union of the delivery CPUs is a strict subset
   * of @ref node_cpus. An interrupt whose mask covers the whole node is
   * "spread" even when the kernel happens to deliver it to one CPU of that
   * mask, and so is an interface whose interrupt CPUs together cover the node:
   * there is no subset of cores to align with.
   */
  [[nodiscard]] bool irqs_steered(
    std::size_t max_cpus_per_irq = default_steered_max_cpus) const noexcept
  {
    if (queue_irqs.empty() || irq_cpus.empty() || node_cpus.empty()) { return false; }
    for (auto const& q : queue_irqs) {
      if (q.cpus.empty() || q.cpus.size() > max_cpus_per_irq) { return false; }
      // The effective mask alone proves nothing: x86 delivers every interrupt
      // to one CPU the kernel picks from the requested mask. Steering is the
      // operator (or irqbalance) narrowing the requested mask itself.
      if (q.requested_cpus.size() > max_cpus_per_irq) { return false; }
    }
    // Strict subset: every interrupt CPU belongs to the node and the node has more.
    if (irq_cpus.size() >= node_cpus.size()) { return false; }
    return std::includes(node_cpus.begin(), node_cpus.end(), irq_cpus.begin(), irq_cpus.end());
  }
};

/// Format CPU ids as a kernel CPU list (`"0-3,8,10-11"`).
[[nodiscard]] inline std::string format_cpu_list(std::vector<int> const& cpus_in)
{
  std::vector<int> cpus(cpus_in);
  std::sort(cpus.begin(), cpus.end());
  cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
  std::ostringstream out;
  for (std::size_t i = 0; i < cpus.size();) {
    auto j = i;
    while (j + 1 < cpus.size() && cpus[j + 1] == cpus[j] + 1) {
      ++j;
    }
    if (i != 0) { out << ','; }
    out << cpus[i];
    if (j > i) { out << '-' << cpus[j]; }
    i = j + 1;
  }
  return out.str();
}

}  // namespace kvikio::io
