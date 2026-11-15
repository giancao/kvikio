/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <utility>
#include <vector>

#include <kvikio/io/staging_source.hpp>

namespace kvikio::test {

using kvikio::io::staging_lease;
using kvikio::io::staging_result;
using kvikio::io::staging_source;
using kvikio::io::staging_status;

/// A staging source for tests: a fixed number of page-aligned blocks from the heap.
class test_staging final : public staging_source {
 public:
  test_staging(std::size_t block_size, std::size_t n_blocks) : _block_size(block_size)
  {
    for (std::size_t i = 0; i < n_blocks; ++i) {
      // Page-aligned, so that an engine can use the blocks for O_DIRECT.
      void* block = std::aligned_alloc(4096, (block_size + 4095) / 4096 * 4096);
      if (block == nullptr) { throw std::bad_alloc{}; }
      _storage.emplace_back(block, &std::free);
      _free.push_back(static_cast<std::byte*>(block));
    }
    _total = n_blocks;
  }

  [[nodiscard]] std::size_t block_size() const noexcept override { return _block_size; }

  [[nodiscard]] staging_result acquire(std::size_t bytes) noexcept override
  {
    auto const need = (bytes + _block_size - 1) / _block_size;
    std::lock_guard lock(_mutex);
    ++acquires;
    staging_result result;
    if (need > _total) { return result; }
    if (need > _free.size()) {
      ++unavailable;
      result.status = staging_status::unavailable;
      return result;
    }
    std::vector<std::byte*> blocks(_free.end() - static_cast<std::ptrdiff_t>(need), _free.end());
    _free.resize(_free.size() - need);
    result.lease  = std::make_unique<lease>(*this, std::move(blocks));
    result.status = staging_status::ok;
    return result;
  }

  [[nodiscard]] std::size_t free_blocks()
  {
    std::lock_guard lock(_mutex);
    return _free.size();
  }

  int acquires{0};
  int unavailable{0};

 private:
  class lease final : public staging_lease {
   public:
    lease(test_staging& owner, std::vector<std::byte*> blocks)
      : _owner(owner), _blocks(std::move(blocks))
    {
    }
    ~lease() override
    {
      std::lock_guard lock(_owner._mutex);
      _owner._free.insert(_owner._free.end(), _blocks.begin(), _blocks.end());
    }
    [[nodiscard]] std::span<std::byte* const> blocks() const noexcept override { return _blocks; }
    [[nodiscard]] std::size_t block_size() const noexcept override { return _owner._block_size; }

   private:
    test_staging& _owner;
    std::vector<std::byte*> _blocks;
  };

  std::size_t _block_size;
  std::size_t _total{0};
  std::mutex _mutex;
  std::vector<std::unique_ptr<void, void (*)(void*)>> _storage;
  std::vector<std::byte*> _free;
};

}  // namespace kvikio::test
