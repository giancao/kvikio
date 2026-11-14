/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <kvikio/io/engine.hpp>

namespace kvikio::test {

/// An object of a fixed size whose byte at offset i is `i % 251`.
class fake_object final : public io::io_object {
 public:
  fake_object(std::string path, std::size_t size) : _path(std::move(path)), _size(size) {}

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return _size; }

 private:
  std::string _path;
  std::size_t _size;
};

inline std::uint8_t fake_byte(std::size_t offset) noexcept
{
  return static_cast<std::uint8_t>(offset % 251);
}

inline void fake_fill(std::uint8_t* dst, std::size_t offset, std::size_t size) noexcept
{
  for (std::size_t i = 0; i < size; ++i) {
    dst[i] = fake_byte(offset + i);
  }
}

/**
 * @brief A read-only engine that serves `fake_object` from memory.
 *
 * It completes every request before returning, and treats each fragment as one
 * physical operation, so tests can observe the per-operation completion calls.
 */
class fake_engine : public io::engine {
 public:
  explicit fake_engine(std::string name, bool fail_start = false)
    : _name(std::move(name)), _fail_start(fail_start)
  {
    _caps.vector_host_read       = true;
    _caps.fragmented_host_buffer = true;
  }

  [[nodiscard]] std::string_view name() const noexcept override { return _name; }

  [[nodiscard]] io::engine_capabilities const& capabilities() const noexcept override
  {
    return _caps;
  }

  [[nodiscard]] bool supports(std::string_view path) const noexcept override
  {
    return path.starts_with("fake://");
  }

  void start() override
  {
    if (_fail_start) { throw std::runtime_error("io_uring is disabled in this process"); }
    started.store(true);
  }

  void shutdown() noexcept override { shutdowns.fetch_add(1); }

  [[nodiscard]] std::vector<io::byte_range> align_and_coalesce(
    std::span<io::byte_range const> ranges, std::optional<std::size_t>) const noexcept override
  {
    return {ranges.begin(), ranges.end()};
  }

  [[nodiscard]] std::shared_ptr<io::io_object> open(std::string path,
                                                    io::open_options const& options) override
  {
    return std::make_shared<fake_object>(std::move(path), options.known_size.value_or(4096));
  }

  std::size_t host_read(io::io_object const& obj,
                        std::size_t offset,
                        std::size_t size,
                        std::uint8_t* dst) override
  {
    auto const n = offset >= obj.size() ? 0 : std::min(size, obj.size() - offset);
    fake_fill(dst, offset, n);
    return n;
  }

  [[nodiscard]] exec::semi_future<std::size_t> read(
    io::io_object const& /*obj*/, std::vector<io::prepared_io_slice>&& slices) noexcept override
  {
    std::size_t total = 0;
    for (auto& s : slices) {
      if (s.is_contiguous()) {
        fake_fill(std::get<std::uint8_t*>(s.h_buffer.buffer), s.rng.offset, s.rng.size);
        total += s.rng.size;
        continue;
      }
      for (auto const& f : s.h_buffer.fragments()) {
        fake_fill(f.data, f.offset, f.size);
        total += f.size;
        if (s.on_complete != nullptr) {
          io::fragment_handle const handle = f.handle;
          (*s.on_complete)(std::span<io::fragment_handle const>{&handle, 1}, true);
        }
      }
    }
    return exec::make_semi_future(exec::try_t<std::size_t>(total));
  }

  std::atomic<bool> started{false};
  std::atomic<int> shutdowns{0};

 private:
  std::string _name;
  bool _fail_start;
  io::engine_capabilities _caps;
};

}  // namespace kvikio::test
