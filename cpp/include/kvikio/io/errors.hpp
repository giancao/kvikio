/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace kvikio::io {

/**
 * @brief The engine exists, but this operation of it is not implemented.
 *
 * Reported by every entry point an engine declares unsupported in its
 * @c engine_capabilities (for example the write and flush entry points of a
 * read-only engine). A @c std::system_error with @c std::errc::not_supported,
 * so callers that catch @c std::system_error and compare its code keep working;
 * the message names the engine and operation and says "not yet implemented".
 */
class not_implemented_error : public std::system_error {
 public:
  /// @p what names the engine and the operation, e.g. "curl: host_write".
  explicit not_implemented_error(std::string const& what)
    : std::system_error(std::make_error_code(std::errc::not_supported),
                        what + ": not yet implemented")
  {
  }
};

/**
 * @brief Opening a local file paired descriptors of two different files.
 *
 * A local reactor opens a buffered descriptor and reopens it for @c O_DIRECT;
 * when the two do not name the same file (device and inode) on every attempt,
 * for example because the path is being replaced concurrently, the open fails
 * with this error rather than read the data of one file and the footer of
 * another.
 */
class descriptor_mismatch_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief Raised by @c s3_request_authorizer implementations when credential
 *        acquisition or signing fails.
 *
 * Surfaces from:
 *   - missing or malformed static credentials (caught at provider construction
 *     or first call),
 *   - misconfigured endpoint / region,
 *   - underlying signing-library failure (HMAC / SHA256),
 *   - upstream credential broker errors (in future refresh-aware impls).
 *
 * Backends translate this into the broader IO error path of the caller.
 * Future IO-layer error types share this header.
 */
class credential_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief A read found the object to be a different version from the one its
 *        datasource was opened on.
 *
 * Raised by the REST reactor when a conditional range GET is refused (412) or
 * when an accepted response does not carry the validator the open recorded.
 * Nothing from such a response is published; the error is terminal for the
 * read and is not retried.  @c observed_tag() is the response's ETag as
 * received, or empty when the response carried none or was refused (412).
 */
class object_changed_error : public std::runtime_error {
 public:
  object_changed_error(std::string object_path, std::string expected_tag, std::string observed_tag)
    : std::runtime_error(describe(object_path, observed_tag)),
      _object_path(std::move(object_path)),
      _expected_tag(std::move(expected_tag)),
      _observed_tag(std::move(observed_tag))
  {
  }

  [[nodiscard]] std::string const& object_path() const noexcept { return _object_path; }
  [[nodiscard]] std::string const& expected_tag() const noexcept { return _expected_tag; }
  [[nodiscard]] std::string const& observed_tag() const noexcept { return _observed_tag; }

 private:
  static std::string describe(std::string const& path, std::string const& observed)
  {
    return "object " + path + " changed since it was opened" +
           (observed.empty() ? " (no matching validator in the response)"
                             : " (response validator differs)");
  }

  std::string _object_path;
  std::string _expected_tag;
  std::string _observed_tag;
};

/**
 * @brief The backend does not implement this operation yet.
 *
 * Reported by every write and flush entry point of a backend that serves
 * reads only (the REST context and reactor), and by
 * the @c ioctx write front ends when the backend cannot write at all.  A
 * @c std::system_error with @c std::errc::not_supported, so callers that catch
 * @c std::system_error and compare its code keep working; the message names
 * the backend and operation and says "not yet implemented".
 */

/**
 * @brief An operation needed staging memory, the source had none, and nothing
 *        the engine holds would free any.
 *
 * What @c staging_source::exhausted_error makes by default. It is a
 * @c std::bad_alloc, so callers that treat memory exhaustion as retryable see
 * it as such. A staging source may report the condition in a type of its own
 * instead.
 */
class staging_exhausted_error : public std::bad_alloc {
 public:
  explicit staging_exhausted_error(std::string what) : _what(std::move(what)) {}
  [[nodiscard]] char const* what() const noexcept override { return _what.c_str(); }

 private:
  std::string _what;
};

/**
 * @brief Selecting an engine failed: the name is unknown, its plugin could not
 *        be loaded, or the engine could not be created or started.
 *
 * The selection that was in effect before the failed call stays in effect.
 */
class engine_selection_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace kvikio::io
