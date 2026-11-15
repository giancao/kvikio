/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace kvikio::test {

/**
 * @brief A minimal HTTP/1.1 server on 127.0.0.1 serving one byte payload at every path.
 *
 * Answers HEAD and GET, honours `Range: bytes=a-b`, keeps connections alive and sends an ETag.
 * Paths starting with `/missing` answer 404. One thread per connection; enough for tests.
 */
class LoopbackHttpServer {
 public:
  explicit LoopbackHttpServer(std::vector<std::uint8_t> payload, std::string etag = "\"v1\"")
    : _payload(std::move(payload)), _etag(std::move(etag))
  {
    _listen_fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (_listen_fd < 0) { throw std::runtime_error("LoopbackHttpServer: socket"); }
    int const one = 1;
    ::setsockopt(_listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    socklen_t len        = sizeof(addr);
    if (::bind(_listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        ::listen(_listen_fd, 64) != 0 ||
        ::getsockname(_listen_fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
      ::close(_listen_fd);
      throw std::runtime_error("LoopbackHttpServer: bind/listen");
    }
    _port     = ntohs(addr.sin_port);
    _acceptor = std::thread([this] { accept_loop(); });
  }

  ~LoopbackHttpServer()
  {
    _stop.store(true);
    ::shutdown(_listen_fd, SHUT_RDWR);
    ::close(_listen_fd);
    if (_acceptor.joinable()) { _acceptor.join(); }
    std::vector<std::thread> workers;
    {
      std::lock_guard lock(_mutex);
      for (int const fd : _client_fds) {
        ::shutdown(fd, SHUT_RDWR);
      }
      workers.swap(_workers);
    }
    for (auto& worker : workers) {
      if (worker.joinable()) { worker.join(); }
    }
  }

  LoopbackHttpServer(LoopbackHttpServer const&)            = delete;
  LoopbackHttpServer& operator=(LoopbackHttpServer const&) = delete;

  [[nodiscard]] std::string url(std::string const& path = "/object.bin") const
  {
    return "http://127.0.0.1:" + std::to_string(_port) + path;
  }

  [[nodiscard]] std::size_t head_requests() const noexcept { return _heads.load(); }
  [[nodiscard]] std::size_t get_requests() const noexcept { return _gets.load(); }

 private:
  void accept_loop()
  {
    while (!_stop.load()) {
      int const fd = ::accept4(_listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
      if (fd < 0) {
        if (_stop.load()) { return; }
        continue;
      }
      std::lock_guard lock(_mutex);
      _client_fds.push_back(fd);
      _workers.emplace_back([this, fd] { serve(fd); });
    }
  }

  static bool send_all(int fd, char const* data, std::size_t size)
  {
    while (size != 0) {
      auto const n = ::send(fd, data, size, MSG_NOSIGNAL);
      if (n <= 0) { return false; }
      data += n;
      size -= static_cast<std::size_t>(n);
    }
    return true;
  }

  void serve(int fd)
  {
    std::string buffer;
    char chunk[4096];
    for (;;) {
      auto end = buffer.find("\r\n\r\n");
      while (end == std::string::npos) {
        auto const n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) {
          ::close(fd);
          return;
        }
        buffer.append(chunk, static_cast<std::size_t>(n));
        end = buffer.find("\r\n\r\n");
      }
      auto const head = buffer.substr(0, end);
      buffer.erase(0, end + 4);
      if (!respond(fd, head)) {
        ::close(fd);
        return;
      }
    }
  }

  bool respond(int fd, std::string const& head)
  {
    bool const is_head = head.rfind("HEAD ", 0) == 0;
    bool const is_get  = head.rfind("GET ", 0) == 0;
    auto const path    = head.substr(head.find(' ') + 1, 8);
    if (is_head) { ++_heads; }
    if (is_get) { ++_gets; }
    if (!is_head && !is_get) {
      std::string const reply = "HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n";
      return send_all(fd, reply.data(), reply.size());
    }
    if (path == "/missing") {
      std::string const reply = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
      return send_all(fd, reply.data(), reply.size());
    }
    std::size_t first = 0;
    std::size_t last  = _payload.empty() ? 0 : _payload.size() - 1;
    bool ranged       = false;
    for (char const* name : {"Range: bytes=", "range: bytes="}) {
      auto const at = head.find(name);
      if (at == std::string::npos) { continue; }
      unsigned long long a = 0;
      unsigned long long b = 0;
      if (std::sscanf(head.c_str() + at + 13, "%llu-%llu", &a, &b) == 2) {
        first  = static_cast<std::size_t>(a);
        last   = std::min(static_cast<std::size_t>(b), last);
        ranged = true;
      }
    }
    if (ranged && (first >= _payload.size() || first > last)) {
      std::string const reply = "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\n\r\n";
      return send_all(fd, reply.data(), reply.size());
    }
    auto const length = _payload.empty() ? 0 : last - first + 1;
    std::string reply = ranged ? "HTTP/1.1 206 Partial Content\r\n" : "HTTP/1.1 200 OK\r\n";
    reply += "Content-Length: " + std::to_string(length) + "\r\n";
    reply += "Accept-Ranges: bytes\r\n";
    reply += "ETag: " + _etag + "\r\n";
    if (ranged) {
      reply += "Content-Range: bytes " + std::to_string(first) + "-" + std::to_string(last) + "/" +
               std::to_string(_payload.size()) + "\r\n";
    }
    reply += "\r\n";
    if (!send_all(fd, reply.data(), reply.size())) { return false; }
    if (is_head || length == 0) { return true; }
    return send_all(fd, reinterpret_cast<char const*>(_payload.data()) + first, length);
  }

  std::vector<std::uint8_t> _payload;
  std::string _etag;
  int _listen_fd{-1};
  std::uint16_t _port{0};
  std::atomic<bool> _stop{false};
  std::atomic<std::size_t> _heads{0};
  std::atomic<std::size_t> _gets{0};
  std::thread _acceptor;
  std::mutex _mutex;
  std::vector<int> _client_fds;
  std::vector<std::thread> _workers;
};

}  // namespace kvikio::test
