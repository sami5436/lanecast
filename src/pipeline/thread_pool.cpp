// Persistent thread pool and row slicing.
//
// A job is (fn, ctx, count). The pool state word packs a job generation
// (bits 48..63), the job's count (24..47) and the next unclaimed index
// (0..23). Participants claim an index with a compare-exchange on the whole
// word, so a claim only succeeds while that exact job is live and its fields
// are stable; a late worker can never run an exhausted job. The caller
// publishes a job with one store, works alongside the workers, then waits
// for the completion count.
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "lanecast/pipeline.hpp"

namespace lanecast {
namespace {

constexpr int kSpinBeforePark = 4000;
constexpr int kSpinBeforeYield = 1000;
constexpr int kMaxJobCount = (1 << 24) - 1;

inline void cpu_relax() noexcept {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
  asm volatile("pause" ::: "memory");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

constexpr std::uint64_t pack(std::uint64_t generation, std::uint64_t count) noexcept {
  return ((generation & 0xFFFF) << 48) | (count << 24);
}
constexpr std::uint32_t generation_of(std::uint64_t s) noexcept {
  return static_cast<std::uint32_t>(s >> 48);
}
constexpr std::uint32_t count_of(std::uint64_t s) noexcept {
  return static_cast<std::uint32_t>((s >> 24) & 0xFFFFFF);
}
constexpr std::uint32_t next_of(std::uint64_t s) noexcept {
  return static_cast<std::uint32_t>(s & 0xFFFFFF);
}

}  // namespace

struct ThreadPool::Impl {
  using Call = void (*)(void*, int);

  int total = 1;
  std::vector<std::thread> workers;

  std::atomic<std::uint64_t> state{0};
  std::atomic<Call> call{nullptr};
  std::atomic<void*> ctx{nullptr};
  std::atomic<int> done{0};

  std::atomic<int> sleepers{0};
  std::atomic<bool> stop{false};
  std::mutex mu;
  std::condition_variable cv;

  // Claims and runs indices until the current job is exhausted.
  void work() {
    for (;;) {
      std::uint64_t s = state.load(std::memory_order_acquire);
      if (next_of(s) >= count_of(s)) return;
      const Call c = call.load(std::memory_order_relaxed);
      void* const o = ctx.load(std::memory_order_relaxed);
      if (!state.compare_exchange_weak(s, s + 1, std::memory_order_acq_rel,
                                       std::memory_order_relaxed)) {
        continue;
      }
      c(o, static_cast<int>(next_of(s)));
      done.fetch_add(1, std::memory_order_release);
    }
  }

  void worker_loop() {
    std::uint32_t seen = 0;
    for (;;) {
      int spins = 0;
      std::uint32_t generation;
      while ((generation = generation_of(state.load(std::memory_order_acquire))) == seen &&
             !stop.load(std::memory_order_relaxed)) {
        if (++spins < kSpinBeforePark) {
          cpu_relax();
          continue;
        }
        std::unique_lock<std::mutex> lock(mu);
        sleepers.fetch_add(1);
        cv.wait(lock, [&] { return stop.load() || generation_of(state.load()) != seen; });
        sleepers.fetch_sub(1);
        spins = 0;
      }
      if (stop.load(std::memory_order_relaxed)) return;
      seen = generation;
      work();
    }
  }

  void run(int n, Call c, void* o) {
    done.store(0, std::memory_order_relaxed);
    call.store(c, std::memory_order_relaxed);
    ctx.store(o, std::memory_order_relaxed);
    const std::uint32_t generation = generation_of(state.load(std::memory_order_relaxed));
    state.store(pack(generation + 1, static_cast<std::uint64_t>(n)), std::memory_order_seq_cst);
    if (sleepers.load() > 0) {
      { std::lock_guard<std::mutex> lock(mu); }
      cv.notify_all();
    }
    work();
    int spins = 0;
    while (done.load(std::memory_order_acquire) < n) {
      if (++spins < kSpinBeforeYield) {
        cpu_relax();
      } else {
        std::this_thread::yield();
      }
    }
  }
};

ThreadPool::ThreadPool(int threads) : impl_(std::make_unique<Impl>()) {
  if (threads <= 0) threads = static_cast<int>(std::thread::hardware_concurrency());
  impl_->total = std::max(1, threads);
  impl_->workers.reserve(static_cast<std::size_t>(impl_->total - 1));
  for (int i = 1; i < impl_->total; ++i) {
    impl_->workers.emplace_back([p = impl_.get()] { p->worker_loop(); });
  }
}

ThreadPool::~ThreadPool() {
  impl_->stop.store(true);
  { std::lock_guard<std::mutex> lock(impl_->mu); }
  impl_->cv.notify_all();
  for (std::thread& t : impl_->workers) t.join();
}

int ThreadPool::size() const noexcept { return impl_->total; }

void ThreadPool::parallel_for(int count, FunctionRef<void(int)> fn) {
  if (count <= 0) return;
  if (count == 1 || impl_->total == 1 || count > kMaxJobCount) {
    for (int i = 0; i < count; ++i) fn(i);
    return;
  }
  impl_->run(
      count, [](void* o, int i) { (*static_cast<FunctionRef<void(int)>*>(o))(i); }, &fn);
}

Status run_sliced(ThreadPool* pool, int height, int alignment,
                  FunctionRef<Status(RowRange)> fn, int slice_rows) {
  if (height <= 0 || alignment <= 0) return Status::invalid_argument;
  const int threads = pool != nullptr ? pool->size() : 1;
  if (threads == 1) return fn(RowRange{0, height});

  const auto align_up = [alignment](int v) { return (v + alignment - 1) / alignment * alignment; };
  int rows = align_up(std::max(slice_rows, 1));
  if ((height + rows - 1) / rows < threads) rows = align_up((height + threads - 1) / threads);
  const int slices = (height + rows - 1) / rows;
  if (slices == 1) return fn(RowRange{0, height});

  std::atomic<int> failure{static_cast<int>(Status::ok)};
  pool->parallel_for(slices, [&](int i) {
    const Status s = fn(RowRange{i * rows, std::min(height, (i + 1) * rows)});
    if (s != Status::ok) {
      int expected = static_cast<int>(Status::ok);
      failure.compare_exchange_strong(expected, static_cast<int>(s));
    }
  });
  return static_cast<Status>(failure.load());
}

}  // namespace lanecast
