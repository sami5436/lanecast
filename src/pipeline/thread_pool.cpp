// Skeleton: runs serially on the calling thread. The pipeline unit replaces
// this with persistent workers (no thread creation per call).
#include "lanecast/pipeline.hpp"

namespace lanecast {

struct ThreadPool::Impl {};

ThreadPool::ThreadPool(int threads) : impl_(std::make_unique<Impl>()) { (void)threads; }
ThreadPool::~ThreadPool() = default;

int ThreadPool::size() const noexcept { return 1; }

void ThreadPool::parallel_for(int count, const std::function<void(int)>& fn) {
  for (int i = 0; i < count; ++i) fn(i);
}

Status run_sliced(ThreadPool* pool, int height, int alignment,
                  const std::function<Status(RowRange)>& fn) {
  (void)pool;
  (void)alignment;
  return fn(RowRange{0, height});
}

}  // namespace lanecast
