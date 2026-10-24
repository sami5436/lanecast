// run_stream: reader thread -> process (calling thread) -> writer thread.
//
// Frames travel as slot indices through four bounded queues:
//   free_in -> reader -> full_in -> process -> full_out -> writer -> free_out
// and back. Every stage is single-threaded, so order is preserved by
// construction. close() lets a queue drain and then ends it; abort() ends it
// at once and drops what is queued.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

#include "lanecast/pipeline.hpp"

namespace lanecast {
namespace {

using Stage = StreamResult::Stage;

class SlotQueue {
 public:
  explicit SlotQueue(std::size_t capacity) : items_(capacity) {}

  // False once aborted. Never blocks: capacity covers every slot.
  bool push(int slot) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (aborted_) return false;
      items_[(head_ + size_++) % items_.size()] = slot;
    }
    cv_.notify_one();
    return true;
  }

  // Blocks for a slot. False when aborted, or closed and empty.
  bool pop(int& slot) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return aborted_ || closed_ || size_ > 0; });
    if (aborted_ || size_ == 0) return false;
    slot = items_[head_];
    head_ = (head_ + 1) % items_.size();
    --size_;
    return true;
  }

  void close() {
    { std::lock_guard<std::mutex> lock(mu_); closed_ = true; }
    cv_.notify_all();
  }
  void abort() {
    { std::lock_guard<std::mutex> lock(mu_); aborted_ = true; }
    cv_.notify_all();
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<int> items_;
  std::size_t head_ = 0;
  std::size_t size_ = 0;
  bool closed_ = false;
  bool aborted_ = false;
};

// Keeps the failure of the earliest frame, stage order breaking ties, so the
// outcome does not depend on how far the reader ran ahead.
class FirstError {
 public:
  void record(Stage stage, long long frame, Status status) {
    std::lock_guard<std::mutex> lock(mu_);
    if (error_.stage == Stage::none || frame < error_.failed_frame ||
        (frame == error_.failed_frame && stage < error_.stage)) {
      error_.stage = stage;
      error_.failed_frame = frame;
      error_.status = status;
    }
  }
  StreamResult take() {
    std::lock_guard<std::mutex> lock(mu_);
    return error_;
  }

 private:
  std::mutex mu_;
  StreamResult error_;
};

double seconds_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

StreamResult run_serial(FrameReader& reader, Pipeline& pipeline, FrameWriter& writer,
                        const StreamOptions& opts) {
  StreamResult r;
  Frame in;
  Frame out;
  const auto fail = [&r](Stage stage, Status s) {
    r.stage = stage;
    r.status = s;
    r.failed_frame = r.frames;
    return r;
  };
  while (opts.max_frames < 0 || r.frames < opts.max_frames) {
    Status s = reader.read(in);
    if (s == Status::end_of_stream) break;
    if (s != Status::ok) return fail(Stage::read, s);
    const auto t0 = std::chrono::steady_clock::now();
    s = pipeline.process(in, out);
    r.process_seconds += seconds_since(t0);
    if (s != Status::ok) return fail(Stage::process, s);
    if (s = writer.write(out); s != Status::ok) return fail(Stage::write, s);
    ++r.frames;
  }
  return r;
}

StreamResult run_overlapped(FrameReader& reader, Pipeline& pipeline, FrameWriter& writer,
                            const StreamOptions& opts) {
  const auto depth = static_cast<std::size_t>(std::max(opts.depth, 2));
  struct Slot {
    Frame frame;
    long long index = 0;
  };
  std::vector<Slot> in_slots(depth);
  std::vector<Slot> out_slots(depth);
  SlotQueue free_in(depth), full_in(depth), full_out(depth), free_out(depth);
  for (std::size_t i = 0; i < depth; ++i) {
    free_in.push(static_cast<int>(i));
    free_out.push(static_cast<int>(i));
  }
  FirstError errors;
  long long written = 0;  // writer thread only until joined
  double process_seconds = 0;

  const auto abort_all = [&] {
    free_in.abort();
    full_in.abort();
    full_out.abort();
    free_out.abort();
  };

  std::thread reader_thread([&] {
    for (long long index = 0; opts.max_frames < 0 || index < opts.max_frames; ++index) {
      int slot = 0;
      if (!free_in.pop(slot)) return;
      const Status s = reader.read(in_slots[static_cast<std::size_t>(slot)].frame);
      if (s == Status::end_of_stream) break;
      if (s != Status::ok) {
        errors.record(Stage::read, index, s);
        break;
      }
      in_slots[static_cast<std::size_t>(slot)].index = index;
      if (!full_in.push(slot)) return;
    }
    full_in.close();
  });

  std::thread writer_thread([&] {
    int slot = 0;
    while (full_out.pop(slot)) {
      Slot& s = out_slots[static_cast<std::size_t>(slot)];
      if (const Status st = writer.write(s.frame); st != Status::ok) {
        errors.record(Stage::write, s.index, st);
        abort_all();
        return;
      }
      ++written;
      free_out.push(slot);
    }
  });

  int in_slot = 0;
  while (full_in.pop(in_slot)) {
    int out_slot = 0;
    if (!free_out.pop(out_slot)) break;
    Slot& src = in_slots[static_cast<std::size_t>(in_slot)];
    Slot& dst = out_slots[static_cast<std::size_t>(out_slot)];
    const auto t0 = std::chrono::steady_clock::now();
    const Status s = pipeline.process(src.frame, dst.frame);
    process_seconds += seconds_since(t0);
    if (s != Status::ok) {
      errors.record(Stage::process, src.index, s);
      break;
    }
    dst.index = src.index;
    full_out.push(out_slot);
    free_in.push(in_slot);
  }
  // Normal end or failure: release the reader, let the writer drain what
  // was already processed.
  free_in.abort();
  full_in.abort();
  full_out.close();
  reader_thread.join();
  writer_thread.join();

  StreamResult r = errors.take();
  r.frames = written;
  r.process_seconds = process_seconds;
  return r;
}

}  // namespace

StreamResult run_stream(FrameReader& reader, Pipeline& pipeline, FrameWriter& writer,
                        const StreamOptions& opts) {
  return opts.overlap ? run_overlapped(reader, pipeline, writer, opts)
                      : run_serial(reader, pipeline, writer, opts);
}

}  // namespace lanecast
