// The thread pool (ARCHITECTURE.md §5; M8).
//
// Resident workers sleep on a condition variable. A parallel job is a number
// of tasks; every participant -- the workers and the calling thread itself --
// takes the next task index from one atomic counter until none are left, so
// fast threads simply take more tasks (no fixed assignment of work to
// threads).
//
// What a task computes never depends on how many threads there are: the
// callers cut work into tasks by size alone, and combine any partial results
// in task order. So every result is bit-for-bit the same for any thread count
// (DESIGN.md §2.4).
#pragma once

#include <cstdint>
#include <functional>

namespace sd::parallel {

// Below this much work (elements, roughly) everything runs on the calling
// thread. M8 measured where 10 threads start to win (bench/m8_threshold.py):
// f + g at ~16K elements, exp at ~8K, sum at ~32K; one constant for all
// (DESIGN.md §2.7), so the most demanding one decides.
inline constexpr std::int64_t kThreshold = 1 << 15;

// How much work one task gets. With the pool's fixed cost per job at ~1.5 us
// (after M8's spin and lock-free pickup), small tasks balance load well.
inline constexpr std::int64_t kTaskWork = 1 << 12;

int num_threads();
void set_num_threads(int n);  // n >= 1; 1 means: never use the pool

// True when a parallel_for would actually run in parallel: more than one
// thread, and not already inside a task (no nesting).
bool available();

// Runs fn(t) for t in [0, ntasks). In parallel when available() and the
// pool is not busy with another Python thread's job (then this call runs on
// the calling thread alone). The GIL is released for the duration, so other
// Python threads keep running. An exception from a task is rethrown here.
void parallel_for(std::int64_t ntasks, const std::function<void(std::int64_t)>& fn);

// Releases the GIL (if this thread holds it) for the lifetime of the object:
// for long single-threaded work too.
class GilRelease {
public:
    GilRelease();
    ~GilRelease();
    GilRelease(const GilRelease&) = delete;
    GilRelease& operator=(const GilRelease&) = delete;

private:
    void* state_ = nullptr;
};

}  // namespace sd::parallel
