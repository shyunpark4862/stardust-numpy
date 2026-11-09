// Sorting algorithms over a contiguous buffer, generic in the element type
// and the comparator (ARCHITECTURE.md §6.2).
//
//   stable=True   top-down merge sort, insertion sort below kSmall elements
//   stable=False  introsort: quicksort, falling back to heapsort when the
//                 recursion gets too deep (no O(n^2) worst case), insertion
//                 sort below kSmall elements
//
// The comparator must be a strict weak ordering. NaN is made "greater than
// everything and equal to itself" by the caller, so it always sorts last.
#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <utility>

namespace sd::sorting {

inline constexpr std::int64_t kSmall = 16;

template <typename T, typename Less>
void insertion_sort(T* a, std::int64_t n, Less less) {
    for (std::int64_t i = 1; i < n; ++i) {
        T v = a[i];
        std::int64_t j = i;
        for (; j > 0 && less(v, a[j - 1]); --j) a[j] = a[j - 1];  // strict: equal keys keep their order
        a[j] = v;
    }
}

// Sorts a[0, n) using tmp[0, n/2) as scratch.
template <typename T, typename Less>
void merge_sort(T* a, T* tmp, std::int64_t n, Less less) {
    if (n <= kSmall) {
        insertion_sort(a, n, less);
        return;
    }
    const std::int64_t m = n / 2;
    merge_sort(a, tmp, m, less);
    merge_sort(a + m, tmp, n - m, less);
    if (!less(a[m], a[m - 1])) return;  // the halves are already in order
    std::copy(a, a + m, tmp);
    // Merge tmp[0, m) and a[m, n) into a. Take from the right half only when
    // strictly smaller: equal keys keep their original order (stability).
    std::int64_t i = 0, j = m, k = 0;
    while (i < m && j < n) a[k++] = less(a[j], tmp[i]) ? a[j++] : tmp[i++];
    while (i < m) a[k++] = tmp[i++];
}

template <typename T, typename Less>
void intro_sort(T* a, std::int64_t n, Less less, int depth) {
    while (n > kSmall) {
        if (depth-- == 0) {
            std::make_heap(a, a + n, less);
            std::sort_heap(a, a + n, less);
            return;
        }
        // Median of three as the pivot, then Hoare partition.
        T* mid = a + n / 2;
        if (less(*mid, a[0])) std::swap(*mid, a[0]);
        if (less(a[n - 1], *mid)) {
            std::swap(a[n - 1], *mid);
            if (less(*mid, a[0])) std::swap(*mid, a[0]);
        }
        const T pivot = *mid;
        std::int64_t i = -1, j = n;
        for (;;) {
            do ++i; while (less(a[i], pivot));
            do --j; while (less(pivot, a[j]));
            if (i >= j) break;
            std::swap(a[i], a[j]);
        }
        // [0, j] and [j+1, n): recurse into the smaller part, loop on the
        // larger, so the stack stays O(log n).
        const std::int64_t left = j + 1;
        if (left < n - left) {
            intro_sort(a, left, less, depth);
            a += left;
            n -= left;
        } else {
            intro_sort(a + left, n - left, less, depth);
            n = left;
        }
    }
    insertion_sort(a, n, less);
}

template <typename T, typename Less>
void unstable_sort(T* a, std::int64_t n, Less less) {
    const int depth = n > 1 ? 2 * (63 - std::countl_zero(static_cast<std::uint64_t>(n))) : 0;  // 2 log2 n
    intro_sort(a, n, less, depth);
}

}  // namespace sd::sorting
