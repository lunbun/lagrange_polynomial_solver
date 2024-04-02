/**
 *
 * This is free and unencumbered software released into the public domain.
 *
 * Anyone is free to copy, modify, publish, use, compile, sell, or
 * distribute this software, either in source code form or as a compiled
 * binary, for any purpose, commercial or non-commercial, and by any
 * means.
 *
 * In jurisdictions that recognize copyright laws, the author or authors
 * of this software dedicate any and all copyright interest in the
 * software to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and
 * successors. We intend this dedication to be an overt act of
 * relinquishment in perpetuity of all present and future rights to this
 * software under copyright law.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * For more information, please refer to <https://unlicense.org>
 *
 *
 *
 * Lagrange polynomial interpolation.
 *
 * This header-only library provides functions to solve for the coefficients of the polynomial that passes through the
 * given points using the Lagrange polynomial interpolation method. The library also provides a function to solve for
 * the coefficients using multiple threads.
 *
 * Sample usage:
 * ```
 * double x[] = { 0, 1, 2, 3, 4, 5 };
 * double y[] = { 0, 1, 4, 9, 16, 25 };
 * double coefficients[6];
 * lagrange::solvePolynomial(x, y, 6, coefficients);                // Single threaded version
 * lagrange::solvePolynomialThreaded(x, y, 6, coefficients, 3);     // Multithreaded version, using 3 additional threads
 * ```
 *
 * Interpolation may be done with any type that supports arithmetic operations and in which basic algebraic identities
 * hold (e.g. floating point numbers, Galois field elements). Interpolation is done with optimized classical O(n^2)
 * algorithms for interpolation. As such, the library is only capable of interpolating a few hundred thousand points in
 * reasonable time. If you need to interpolate millions of points, it is possible to do so in O(n log n) time using
 * Fast Fourier Transform (FFT) based algorithms given certain constraints on the input points, but this library does
 * not provide such functionality.
 *
 * This library implements the standard definition of Lagrange interpolation as described in the Wikipedia article
 * (https://en.wikipedia.org/wiki/Lagrange_polynomial), but three optimizations are added:
 *
 * 1. This implementation uses the first barycentric form of the Lagrange polynomial, as described in the Wikipedia
 *      article, to bring the complexity of evaluating a single basis polynomial from O(n^2) to O(n), and thus the
 *      complexity of evaluating the entire polynomial from O(n^3) to O(n^2). Synthetic division is used to efficiently
 *      divide the precomputed coefficients of the common factor of each basis polynomial by (x - x[i]) to get the
 *      coefficients of the individual basis polynomial.
 * 2. If the x-values of all input points are equidistant from one another, the denominator of the basis polynomials can
 *      be incrementally updated by taking advantage of some algebraic identities, bringing the complexity of evaluating
 *      the denominator of each basis polynomial from O(n) to O(1).
 * 3. Computations of the common factor in the first barycentric form and the evaluation of individual basis polynomials
 *      may be parallelized using multiple threads. In fact, for machines with around a dozen or more
 *      hardware-concurrency threads, the bottleneck is in the non-parallelizable action of merging the results of all
 *      the threads, not in the computation of the basis polynomials themselves.
 *
 */

#pragma once

#include <limits>
#include <type_traits>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>

#ifdef LAGRANGE_INCLUDE_THREADING
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#endif // LAGRANGE_INCLUDE_THREADING

#ifndef LAGRANGE_INLINE
#define LAGRANGE_INLINE inline __attribute__((always_inline))
#endif // LAGRANGE_INLINE

namespace lagrange {

/**
 * Solves for the coefficients of the polynomial that passes through the given points.
 *
 * @tparam T The type of the x and y values. Must support arithmetic operations.
 * @param x The x-values of the points we are interpolating.
 * @param y The y-values of the points we are interpolating.
 * @param count The number of points we are interpolating.
 * @param coefficients The output coefficients. Must be of size count.
 */
template<typename T>
void solvePolynomial(const T *x, const T *y, size_t count, T *coefficients);

#ifdef LAGRANGE_INCLUDE_THREADING
/**
 * Solves for the coefficients of the polynomial that passes through the given points using multithreading.
 *
 * @tparam T The type of the x and y values. Must support arithmetic operations.
 * @param x The x-values of the points we are interpolating.
 * @param y The y-values of the points we are interpolating.
 * @param count The number of points we are interpolating.
 * @param coefficients The output coefficients. Must be of size count.
 * @param numAdditionalThreads The number of additional threads to use. numAdditionalThreads + 1 (for the main thread)
 *                             will be the total number of threads used.
 */
template<typename T>
void solvePolynomialThreaded(const T *x, const T *y, size_t count, T *coefficients, size_t numAdditionalThreads);
#endif // LAGRANGE_INCLUDE_THREADING



namespace internal {

template<typename T>
class DynamicArray {
public:
    LAGRANGE_INLINE DynamicArray() : size_(0), data_(nullptr) { }
    LAGRANGE_INLINE explicit DynamicArray(size_t size) : size_(size), data_(new T[size]()) { }

    LAGRANGE_INLINE DynamicArray(const DynamicArray &other) : size_(other.size_), data_(new T[other.size_]) {
        std::copy(other.data_, other.data_ + other.size_, this->data_);
    }
    LAGRANGE_INLINE DynamicArray &operator=(const DynamicArray &other) {
        DynamicArray tmp(other);
        swap(*this, tmp);
        return *this;
    }
    LAGRANGE_INLINE DynamicArray(DynamicArray &&other) noexcept : DynamicArray() { swap(*this, other); }
    LAGRANGE_INLINE DynamicArray &operator=(DynamicArray &&other) noexcept {
        swap(*this, other);
        return *this;
    }

    LAGRANGE_INLINE ~DynamicArray() noexcept { delete[] this->data_; }

    LAGRANGE_INLINE size_t size() const { return this->size_; }
    LAGRANGE_INLINE T *data() { return this->data_; }
    LAGRANGE_INLINE T &operator[](size_t index) { return this->data_[index]; }
    LAGRANGE_INLINE const T *data() const { return this->data_; }
    LAGRANGE_INLINE const T &operator[](size_t index) const { return this->data_[index]; }

    LAGRANGE_INLINE T *begin() { return this->data_; }
    LAGRANGE_INLINE T *end() { return this->data_ + this->size_; }
    LAGRANGE_INLINE const T *begin() const { return this->data_; }
    LAGRANGE_INLINE const T *end() const { return this->data_ + this->size_; }

private:
    size_t size_;
    T *data_;

    friend void swap(DynamicArray &a, DynamicArray &b) {
        std::swap(a.size_, b.size_);
        std::swap(a.data_, b.data_);
    }
};

// Precompute multiplication of all the terms (x - x[i]) for i = 0 to count - 1 so that we can simply divide by x -
// x[i] to get the coefficients of the basis polynomials, rather than having to multiply out every single term
// every time.
//
// Parameters:
//  - x: The x-values of the points we are interpolating.
//  - count: The number of points we are interpolating.
//  - coefficientsPrecomputed: The output precomputed coefficients.
template<typename T>
void precomputeCoefficients(const T *x, size_t start, size_t end, DynamicArray<T> &coefficientsPrecomputed) {
    std::fill(coefficientsPrecomputed.begin(), coefficientsPrecomputed.end(), 0);
    coefficientsPrecomputed[0] = 1;
    DynamicArray<T> tmp(coefficientsPrecomputed.size() + 1);
    for (size_t i = start; i != end + 1; i++) {
        // Shift the coefficients to the right by one to multiply by x.
        std::copy(coefficientsPrecomputed.begin(), coefficientsPrecomputed.end(), tmp.begin() + 1);
        tmp[0] = 0;

        // Add the term -x[i] to the coefficients.
        for (size_t j = 0; j < coefficientsPrecomputed.size(); j++) {
            tmp[j] -= x[i] * coefficientsPrecomputed[j];
        }

        std::copy(tmp.begin(), tmp.end() - 1, coefficientsPrecomputed.begin());
    }
}

// Divide the precomputed coefficients by x - x[i] to get the coefficients of the basis polynomial.
//
// Parameters:
//  - polynomial: The polynomial coefficients to divide.
//  - root: The root to divide by.
//  - quotient: The output quotient.
template<typename T>
LAGRANGE_INLINE void syntheticDivision(const DynamicArray<T> &polynomial, T root, DynamicArray<T> &quotient) {
    quotient[quotient.size() - 1] = polynomial[polynomial.size() - 1];
    for (size_t i = quotient.size() - 2; i != SIZE_MAX; i--) {
        quotient[i] = polynomial[i + 1] + root * quotient[i + 1];
    }
}

// Returns true if all the given values are equidistant from one another, false otherwise.
//
// Parameters:
//  - x: The values to check.
//  - count: The number of values.
template<typename T>
bool checkEquidistant(const T *x, size_t count) {
    T diff = x[1] - x[0];
    for (size_t i = 2; i < count; i++) {
        if (x[i] - x[i - 1] != diff) {
            return false;
        }
    }
    return true;
}

// Sums the basis polynomials to get the coefficients of the polynomial that passes through the given points.
//
// Parameters:
//  - x: The x-values of the points we are interpolating.
//  - y: The y-values of the points we are interpolating.
//  - start: The starting index of the points to sum the basis polynomials for.
//  - end: The ending index of the points to sum the basis polynomials for.
//  - count: The total number of points we are interpolating.
//  - coefficientsPrecomputed: The precomputed coefficients.
//  - coefficients: The output coefficients.
template<typename T>
void sumBasisPolynomials(const T *x, const T *y, size_t start, size_t end, size_t count,
                         const DynamicArray<T> &coefficientsPrecomputed, T *coefficients) {
    DynamicArray<T> coefficientsBasis(count);
    for (size_t i = start; i != end + 1; i++) {
        // Divide the precomputed coefficients by x - x[i] to get the coefficients of the basis polynomial.
        syntheticDivision(coefficientsPrecomputed, x[i], coefficientsBasis);

        T denominator = 1;
        for (size_t j = 0; j < count; j++) {
            if (i == j) {
                continue;
            }

            denominator *= x[i] - x[j];
        }

        T factor = y[i] / denominator;
        for (size_t j = 0; j < count; j++) {
            coefficients[j] += coefficientsBasis[j] * factor;
        }
    }
}

// Sums the basis polynomials to get the coefficients of the polynomial that passes through the given points. If all the
// x-values are equidistant, we can incrementally calculate the denominator as most terms will remain the same for each
// basis polynomial. This is a specialization of the sumBasisPolynomials function for the equidistant case.
//
// Example of equidistant x-values, and how the denominator calculation can be incrementally calculated:
//      x = { 0, 1, 2, 3, 4, 5 }
//      denominator_0 = 1/(0-1)(0-2)(0-3)(0-4)(0-5) = 1/(-1)(-2)(-3)(-4)(-5)
//      denominator_1 = 1/(1-0)(1-2)(1-3)(1-4)(1-5) = 1/(1) (-1)(-2)(-3)(-4) = denominator_0 / (1-0) * (0-5)
//      denominator_2 = 1/(2-0)(2-1)(2-3)(2-4)(2-5) = 1/(2) (1) (-1)(-2)(-3) = denominator_1 / (2-0) * (1-5)
//
// Parameters:
//  - x: The x-values of the points we are interpolating.
//  - y: The y-values of the points we are interpolating.
//  - start: The starting index of the points to sum the basis polynomials for.
//  - end: The ending index of the points to sum the basis polynomials for.
//  - count: The total number of points we are interpolating.
//  - coefficientsPrecomputed: The precomputed coefficients.
//  - coefficients: The output coefficients.
template<typename T>
void sumBasisPolynomialsEquidistant(const T *x, const T *y, size_t start, size_t end, size_t count,
                                    const DynamicArray<T> &coefficientsPrecomputed, T *coefficients) {
    // Compute the denominator for the first basis polynomial, then it can be incrementally updated for subsequent
    // basis polynomials.
    T denominator = 1;
    for (size_t i = 0; i < count; i++) {
        if (i != start) {
            denominator *= x[start] - x[i];
        }
    }
    denominator = T(1) / denominator;

    DynamicArray<T> coefficientsBasis(count);
    for (size_t i = start; i != end + 1; i++) {
        // Divide the precomputed coefficients by x - x[i] to get the coefficients of the basis polynomial.
        syntheticDivision(coefficientsPrecomputed, x[i], coefficientsBasis);

        T factor = y[i] * denominator;
        for (size_t j = 0; j < count; j++) {
            coefficients[j] += coefficientsBasis[j] * factor;
        }

        // Incrementally update the denominator for the next basis polynomial.
        if (i != end) {
            denominator *= (x[i] - x[count - 1]) / (x[i + 1] - x[0]);
        }
    }
}

} // namespace internal

/**
 * Solves for the coefficients of the polynomial that passes through the given points.
 *
 * @tparam T The type of the x and y values. Must support arithmetic operations.
 * @param x The x-values of the points we are interpolating.
 * @param y The y-values of the points we are interpolating.
 * @param count The number of points we are interpolating.
 * @param coefficients The output coefficients.
 */
template<typename T>
void solvePolynomial(const T *x, const T *y, size_t count, T *coefficients) {
    if (count < 2) {
        throw std::invalid_argument("count must be at least 2");
    }

    std::fill(coefficients, coefficients + count, 0);

    // Precompute multiplication of all the terms (x - x[i]) for i = 0 to count - 1 so that we can simply divide by x -
    // x[i] to get the coefficients of the basis polynomials, rather than having to multiply out every single term
    // every time.
    internal::DynamicArray<T> coefficientsPrecomputed(count + 1);
    internal::precomputeCoefficients(x, 0, count - 1, coefficientsPrecomputed);

    if (internal::checkEquidistant(x, count)) {
        internal::sumBasisPolynomialsEquidistant(x, y, 0, count - 1, count, coefficientsPrecomputed, coefficients);
    } else {
        internal::sumBasisPolynomials(x, y, 0, count - 1, count, coefficientsPrecomputed, coefficients);
    }
}



template<typename T, bool Pretty>
std::string polynomialToString(const T *coefficients, size_t count) {
    std::stringstream ss;
    if constexpr (std::is_floating_point_v<T>) {
        ss << std::fixed << std::setprecision(std::numeric_limits<T>::max_digits10);
    }
    for (size_t i = count - 1; i != SIZE_MAX; i--) {
        ss << coefficients[i];
        if (i != 0) {
            if constexpr (Pretty) {
                ss << "x^" << i << " + ";
            } else {
                ss << " ";
            }
        }
    }
    return ss.str();
}



#ifdef LAGRANGE_INCLUDE_THREADING
namespace internal {

// Thread synchronization primitive that allows one or more threads to wait until a set of operations being performed
// in other threads completes.
class Latch {
public:
    explicit Latch(size_t count) : count_(count) { }

    void countDown() {
        std::lock_guard lock(this->mutex_);
        if (this->count_ == 0) {
            throw std::runtime_error("countDown() called too many times");
        }

        this->count_--;
        this->condition_.notify_all();
    }

    void wait(size_t count) {
        std::unique_lock lock(this->mutex_);
        this->condition_.wait(lock, [this, count] { return this->count_ == count; });
    }

    LAGRANGE_INLINE void wait() {
        this->wait(0);
    }

    LAGRANGE_INLINE void countDownAndWait() {
        this->countDown();
        this->wait();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    size_t count_;
};

// Performs a convolution of the two given arrays (polynomial multiplication).
template<typename T>
void convolution(const T *a, const T *b, size_t countA, size_t countB, DynamicArray<T> &result) {
    if (result.size() < countA + countB - 1) {
        throw std::invalid_argument("result must be at least a.size() + b.size() - 1");
    }

    std::fill(result.begin(), result.end(), 0);
    for (size_t i = 0; i < countA; i++) {
        for (size_t j = 0; j < countB; j++) {
            result[i + j] += a[i] * b[j];
        }
    }
}

// Each thread does two things: it first aids in precomputing basis polynomial coefficients, and then it waits for all
// other threads to finish, then aids in summing up the basis polynomials.
template<typename T>
class LagrangeSolveThread {
public:
    LagrangeSolveThread(const T *x, const T *y, size_t start, size_t end, size_t count, bool isEquidistant,
                        const DynamicArray<T> &coefficientsPrecomputed, Latch &precomputeLatch, Latch &sumLatch)
            : precomputeLatch_(precomputeLatch), sumLatch_(sumLatch), x_(x), y_(y), start_(start), end_(end),
              count_(count), isEquidistant_(isEquidistant), threadCoefficientsPrecomputed_(end - start + 2),
              threadCoefficients_(count), coefficientsPrecomputed_(coefficientsPrecomputed) {
        this->thread_ = std::thread(&LagrangeSolveThread::run, this);
        this->thread_.detach();
    }

    LagrangeSolveThread(const LagrangeSolveThread &) = delete;
    LagrangeSolveThread &operator=(const LagrangeSolveThread &) = delete;
    LagrangeSolveThread(LagrangeSolveThread &&) = default;
    LagrangeSolveThread &operator=(LagrangeSolveThread &&) = default;

    LAGRANGE_INLINE const DynamicArray<T> &
    threadCoefficientsPrecomputed() const { return this->threadCoefficientsPrecomputed_; }
    LAGRANGE_INLINE const DynamicArray<T> &threadCoefficients() const { return this->threadCoefficients_; }

private:
    std::thread thread_;

    Latch &precomputeLatch_;
    Latch &sumLatch_;

    const T *x_;                                        // The x-values of the points we are interpolating.
    const T *y_;                                        // The y-values of the points we are interpolating.
    size_t start_;                                      // The start index of the range of points this thread is responsible for.
    size_t end_;                                        // The end index of the range of points this thread is responsible for.
    size_t count_;                                      // The total number of points we are interpolating.
    bool isEquidistant_;                                // Whether the x-values are equidistant.

    // The precomputed coefficients of this thread's range for the basis polynomials. Will be read by the main thread
    // once all child thread precomputations are complete, and merged together into the coefficientsPrecomputed_ array.
    DynamicArray<T> threadCoefficientsPrecomputed_;

    // The coefficients of the basis polynomials for this thread's range. Will be summed up by the main thread once all
    // child threads are done.
    DynamicArray<T> threadCoefficients_;

    // The precomputed coefficients of the basis polynomials for all threads (written to by the main thread after all
    // precomputations are done).
    const DynamicArray<T> &coefficientsPrecomputed_;

    void run();
};

template<typename T>
void LagrangeSolveThread<T>::run() {
    // Precompute the coefficients of the basis polynomials for this thread's range.
    precomputeCoefficients(this->x_, this->start_, this->end_, this->threadCoefficientsPrecomputed_);

    // Wait for all precomputation to finish.
    this->precomputeLatch_.countDownAndWait();

    // Sum up the basis polynomials for this thread's range.
    if (this->isEquidistant_) {
        sumBasisPolynomialsEquidistant(this->x_, this->y_, this->start_, this->end_, this->count_,
                                       this->coefficientsPrecomputed_, this->threadCoefficients_.data());
    } else {
        sumBasisPolynomials(this->x_, this->y_, this->start_, this->end_, this->count_,
                            this->threadCoefficientsPrecomputed_, this->threadCoefficients_.data());
    }

    // Indicate that this thread is done summing up the basis polynomials.
    this->sumLatch_.countDown();
}

} // namespace internal

/**
 * Solves for the coefficients of the polynomial that passes through the given points using multithreading.
 *
 * @tparam T The type of the x and y values. Must support arithmetic operations.
 * @param x The x-values of the points we are interpolating.
 * @param y The y-values of the points we are interpolating.
 * @param count The number of points we are interpolating.
 * @param coefficients The output coefficients. Must be of size count.
 * @param numAdditionalThreads The number of additional threads to use. numAdditionalThreads + 1 (for the main thread)
 *                             will be the total number of threads used.
 */
template<typename T>
void solvePolynomialThreaded(const T *x, const T *y, size_t count, T *coefficients, size_t numAdditionalThreads) {
    if (numAdditionalThreads == 0) {
        solvePolynomial(x, y, count, coefficients);
        return;
    }

    if (count < 2) {
        throw std::invalid_argument("count must be at least 2");
    }

    size_t numThreads = numAdditionalThreads + 1;

    // Main thread is responsible for the last range of points.
    size_t step = count / numThreads;
    size_t mainStart = (numThreads - 1) * step;
    size_t mainEnd = count - 1;

    bool isEquidistant = internal::checkEquidistant(x, count);

    internal::DynamicArray<T> coefficientsPrecomputed(count + 1);

    internal::Latch precomputeLatch(numThreads);
    internal::Latch sumLatch(numThreads);

    std::vector<std::unique_ptr<internal::LagrangeSolveThread<T>>> threads;
    for (size_t i = 0; i < numAdditionalThreads; i++) {
        size_t start = i * step;
        size_t end = (i + 1) * step - 1;
        threads.push_back(std::make_unique<internal::LagrangeSolveThread<T>>(x, y, start, end, count, isEquidistant,
                                                                             coefficientsPrecomputed, precomputeLatch,
                                                                             sumLatch));
    }

    // Precompute
    {
        // Precompute multiplication of all the terms (x - x[i]) for i = 0 to count - 1 so that we can simply divide by
        // x - x[i] to get the coefficients of the basis polynomials, rather than having to multiply out every single
        // term every time.
        internal::precomputeCoefficients(x, mainStart, mainEnd, coefficientsPrecomputed);

        // Most of the coefficientsPrecomputed array is 0, and we can exploit this fact later to convolve faster. We
        // only need to convolve the non-zero parts of the array.
        size_t coefficientsPrecomputedEffectiveSize = mainEnd - mainStart + 2;

        // Wait for all child threads to finish precomputing the coefficients. Once the latch count reaches 1, it means
        // the main thread is the only one left to finish precomputing (main thread will count down the latch once we
        // merge all the precomputed coefficients together).
        precomputeLatch.wait(1);

        // Convolve the coefficients of the basis polynomials for each thread's range together.
        internal::DynamicArray<T> tmp(count * 2);
        for (auto &thread : threads) {
            const internal::DynamicArray<T> &threadPrecomputed = thread->threadCoefficientsPrecomputed();
            convolution(coefficientsPrecomputed.data(), threadPrecomputed.data(), coefficientsPrecomputedEffectiveSize,
                        threadPrecomputed.size(), tmp);

            std::copy(tmp.begin(), tmp.begin() + coefficientsPrecomputed.size(), coefficientsPrecomputed.begin());
            coefficientsPrecomputedEffectiveSize += threadPrecomputed.size() - 1;
        }

        // Release the precompute latch, and allow all threads to begin summing up the basis polynomials.
        precomputeLatch.countDown();
    }

    // Sum basis polynomials
    {
        std::fill(coefficients, coefficients + count, 0);

        if (isEquidistant) {
            internal::sumBasisPolynomialsEquidistant(x, y, mainStart, mainEnd, count, coefficientsPrecomputed, coefficients);
        } else {
            internal::sumBasisPolynomials(x, y, mainStart, mainEnd, count, coefficientsPrecomputed, coefficients);
        }

        // Wait for all child threads to finish summing up the basis polynomials.
        sumLatch.wait(1);

        // Sum up the basis polynomials for each thread's range.
        for (auto &thread : threads) {
            for (size_t i = 0; i < count; i++) {
                coefficients[i] += thread->threadCoefficients()[i];
            }
        }

        // Release the sum latch, and allow all threads to finish.
        sumLatch.countDown();
    }
}
#endif // LAGRANGE_INCLUDE_THREADING

} // namespace lagrange

#undef LAGRANGE_INLINE
