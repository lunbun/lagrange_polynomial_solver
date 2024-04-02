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
 * Basic Galois field arithmetic implementation.
 *
 */

#pragma once

#include <limits>
#include <type_traits>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

#ifndef LAGRANGE_INLINE
#define LAGRANGE_INLINE inline __attribute__((always_inline))
#endif // LAGRANGE_INLINE

namespace lagrange {

template<typename T, T Mod>
class Galois {
public:
    static_assert(std::is_integral_v<T>, "T must be an integral type");

    LAGRANGE_INLINE Galois() : value_(0) { }
    LAGRANGE_INLINE /* implicit */ Galois(T value) : value_(value) { }           // NOLINT(google-explicit-constructor)
    LAGRANGE_INLINE /* implicit */ operator T() const { return this->value_; }   // NOLINT(google-explicit-constructor)

    LAGRANGE_INLINE Galois operator-() const { return Mod - this->value_; }
    LAGRANGE_INLINE Galois operator+(Galois other) const {
        return (this->value_ >= Mod - other.value_) ? this->value_ - (Mod - other.value_) : this->value_ + other.value_;
    }
    LAGRANGE_INLINE Galois operator-(Galois other) const {
        return (this->value_ < other.value_) ? this->value_ + (Mod - other.value_) : this->value_ - other.value_;
    }
    LAGRANGE_INLINE Galois operator*(Galois other) const {
        return static_cast<T>((static_cast<__uint128_t>(this->value_) * other.value_) % Mod);
    }
    LAGRANGE_INLINE Galois operator/(Galois other) const { return this->operator*(other.invert()); }
    LAGRANGE_INLINE Galois operator+=(Galois other) { return *this = *this + other; }
    LAGRANGE_INLINE Galois operator-=(Galois other) { return *this = *this - other; }
    LAGRANGE_INLINE Galois operator*=(Galois other) { return *this = *this * other; }
    LAGRANGE_INLINE Galois operator/=(Galois other) { return *this = *this / other; }

    LAGRANGE_INLINE Galois invert() const {
        __int128_t t = 0, newT = 1;
        T r = Mod, newR = this->value_;
        while (newR != 0) {
            T quotient = r / newR;
            std::swap(t, newT);
            newT = newT - quotient * t;
            std::swap(r, newR);
            newR = newR - quotient * r;
        }
        if (r > 1) {
            throw std::invalid_argument("a is not invertible");
        }
        return (t < 0) ? t + Mod : t;
    }

private:
    T value_;
};

} // namespace lagrange

#undef LAGRANGE_INLINE
