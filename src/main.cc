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
 * CLI for Lagrange polynomial solver.
 *
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <optional>

#include "argparse.h"

#define LAGRANGE_INCLUDE_THREADING
#include "lagrange.h"
#include "galois.h"

#define INLINE inline __attribute__((always_inline))

namespace {

// Change this and recompile to use the more optimized Galois implementation.
constexpr uint64_t DefaultModulus = 7514777789ULL;

namespace config {

std::string inputPath;
std::optional<std::string> outputPath;
std::string type;
std::optional<uint64_t> mod;
std::string format;
uint32_t threads;
bool silent;

} // namespace config

// Kinda a hack for getting dynamic modulus working, since the Galois class requires the modulus be specified at compile
// time. Relies on the fact that there is only one instance of the polynomial solver running, so the modulus can be
// changed globally.
class DynamicGalois {
public:
    static inline uint64_t Modulus = DefaultModulus;

    INLINE DynamicGalois() : value_(0) { }
    INLINE /* implicit */ DynamicGalois(uint64_t value) : value_(value) { }     // NOLINT(google-explicit-constructor)
    INLINE /* implicit */ operator uint64_t() const { return this->value_; }    // NOLINT(google-explicit-constructor)

    INLINE DynamicGalois operator-() const { return Modulus - this->value_; }
    INLINE DynamicGalois operator+(DynamicGalois other) const {
        return (this->value_ >= Modulus - other.value_) ? this->value_ - (Modulus - other.value_) : this->value_ + other.value_;
    }
    INLINE DynamicGalois operator-(DynamicGalois other) const {
        return (this->value_ < other.value_) ? this->value_ + (Modulus - other.value_) : this->value_ - other.value_;
    }
    INLINE DynamicGalois operator*(DynamicGalois other) const {
        return static_cast<uint64_t>((static_cast<__uint128_t>(this->value_) * other.value_) % Modulus);
    }
    INLINE DynamicGalois operator/(DynamicGalois other) const { return this->operator*(other.invert()); }
    INLINE DynamicGalois operator+=(DynamicGalois other) { return *this = *this + other; }
    INLINE DynamicGalois operator-=(DynamicGalois other) { return *this = *this - other; }
    INLINE DynamicGalois operator*=(DynamicGalois other) { return *this = *this * other; }
    INLINE DynamicGalois operator/=(DynamicGalois other) { return *this = *this / other; }

    INLINE DynamicGalois invert() const {
        __int128_t t = 0, newT = 1;
        uint64_t r = Modulus, newR = this->value_;
        while (newR != 0) {
            uint64_t quotient = r / newR;
            std::swap(t, newT);
            newT = newT - quotient * t;
            std::swap(r, newR);
            newR = newR - quotient * r;
        }
        if (r > 1) {
            throw std::runtime_error("Modulus is not prime");
        }
        return newT < 0 ? newT + Modulus : newT;
    }

private:
    uint64_t value_;
};

bool isPrime(uint64_t n) {
    if (n <= 1) {
        return false;
    }
    if (n <= 3) {
        return true;
    }
    if (n % 2 == 0 || n % 3 == 0) {
        return false;
    }
    for (uint64_t i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) {
            return false;
        }
    }
    return true;
}

void readPointsDouble(std::ifstream &file, std::vector<double> &x, std::vector<double> &y) {
    double pointX, pointY;
    while (file >> pointX >> pointY) {
        x.push_back(pointX);
        y.push_back(pointY);
    }
}

void readPointsDefaultGalois(std::ifstream &file, std::vector<lagrange::Galois<uint64_t, DefaultModulus>> &x, std::vector<lagrange::Galois<uint64_t, DefaultModulus>> &y) {
    uint64_t pointX, pointY;
    while (file >> pointX >> pointY) {
        x.emplace_back(pointX % DefaultModulus);
        y.emplace_back(pointY % DefaultModulus);
    }
}

void readPointsDynamicGalois(std::ifstream &file, std::vector<DynamicGalois> &x, std::vector<DynamicGalois> &y) {
    uint64_t pointX, pointY;
    while (file >> pointX >> pointY) {
        x.emplace_back(pointX % DynamicGalois::Modulus);
        y.emplace_back(pointY % DynamicGalois::Modulus);
    }
}

void log(const std::string &message) {
    if (!config::silent) {
        std::cout << message << std::endl;
    }
}

template<typename T>
std::string stringify(const lagrange::internal::DynamicArray<T> &polynomial) {
    if (config::format == "raw") {
        return lagrange::polynomialToString<T, false>(polynomial.data(), polynomial.size());
    } else if (config::format == "pretty") {
        return lagrange::polynomialToString<T, true>(polynomial.data(), polynomial.size());
    } else {
        return "";
    }
}

void writeOutput(const std::string &polynomial) {
    if (config::outputPath.has_value()) {
        std::ofstream output(*config::outputPath);
        if (!output.is_open()) {
            std::cerr << "Failed to open output file" << std::endl;
            return;
        }
        output << polynomial << std::endl;
    } else {
        std::cout << polynomial << std::endl;
    }
}

template<typename T, void (*ReadPoints)(std::ifstream &, std::vector<T> &, std::vector<T> &)>
void run() {
    std::ifstream inputFile(config::inputPath);
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open input file" << std::endl;
        return;
    }

    std::vector<T> x, y;
    ReadPoints(inputFile, x, y);

    lagrange::internal::DynamicArray<T> coefficients(x.size());

    log("Solving polynomial...");
    auto start = std::chrono::high_resolution_clock::now();

    if (config::threads == 0) {
        std::cerr << "Invalid number of threads" << std::endl;
        return;
    }
    lagrange::solvePolynomialThreaded(x.data(), y.data(), x.size(), coefficients.data(), config::threads - 1);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    log("Solved polynomial in " + std::to_string(elapsed.count()) + " seconds");

    std::string output = stringify(coefficients);
    writeOutput(output);
}

} // namespace

int main(int argc, char **argv) {
    argparse::ArgumentParser program("lagrange_polynomial_solver");

    program.add_argument("-i", "--input", "--input-file")
        .help("The file containing the points to solve the polynomial for.")
        .required();
    program.add_argument("-o", "--output", "--output-file")
        .help("The file to write the polynomial to. If omitted, the polynomial will be written to stdout.");
    program.add_argument("-t", "--type")
        .help("The type of the polynomial.")
        .choices("double", "galois")
        .default_value(std::string("double"));
    program.add_argument("-m", "--mod")
        .help("The modulus to use for the Galois field.")
        .scan<'u', uint64_t>();
    program.add_argument("-f", "--format")
        .help("The format to write the polynomial in.")
        .choices("raw", "pretty")
        .default_value(std::string("pretty"));
    program.add_argument("-n", "--threads")
        .help("The number of threads to use.")
        .scan<'u', uint32_t>()
        .default_value(uint32_t(1));
    program.add_argument("-s", "--silent")
        .help("Suppress all unnecessary output to stdout.")
        .flag();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    config::inputPath = program.get<std::string>("--input");
    config::outputPath = program.present<std::string>("--output");
    config::type = program.get<std::string>("--type");
    config::mod = program.present<uint64_t>("--mod");
    config::format = program.get<std::string>("--format");
    config::threads = program.get<uint32_t>("--threads");
    config::silent = program.get<bool>("--silent");

    if (config::type == "double") {
        run<double, readPointsDouble>();
    } else if (config::type == "galois") {
        if (!config::mod.has_value()) {
            std::cerr << "Modulus is required for Galois field" << std::endl;
            return 1;
        }

        if (!isPrime(config::mod.value())) {
            std::cerr << "Modulus must be prime" << std::endl;
            return 1;
        }

        if (config::mod.value() == DefaultModulus) {
            run<lagrange::Galois<uint64_t, DefaultModulus>, readPointsDefaultGalois>();
        } else {
            log("Warning: using less optimized Galois implementation. To use the more optimized implementation, change "
                "the DefaultModulus constant in the source code and recompile (currently "
                + std::to_string(DefaultModulus) + ")");
            DynamicGalois::Modulus = config::mod.value();
            run<DynamicGalois, readPointsDynamicGalois>();
        }
    } else {
        std::cerr << "Invalid type" << std::endl;
        return 1;
    }

    return 0;
}
