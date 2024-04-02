# Lagrange Polynomial Solver

This is a small CLI tool and header-only library that can be used to efficiently perform Lagrange interpolation and
compute the coefficients of the resulting polynomial. Out-of-the-box, this CLI tool can be used to interpolate
with double-precision floating-point or Galois field arithmetic, but the library can be easily extended to support
other arithmetic types. Multithreaded computation is supported.

*Note: if you are interpolating over a Galois field, do not use this, use
[FLINT](https://flintlib.org/doc/nmod_poly.html#c.nmod_poly_interpolate_nmod_vec_fast) as it is much, much faster.*

## Benchmarks

The following benchmarks were performed on an Intel i7-10750H CPU with 6 cores and 12 threads. Interpolation was done
over the same Galois field with the same set of points.

| Points | Solver (1 thread) | Solver (12 threads) | Python's [galois library](https://galois.readthedocs.io/en/v0.3.8/api/galois.lagrange_poly/) |
|--------|-------------------|---------------------|----------------------------------------------------------------------------------------------|
| 100    | 0.00s             | 0.00s               | 0.24s                                                                                        |
| 500    | 0.00s             | 0.00s               | 16.21s                                                                                       |
| 1000   | 0.07s             | 0.02s               | 1m 57.86s                                                                                    |
| 2000   | 0.28s             | 0.08s               | 13m 32.21s                                                                                   |
| 5000   | 1.82s             | 0.47s               | Did not complete within 15 minutes                                                           |
| 10000  | 6.75s             | 1.88s               | Did not complete within 15 minutes                                                           |
| 25000  | 41.65s            | 11.81s              | Did not complete within 15 minutes                                                           |
| 100000 | 11m 11.68s        | 2m 16.64s           | Did not complete within 15 minutes                                                           |

## CLI Tool

### Installation

Installation can be done simply by cloning the repository and using
[standard CMake commands](https://stackoverflow.com/a/7859663). You should build in release mode for optimal
performance. Note that the CLI tool may not play nice with MSVC, it has only been tested on GCC and Clang.

### Usage

```
Usage: lagrange_polynomial_solver [--help] [--version] --input-file VAR [--output-file VAR] [--type VAR] [--mod VAR] [--format VAR] [--threads VAR] [--silent]

Optional arguments:
  -h, --help                   shows help message and exits
  -v, --version                prints version information and exits
  -i, --input, --input-file    The file containing the points to solve the polynomial for. [required]
  -o, --output, --output-file  The file to write the polynomial to. If omitted, the polynomial will be written to stdout.
  -t, --type                   The type of the polynomial. [nargs=0..1] [default: "double"]
  -m, --mod                    The modulus to use for the Galois field.
  -f, --format                 The format to write the polynomial in. [nargs=0..1] [default: "pretty"]
  -n, --threads                The number of threads to use. [nargs=0..1] [default: 1]
  -s, --silent                 Suppress all output.
```

The input file must contain one point per line, in the format `x y`. Example:
```
0 0
1 1
2 4
3 9
4 16
```

*Note: I have found that after around 20 or so points, the output polynomial coefficients become bogus due to
floating-point precision issues, even with double-precision floating-point. This is not an issue with the Galois field
mode, as it does not suffer from imprecision.*

## Header-Only Library

The `src/lagrange.h` file contains the entire library. It is written in C++17 and requires a C++17-compliant compiler.

The `src/galois.h` file also contains a small basic Galois field implementation that can be used to perform arithmetic
over a Galois field. It is used by the CLI tool to perform arithmetic over a Galois field.

### Usage

To include multithreaded computation, `#define LAGRANGE_INCLUDE_THREADING` before including the header file.

Both the single-threaded and multithreaded versions of the solver support any type that supports arithmetic
operations and holds basic algebraic identities.

Sample usage:
```c++
// Single-threaded
constexpr int PointCount = 5;
double x[] = { 0, 1, 2, 3, 4 };
double y[] = { 0, 1, 4, 9, 16 };
double coefficients[PointCount];

// Both solve methods will write to the coefficients array
lagrange::solvePolynomial<double>(x, y, PointCount, coefficients);

// Multithreaded
constexpr int ThreadCount = 4;
lagrange::solvePolynomialThreaded<double>(x, y, PointCount, coefficients, ThreadCount);
```
