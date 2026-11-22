// SPDX-License-Identifier: MIT
//
// tests/tools/stage_bench_s9_riekf.cpp - S9 R-IEKF marginalization stage bench
// Part of issue #612: S3, S5, S6 (done in #622), S9.
//
// Oracle: principal submatrix - the covariance after dropping clone idx must
// equal the rows/cols of P_pre that correspond to the surviving dimensions.

#include <branes/tools/bench/s9_riekf_marginalization_bench.hpp>

#include <catch2/catch_test_macros.hpp>

namespace {
namespace bn = branes::tools::bench;
using B = bn::S9RiEkfMargBench<double>;
using Bf = bn::S9RiEkfMargBench<float>;
}  // namespace

TEST_CASE("S9 R-IEKF: principal submatrix - 3 clones, drop oldest", "[tools][bench][s9][riekf]") {
    const auto res = B::run(3, 0, 42);
    INFO("psub_err = " << res.psub_err);
    INFO("sym_err  = " << res.sym_err);
    REQUIRE(res.dim_ok);
    REQUIRE(res.psub_ok);
    REQUIRE(res.sym_ok);
}

TEST_CASE("S9 R-IEKF: principal submatrix - 5 clones, drop oldest", "[tools][bench][s9][riekf]") {
    const auto res = B::run(5, 0, 7);
    INFO("psub_err = " << res.psub_err);
    INFO("sym_err  = " << res.sym_err);
    REQUIRE(res.dim_ok);
    REQUIRE(res.psub_ok);
    REQUIRE(res.sym_ok);
}

TEST_CASE("S9 R-IEKF: principal submatrix - 2 clones, drop middle", "[tools][bench][s9][riekf]") {
    const auto res = B::run(2, 1, 99);
    INFO("psub_err = " << res.psub_err);
    INFO("sym_err  = " << res.sym_err);
    REQUIRE(res.dim_ok);
    REQUIRE(res.psub_ok);
    REQUIRE(res.sym_ok);
}

TEST_CASE("S9 R-IEKF: clone blocks are distinguishable after propagation", "[tools][bench][s9][riekf]") {
    // Explicitly verify that the competing submatrices P_pre[{1,2,3},{1,2,3}]
    // and P_pre[{0,1,3},{0,1,3}] differ — proving that run_propagated() can
    // detect an impl that always removes clone 0 regardless of drop_idx.
    const double diff = B::clone_block_diff(4, 0, 2, 13);
    INFO("clone_block_diff(0,2) = " << diff);
    REQUIRE(diff > 1e-6);
}

TEST_CASE("S9 R-IEKF: principal submatrix - 4 clones, drop middle (propagated)", "[tools][bench][s9][riekf]") {
    const auto res = B::run_propagated(4, 2, 13);
    INFO("psub_err = " << res.psub_err);
    INFO("sym_err  = " << res.sym_err);
    REQUIRE(res.dim_ok);
    REQUIRE(res.psub_ok);
    REQUIRE(res.sym_ok);
}

TEST_CASE("S9 R-IEKF: principal submatrix - float, 3 clones, drop oldest", "[tools][bench][s9][riekf][float]") {
    const auto res = Bf::run(3, 0, 42);
    INFO("psub_err = " << res.psub_err);
    INFO("sym_err  = " << res.sym_err);
    REQUIRE(res.dim_ok);
    REQUIRE(res.psub_ok);
    REQUIRE(res.sym_ok);
}

TEST_CASE("S9 R-IEKF: principal submatrix - float, 5 clones, drop oldest", "[tools][bench][s9][riekf][float]") {
    const auto res = Bf::run(5, 0, 7);
    INFO("psub_err = " << res.psub_err);
    INFO("sym_err  = " << res.sym_err);
    REQUIRE(res.dim_ok);
    REQUIRE(res.psub_ok);
    REQUIRE(res.sym_ok);
}
