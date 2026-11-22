// SPDX-License-Identifier: MIT
//
// branes/tools/bench/s9_riekf_marginalization_bench.hpp — S9_marginalization
// stage bench for the Right-Invariant EKF backend (issue #612).
//
// Oracle: principal submatrix.  After marginalize_clone(idx) the surviving
// covariance must equal the rows/cols of P_pre that correspond to the kept
// dimensions — identical oracle to the body-frame S9 bench.
//
// Stage under test: MsckfInvariantBackend<T>::marginalize_clone(idx).
// Construction: seed_covariance(I) + N × augment_clone() builds a valid
// backend with N clones without needing any private test hooks.

#pragma once

#include <branes/math/arithmetic.hpp>
#include <branes/sdk/msckf/covariance.hpp>
#include <branes/sdk/msckf/dense.hpp>
#include <branes/sdk/msckf/msckf_invariant_backend.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace branes::tools::bench {

using sdk::msckf::DynMat;

// ── oracle ────────────────────────────────────────────────────────────────────

/// Return the principal submatrix of P after removing the 6-dim block at [off, off+6).
template <math::Scalar T>
DynMat<T> principal_submatrix(const DynMat<T>& P, std::size_t off) {
    constexpr std::size_t kCD = 6;
    assert(P.rows == P.cols);
    const std::size_t d = P.rows;
    const std::size_t d2 = d - kCD;

    std::vector<std::size_t> keep;
    keep.reserve(d2);
    for (std::size_t i = 0; i < d; ++i)
        if (i < off || i >= off + kCD)
            keep.push_back(i);

    DynMat<T> Q(d2, d2);
    for (std::size_t r = 0; r < d2; ++r)
        for (std::size_t c = 0; c < d2; ++c)
            Q(r, c) = P(keep[r], keep[c]);
    return Q;
}

/// Max |a(i,j) - b(i,j)| over all entries.
template <math::Scalar T>
double max_abs_diff(const DynMat<T>& a, const DynMat<T>& b) {
    assert(a.rows == b.rows && a.cols == b.cols);
    double mx = 0.0;
    for (std::size_t i = 0; i < a.rows * a.cols; ++i)
        mx = std::max(mx, std::abs(static_cast<double>(a.d[i] - b.d[i])));
    return mx;
}

/// Max |P(i,j) - P(j,i)|.
template <math::Scalar T>
double symmetry_error(const DynMat<T>& P) {
    assert(P.rows == P.cols);
    double mx = 0.0;
    for (std::size_t r = 0; r < P.rows; ++r)
        for (std::size_t c = r + 1; c < P.cols; ++c)
            mx = std::max(mx, std::abs(static_cast<double>(P(r, c) - P(c, r))));
    return mx;
}

// ── bench ─────────────────────────────────────────────────────────────────────

template <math::Scalar T>
struct S9RiEkfMargBench {
    static constexpr std::size_t kNavDim = 15;
    static constexpr std::size_t kCloneDim = 6;

    using Backend = sdk::msckf::MsckfInvariantBackend<T>;

    struct Result {
        bool dim_ok = false;
        bool psub_ok = false;
        bool sym_ok = false;
        double psub_err = 0.0;
        double sym_err = 0.0;
        bool all_ok() const {
            return dim_ok && psub_ok && sym_ok;
        }
    };

    /// Build a backend with `n_clones` clones, snapshot P, drop `drop_idx`,
    /// check the three invariants.
    static Result run(std::size_t n_clones, std::size_t drop_idx, [[maybe_unused]] unsigned seed = 42) {
        // tolerance: double ~1e-10, float ~1e-4
        const double tol = (sizeof(T) >= 8) ? 1e-10 : 1e-4;

        // ── construct backend ────────────────────────────────────────────────
        typename Backend::Config cfg;
        cfg.max_clones = n_clones + 1;  // prevent auto-marginalization
        Backend be(cfg);

        // seed nav covariance = I₁₅ (seed_covariance takes G s.t. P = G·Gᵀ)
        DynMat<T> G = DynMat<T>::identity(kNavDim);
        be.seed_covariance(G);

        const typename Backend::Vec3 zero{};
        // set nav to identity SE₂(3) with zero biases
        be.set_nav(typename Backend::SE23{}, zero, zero, 0.0);

        // augment clones — vary nav rotation between augmentations so each
        // clone's pose (and therefore its covariance block) is distinguishable.
        // This lets the principal-submatrix oracle detect incorrect clone-index
        // handling (e.g. always dropping clone 0 regardless of drop_idx).
        for (std::size_t i = 0; i < n_clones; ++i) {
            // propagate over a small IMU interval so each clone's covariance
            // block accumulates distinct noise, making the principal-submatrix
            // oracle sensitive to incorrect clone-index handling.
            be.propagate(zero, zero, T(0.1));
            be.augment_clone();
        }

        // ── snapshot pre-marginalization covariance ──────────────────────────
        const DynMat<T> P_pre = be.covariance();
        const std::size_t d_pre = P_pre.rows;
        const std::size_t d_post_expected = d_pre - kCloneDim;

        // ── run the stage under test ─────────────────────────────────────────
        be.marginalize_clone(drop_idx);
        const DynMat<T> P_post = be.covariance();

        // ── invariant 1: dimension ───────────────────────────────────────────
        Result res;
        res.dim_ok =
            (P_post.rows == d_post_expected) && (P_post.cols == d_post_expected) && (be.num_clones() == n_clones - 1);

        // ── invariant 2: principal submatrix ─────────────────────────────────
        const std::size_t off = kNavDim + kCloneDim * drop_idx;
        const DynMat<T> Q = principal_submatrix(P_pre, off);
        res.psub_err = max_abs_diff(P_post, Q);
        res.psub_ok = (res.psub_err < tol);

        // ── invariant 3: symmetry ────────────────────────────────────────────
        res.sym_err = symmetry_error(P_post);
        res.sym_ok = (res.sym_err < tol);

        return res;
    }

    /// Return the max absolute difference between the diagonal 6×6 clone blocks
    /// for clone i and clone j in a propagated backend.  Used by tests to prove
    /// that run_propagated() produces genuinely distinct blocks.
    static double clone_block_diff(std::size_t n_clones, std::size_t i, std::size_t j, unsigned seed = 42) {
        typename Backend::Config cfg;
        cfg.max_clones = n_clones + 1;
        Backend be(cfg);

        DynMat<T> G = DynMat<T>::identity(kNavDim);
        be.seed_covariance(G);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> ud(-0.3, 0.3);
        const typename Backend::Vec3 zero{};
        const auto small_gyro = [&]() -> typename Backend::Vec3 {
            return {{static_cast<T>(ud(rng)), static_cast<T>(ud(rng)), static_cast<T>(ud(rng))}};
        };

        be.set_nav(typename Backend::SE23{}, zero, zero, 0.0);
        for (std::size_t k = 0; k < n_clones; ++k) {
            const typename Backend::Vec3 phi = small_gyro();
            const typename Backend::SE23 X_k(Backend::SO3::exp(phi), zero, zero);
            be.set_nav(X_k, zero, zero, static_cast<double>(k) * 0.1);
            be.augment_clone();
            if (k + 1 < n_clones) {
                const typename Backend::Vec3 gyro = small_gyro();
                const typename Backend::Vec3 accel{{T(0), T(0), T(9.81)}};
                be.propagate(gyro, accel, T(0.1));
            }
        }

        const DynMat<T> P = be.covariance();
        const std::size_t bi = kNavDim + kCloneDim * i;
        const std::size_t bj = kNavDim + kCloneDim * j;
        double mx = 0.0;
        for (std::size_t r = 0; r < kCloneDim; ++r)
            for (std::size_t c = 0; c < kCloneDim; ++c)
                mx = std::max(mx, std::abs(static_cast<double>(P(bi + r, bi + c) - P(bj + r, bj + c))));
        return mx;
    }

    /// Like run(), but propagates a held IMU interval between each augmentation
    /// so that every clone's covariance block is genuinely distinct after S2.
    /// A buggy marginalize_clone() that always removes clone 0 regardless of
    /// drop_idx will fail psub_ok whenever drop_idx != 0, because the blocks
    /// for clone 0 and clone k (k > 0) differ once propagation has run.
    static Result run_propagated(std::size_t n_clones, std::size_t drop_idx, unsigned seed = 42) {
        const double tol = (sizeof(T) >= 8) ? 1e-10 : 1e-4;

        typename Backend::Config cfg;
        cfg.max_clones = n_clones + 1;
        Backend be(cfg);

        // seed nav covariance = I₁₅
        DynMat<T> G = DynMat<T>::identity(kNavDim);
        be.seed_covariance(G);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> ud(-0.3, 0.3);
        const typename Backend::Vec3 zero{};
        const auto small_gyro = [&]() -> typename Backend::Vec3 {
            return {{static_cast<T>(ud(rng)), static_cast<T>(ud(rng)), static_cast<T>(ud(rng))}};
        };

        be.set_nav(typename Backend::SE23{}, zero, zero, 0.0);

        // Augment N clones, propagating a 0.1 s IMU interval between each so
        // that the nav covariance block (and therefore the cross-terms copied
        // into each clone block) differs for every clone.
        for (std::size_t i = 0; i < n_clones; ++i) {
            // rotate nav pose so clone pose is non-trivial
            const typename Backend::Vec3 phi = small_gyro();
            const typename Backend::SE23 X_i(Backend::SO3::exp(phi), zero, zero);
            be.set_nav(X_i, zero, zero, static_cast<double>(i) * 0.1);
            be.augment_clone();
            // propagate so the next clone inherits a different covariance block
            if (i + 1 < n_clones) {
                const typename Backend::Vec3 gyro = small_gyro();
                const typename Backend::Vec3 accel{{T(0), T(0), T(9.81)}};
                be.propagate(gyro, accel, T(0.1));
            }
        }

        // snapshot pre-marginalization covariance
        const DynMat<T> P_pre = be.covariance();
        const std::size_t d_pre = P_pre.rows;
        const std::size_t d_post_expected = d_pre - kCloneDim;

        // verify that clone 0 and clone 1 blocks genuinely differ so the
        // fixture is a real discriminator (assert rather than silently pass)
        if (n_clones >= 2) {
            const std::size_t b0 = kNavDim;
            const std::size_t b1 = kNavDim + kCloneDim;
            double max_diff = 0.0;
            for (std::size_t r = 0; r < kCloneDim; ++r)
                for (std::size_t c = 0; c < kCloneDim; ++c)
                    max_diff = std::max(max_diff,
                                        std::abs(static_cast<double>(P_pre(b0 + r, b0 + c) - P_pre(b1 + r, b1 + c))));
            assert(max_diff > 1e-6 && "propagation must make clone blocks distinct");
        }

        be.marginalize_clone(drop_idx);
        const DynMat<T> P_post = be.covariance();

        Result res;
        res.dim_ok =
            (P_post.rows == d_post_expected) && (P_post.cols == d_post_expected) && (be.num_clones() == n_clones - 1);

        const std::size_t off = kNavDim + kCloneDim * drop_idx;
        const DynMat<T> Q = principal_submatrix(P_pre, off);
        res.psub_err = max_abs_diff(P_post, Q);
        res.psub_ok = (res.psub_err < tol);

        res.sym_err = symmetry_error(P_post);
        res.sym_ok = (res.sym_err < tol);

        return res;
    }
};

}  // namespace branes::tools::bench
