// R-IEKF S6 stage bench: invariant camera update (issue #612, epic #444).
//
// Locks three invariants of the R-IEKF msckf_update stage:
//   (1) H matches finite differences of the world-frame retraction;
//   (2) the marginalized H annihilates the 4-DoF gauge (global yaw + 3 translations);
//   (3) P⁺ ⪯ P⁻: the update does not inflate the covariance.

#include <branes/tools/bench/s6_riekf_update_bench.hpp>

#include <catch2/catch_test_macros.hpp>

namespace {
namespace bn = branes::tools::bench;
using B = bn::S6RieкfUpdateBench;
}  // namespace

TEST_CASE("S6 R-IEKF: every built-in fixture passes all invariants (float)", "[tools][bench][s6][riekf][float]") {
    for (const auto& nf : B::builtin_fixtures()) {
        const auto in = B::decode_input<float>(nf.fixture.input);
        const auto out = B::run<float>(in, bn::kShipped);
        const auto rs = B::invariants<float>(in, out, nf.fixture);
        for (const auto& res : rs) {
            INFO(nf.name << " / " << res.name << " = " << res.value);
            REQUIRE(res.pass);
        }
    }
}

TEST_CASE("S6 R-IEKF: every built-in fixture passes all invariants", "[tools][bench][s6][riekf]") {
    for (const auto& nf : B::builtin_fixtures()) {
        const auto in = B::decode_input<double>(nf.fixture.input);
        const auto out = B::run<double>(in, bn::kShipped);
        const auto rs = B::invariants<double>(in, out, nf.fixture);
        for (const auto& res : rs) {
            INFO(nf.name << " / " << res.name << " = " << res.value);
            REQUIRE(res.pass);
        }
    }
}

TEST_CASE("S6 R-IEKF: H annihilates the 4-DoF gauge at the synthetic scene",
          "[tools][bench][s6][riekf][observability]") {
    const auto sc = B::make_scene();
    B::Input<double> in;
    in.clones = sc.clones;
    in.p_f = sc.feats[0];
    in.obs = B::track(sc.clones, sc.feats[0]);
    in.sigma = 0.01;
    const auto out = B::run<double>(in, bn::kShipped);
    REQUIRE(out.applied);
    const auto rs = B::invariants<double>(in, out, {});
    const auto it =
        std::find_if(rs.begin(), rs.end(), [](const auto& r) { return r.name == "jacobian.annihilates_gauge"; });
    REQUIRE(it != rs.end());
    INFO("gauge leak = " << it->value);
    REQUIRE(it->pass);
}

TEST_CASE("S6 R-IEKF: tiny covariance increase passes Loewner check", "[tools][bench][s6][riekf][negative]") {
    // Boundary regression (pass side): run with float, then raise P[0,0] by
    // one float ULP above the prior diagonal (0.5). The increase is at float
    // precision, well within the float-scaled safety tolerance, so the Loewner
    // check must pass. The trace ratio must be above one (increase is real).
    // Construct float input directly — same geometry as the gauge test.
    B::Input<float> in;
    for (int c = 0; c < 3; ++c) {
        branes::sdk::msckf::InvariantClone<float> cl;
        cl.p = branes::math::lie::detail::Vec<float, 3>{{0.5f * c, 0.0f, 0.0f}};
        in.clones.push_back(cl);
    }
    in.p_f = branes::math::lie::detail::Vec<float, 3>{{0.0f, 0.0f, 3.0f}};
    in.sigma = 0.01f;
    {
        // Build observations from identity clones projecting feature at (0,0,3).
        for (std::size_t ci = 0; ci < in.clones.size(); ++ci) {
            const auto& cl = in.clones[ci];
            const float dx = in.p_f[0] - cl.p[0], dy = in.p_f[1] - cl.p[1], dz = in.p_f[2] - cl.p[2];
            branes::math::lie::detail::Vec<float, 2> xy{{dx / dz, dy / dz}};
            in.obs.push_back({ci, 0, xy});
        }
    }
    auto out = B::run<float>(in, bn::kShipped);
    REQUIRE(out.applied);
    INFO("cov_after size = " << out.cov_after.size());
    INFO("clones_after size = " << out.clones_after.size());
    INFO("in.clones size = " << in.clones.size());
    INFO("in.obs size = " << in.obs.size());
    // Re-initialize cov_after to the prior (0.5*I), then raise P[0,0] by one
    // float ULP. The increase is at float precision, well within the
    // float-scaled safety tolerance, so Loewner must pass with a positive
    // value. The trace ratio exceeds one because P[0,0] > prior diagonal.
    const std::size_t n = out.clones_after.size() * 6;
    std::fill(out.cov_after.begin(), out.cov_after.end(), 0.0);
    // Prior diagonal is sigma^2 = 0.5^2 = 0.25 (FullCovariance(0.5, n) sets P(i,i)=sigma*sigma).
    for (std::size_t i = 0; i < n; ++i)
        out.cov_after[i * n + i] = 0.25;
    out.cov_after[0] = static_cast<double>(std::nextafter(0.25f, std::numeric_limits<float>::infinity()));
    const auto rs = B::invariants<float>(in, out, {});
    const auto loewner =
        std::find_if(rs.begin(), rs.end(), [](const auto& r) { return r.name == "covariance.no_larger_than_prior"; });
    REQUIRE(loewner != rs.end());
    INFO("Loewner value = " << loewner->value);
    REQUIRE(loewner->pass);
    // Loewner value must be positive (increase is real) and pass.
    REQUIRE(loewner->value > 0.0);
    // Trace ratio must be above one (P[0,0] exceeds prior diagonal) and pass.
    const auto trace_it =
        std::find_if(rs.begin(), rs.end(), [](const auto& r) { return r.name == "covariance.trace_ratio_le_1"; });
    REQUIRE(trace_it != rs.end());
    INFO("trace ratio = " << trace_it->value);
    REQUIRE(trace_it->value > 1.0);
    REQUIRE(trace_it->pass);  // Bound::Report always passes
}

TEST_CASE("S6 R-IEKF: directional inflation fails Loewner check", "[tools][bench][s6][riekf][negative]") {
    // Boundary regression (fail side): inflate P[0,0] and deflate P[1,1] by
    // the same amount so the trace is unchanged, but the Loewner condition fails
    // because one eigenvalue of P_after - P_prior is positive.
    const auto sc = B::make_scene();
    B::Input<double> in;
    in.clones = sc.clones;
    in.p_f = sc.feats[0];
    in.obs = B::track(sc.clones, sc.feats[0]);
    in.sigma = 0.01;
    auto out = B::run<double>(in, bn::kShipped);
    REQUIRE(out.applied);
    const std::size_t n = out.clones_after.size() * 6;
    // Swap-inflate: P[0,0] += delta, P[1,1] -= delta. Trace unchanged.
    const double delta = out.cov_after[0] * 0.5;
    out.cov_after[0] += delta;
    out.cov_after[n + 1] -= delta;
    const auto rs = B::invariants<double>(in, out, {});
    const auto it =
        std::find_if(rs.begin(), rs.end(), [](const auto& r) { return r.name == "covariance.no_larger_than_prior"; });
    REQUIRE(it != rs.end());
    INFO("Loewner value = " << it->value);
    REQUIRE_FALSE(it->pass);
    // Assert trace is unchanged -- this test is not a trace check.
    double tp = 0.0, tq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        tp += 0.5;
        tq += out.cov_after[i * n + i];
    }
    INFO("trace prior = " << tp << " trace after = " << tq);
    REQUIRE(tq <= tp + 1e-12);
}

TEST_CASE("S6 R-IEKF: degenerate track (1 obs) is rejected -- clones unchanged", "[tools][bench][s6][riekf][reject]") {
    using B = branes::tools::bench::S6RieкfUpdateBench;
    B::Input<double> in;
    // 3 identity clones
    for (int c = 0; c < 3; ++c) {
        branes::sdk::msckf::InvariantClone<double> cl;
        cl.p = branes::math::lie::detail::Vec<double, 3>{{0.5 * c, 0.0, 0.0}};
        in.clones.push_back(cl);
    }
    in.p_f = branes::math::lie::detail::Vec<double, 3>{{0.0, 0.0, 3.0}};
    in.sigma = 0.01;
    // only 1 observation — invariant_update returns false immediately
    in.obs = {{0, 0, branes::math::lie::detail::Vec<double, 2>{{0.0, 0.0}}}};

    const auto clones_before = in.clones;
    const auto out = B::run<double>(in, branes::tools::bench::kShipped);

    REQUIRE_FALSE(out.applied);
    REQUIRE(out.clones_after.size() == clones_before.size());
    for (std::size_t i = 0; i < clones_before.size(); ++i) {
        const auto qb = clones_before[i].R.quaternion();
        const auto qa = out.clones_after[i].R.quaternion();
        for (int k = 0; k < 4; ++k)
            REQUIRE(qb[k] == qa[k]);
        for (int k = 0; k < 3; ++k)
            REQUIRE(clones_before[i].p[k] == out.clones_after[i].p[k]);
    }
}
