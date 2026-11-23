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
    // Boundary regression (pass side): a one-ULP increase in a single diagonal
    // entry must pass covariance.no_larger_than_prior because the scaled safety
    // tolerance covers double roundoff in the double-held matrices.
    const auto sc = B::make_scene();
    B::Input<double> in;
    in.clones = sc.clones;
    in.p_f = sc.feats[0];
    in.obs = B::track(sc.clones, sc.feats[0]);
    in.sigma = 0.01;
    auto out = B::run<double>(in, bn::kShipped);
    REQUIRE(out.applied);
    // Add one ULP to P[0,0] — within double roundoff, must pass.
    out.cov_after[0] = std::nextafter(out.cov_after[0], std::numeric_limits<double>::infinity());
    const auto rs = B::invariants<double>(in, out, {});
    const auto it =
        std::find_if(rs.begin(), rs.end(), [](const auto& r) { return r.name == "covariance.no_larger_than_prior"; });
    REQUIRE(it != rs.end());
    INFO("Loewner value = " << it->value);
    REQUIRE(it->pass);
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
