// src/elements/ctria3.cpp
// CTRIA3: 3-node constant-strain triangle shell element.
// Membrane: CST (plane-stress), closed-form integration
// Bending:  DKT (Discrete Kirchhoff Triangle), Batoz-Bathe-Ho (1980): normal
//           slopes interpolated quadratically from corner and midside values,
//           with Kirchhoff constraints imposed at the corners and edge
//           midsides; the linear curvature field is integrated exactly.

#include "elements/ctria3.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <format>

namespace vibestran {

CTria3::CTria3(ElementId eid, PropertyId pid,
               std::array<NodeId, 3> node_ids,
               const Model& model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {}

const PShell& CTria3::pshell() const {
    const auto& prop = model_.property(pid_);
    if (!std::holds_alternative<PShell>(prop))
        throw SolverError(std::format("CTRIA3 {}: property {} is not PSHELL", eid_.value, pid_.value));
    return std::get<PShell>(prop);
}

const Mat1& CTria3::material() const {
    return model_.material(pshell().mid1);
}

double CTria3::thickness() const { return pshell().t; }

std::array<Vec3, 3> CTria3::node_coords() const {
    return {model_.node(nodes_[0]).position,
            model_.node(nodes_[1]).position,
            model_.node(nodes_[2]).position};
}

// Local element frame for arbitrary 3D orientation.
// e1 = along node1→node2, e3 = normal (e1×edge13), e2 = e3×e1
// xl[n], yl[n] are local in-plane coords of each node.
// T is the 18×18 block-diagonal rotation matrix: K_global = T^T * K_local * T
struct TriaFrame {
    Vec3 e1, e2, e3;
    std::array<double, 3> xl{};
    std::array<double, 3> yl{};
    Eigen::Matrix<double, 18, 18> T;
};

// eid is used only for error messages when nodes are degenerate.
static TriaFrame compute_tria_frame(const std::array<Vec3, 3>& g, ElementId eid) {
    TriaFrame fr;
    Vec3 v12 = g[1] - g[0];
    Vec3 v13 = g[2] - g[0];
    Vec3 normal = v12.cross(v13);
    if (normal.norm() < 1e-15)
        throw SolverError(std::format("CTRIA3 {}: zero area (collinear nodes)", eid.value));
    fr.e3 = normal.normalized();
    fr.e1 = v12.normalized();
    fr.e2 = fr.e3.cross(fr.e1);
    for (int n = 0; n < 3; ++n) {
        fr.xl[n] = g[n].dot(fr.e1);
        fr.yl[n] = g[n].dot(fr.e2);
    }
    Eigen::Matrix3d R;
    R << fr.e1.x, fr.e1.y, fr.e1.z,
         fr.e2.x, fr.e2.y, fr.e2.z,
         fr.e3.x, fr.e3.y, fr.e3.z;
    // Same slope-convention rotation correction as CQUAD4:
    // DOF3 = θx = slope-in-x, DOF4 = θy = slope-in-y (γ_xz = ∂w/∂x - θx = 0 for no shear).
    // Zero-shear condition gives: DOF3 = -(e2·ω_global), DOF4 = (e1·ω_global).
    // T_rot rows: [-e2, e1, e3]  (avoids constructing and multiplying M).
    fr.T.setZero();
    for (int n = 0; n < 3; ++n) {
        fr.T.template block<3, 3>(6 * n,     6 * n)     = R;
        auto Tr = fr.T.template block<3, 3>(6 * n + 3, 6 * n + 3);
        Tr.row(0) = -R.row(1);  // DOF3 = -(e2·ω)
        Tr.row(1) =  R.row(0);  // DOF4 = +(e1·ω)
        Tr.row(2) =  R.row(2);  // DOF5 = drilling
    }
    return fr;
}

Eigen::Matrix3d CTria3::plane_stress_D() const {
    const Mat1& m = material();
    double c = m.E / (1.0 - m.nu * m.nu);
    Eigen::Matrix3d D;
    D << c,      c*m.nu, 0,
         c*m.nu, c,      0,
         0,      0,      c*(1.0-m.nu)/2.0;
    return D;
}

Eigen::MatrixXd CTria3::membrane_stiffness(const std::array<double,3>& xl,
                                             const std::array<double,3>& yl) const {
    double x1 = xl[0], y1 = yl[0];
    double x2 = xl[1], y2 = yl[1];
    double x3 = xl[2], y3 = yl[2];

    // Area in local plane
    double A2 = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1);
    if (std::abs(A2) < 1e-15)
        throw SolverError(std::format("CTRIA3 {}: zero area", eid_.value));
    double A = 0.5 * A2;

    // Constant strain-displacement matrix B_m [3x6]
    // Strains: εxx, εyy, γxy
    // DOFs per node: u, v (2 per node, 6 total for membrane)
    double b1 = y2 - y3, b2 = y3 - y1, b3 = y1 - y2;
    double d1 = x3 - x2, d2 = x1 - x3, d3 = x2 - x1;

    Eigen::MatrixXd Bm(3, 6);
    Bm.setZero();
    Bm(0,0)=b1; Bm(0,2)=b2; Bm(0,4)=b3;
    Bm(1,1)=d1; Bm(1,3)=d2; Bm(1,5)=d3;
    Bm(2,0)=d1; Bm(2,1)=b1;
    Bm(2,2)=d2; Bm(2,3)=b2;
    Bm(2,4)=d3; Bm(2,5)=b3;
    Bm /= (2.0 * A);

    double t = thickness();
    Eigen::Matrix3d D = plane_stress_D();
    return t * A * Bm.transpose() * D * Bm;
}

Eigen::Matrix3d CTria3::bending_D() const {
    const PShell& ps = pshell();
    const Mat1& m = (ps.mid2.value != 0) ? model_.material(ps.mid2) : material();
    const double c = m.E / (1.0 - m.nu * m.nu);
    Eigen::Matrix3d Q;
    Q << c,      c*m.nu, 0,
         c*m.nu, c,      0,
         0,      0,      c*(1.0-m.nu)/2.0;
    const double t = ps.t;
    return (ps.twelveI_t3 * t * t * t / 12.0) * Q;
}

Eigen::Matrix<double, 3, 9>
CTria3::dkt_curvature_B(const std::array<double,3>& xl,
                        const std::array<double,3>& yl,
                        const double xi, const double eta) {
    // Unknowns per node: [w, s_x, s_y] with s = (w_x, w_y) the slope
    // rotations of the shell frame (local DOFs 2, 3, 4).  The slope field is
    // quadratic:  s = sum_c N_c s_c + sum_k N_k s_k  over corners c and edge
    // midsides k, where at the midside of edge (i -> j), with unit tangent t,
    // unit normal n, and length l,
    //   tangential slope = 3/(2l) (w_j - w_i) - (t.s_i + t.s_j)/4  (cubic w)
    //   normal slope     = (n.s_i + n.s_j)/2                        (linear)
    // i.e. s_k = 3/(2l) t (w_j - w_i) + (I/2 - 3/4 t t^T)(s_i + s_j).
    const double L1 = 1.0 - xi - eta, L2 = xi, L3 = eta;

    // Quadratic shape functions: corners 1-3, midsides 4 (2-3), 5 (3-1),
    // 6 (1-2), and their (xi, eta) derivatives.
    const std::array<double, 6> dN_dxi{-(4*L1 - 1), 4*L2 - 1, 0.0,
                                       4*L3, -4*L3, 4*L1 - 4*L2};
    const std::array<double, 6> dN_deta{-(4*L1 - 1), 0.0, 4*L3 - 1,
                                        4*L2, 4*L1 - 4*L3, -4*L2};
    const double x21 = xl[1] - xl[0], y21 = yl[1] - yl[0];
    const double x31 = xl[2] - xl[0], y31 = yl[2] - yl[0];
    const double det = x21 * y31 - x31 * y21;
    std::array<double, 6> dNx{}, dNy{};
    for (size_t q = 0; q < 6; ++q) {
        dNx[q] = ( y31 * dN_dxi[q] - y21 * dN_deta[q]) / det;
        dNy[q] = (-x31 * dN_dxi[q] + x21 * dN_deta[q]) / det;
    }

    // C_q (2x9): s_q = C_q U for each of the six interpolation points.
    std::array<Eigen::Matrix<double, 2, 9>, 6> C;
    for (auto& c : C) c.setZero();
    for (int c = 0; c < 3; ++c) {
        C[static_cast<size_t>(c)](0, 3*c + 1) = 1.0;
        C[static_cast<size_t>(c)](1, 3*c + 2) = 1.0;
    }
    static constexpr std::array<std::array<int, 2>, 3> kEdges{{{1, 2}, {2, 0}, {0, 1}}};
    for (size_t k = 0; k < 3; ++k) {
        const int i = kEdges[k][0], j = kEdges[k][1];
        const double dx = xl[static_cast<size_t>(j)] - xl[static_cast<size_t>(i)];
        const double dy = yl[static_cast<size_t>(j)] - yl[static_cast<size_t>(i)];
        const double l = std::sqrt(dx*dx + dy*dy);
        const Eigen::Vector2d t(dx / l, dy / l);
        const Eigen::Matrix2d P = 0.5 * Eigen::Matrix2d::Identity() - 0.75 * t * t.transpose();
        Eigen::Matrix<double, 2, 9>& Ck = C[3 + k];
        Ck.col(3*j) += 1.5 / l * t;
        Ck.col(3*i) -= 1.5 / l * t;
        Ck.block<2, 2>(0, 3*i + 1) += P;
        Ck.block<2, 2>(0, 3*j + 1) += P;
    }

    // Nastran curvature convention: kappa = -[sx_x, sy_y, sx_y + sy_x]
    // (= -[w_xx, w_yy, 2 w_xy]), matching CQUAD4.
    Eigen::Matrix<double, 3, 9> B = Eigen::Matrix<double, 3, 9>::Zero();
    for (size_t q = 0; q < 6; ++q) {
        B.row(0) -= dNx[q] * C[q].row(0);
        B.row(1) -= dNy[q] * C[q].row(1);
        B.row(2) -= dNy[q] * C[q].row(0) + dNx[q] * C[q].row(1);
    }
    return B;
}

Eigen::MatrixXd CTria3::bending_stiffness(const std::array<double,3>& xl,
                                            const std::array<double,3>& yl) const {
    // K_b = int B^T D B dA.  B is linear, so the integrand is quadratic and
    // the 3-point midside rule (weights A/3) is exact.
    const double A = 0.5 * std::abs((xl[1]-xl[0])*(yl[2]-yl[0]) -
                                    (xl[2]-xl[0])*(yl[1]-yl[0]));
    const Eigen::Matrix3d Db = bending_D();
    static constexpr std::array<std::array<double, 2>, 3> kPoints{
        {{0.5, 0.0}, {0.5, 0.5}, {0.0, 0.5}}};
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(9, 9);
    for (const auto& p : kPoints) {
        const Eigen::Matrix<double, 3, 9> B = dkt_curvature_B(xl, yl, p[0], p[1]);
        K += (A / 3.0) * B.transpose() * Db * B;
    }
    return K;
}

LocalKe CTria3::stiffness_matrix() const {
    auto g = node_coords();
    TriaFrame frame = compute_tria_frame(g, eid_);

    LocalKe Ke = LocalKe::Zero(NUM_DOFS, NUM_DOFS);

    // Membrane stiffness (DOF layout per node: u=6i, v=6i+1) in local frame
    Eigen::MatrixXd Km = membrane_stiffness(frame.xl, frame.yl); // 6x6
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            Ke(6*i+0, 6*j+0) += Km(2*i+0, 2*j+0);
            Ke(6*i+0, 6*j+1) += Km(2*i+0, 2*j+1);
            Ke(6*i+1, 6*j+0) += Km(2*i+1, 2*j+0);
            Ke(6*i+1, 6*j+1) += Km(2*i+1, 2*j+1);
        }

    // Bending stiffness (DOF layout per node: w=6i+2, θx=6i+3, θy=6i+4) in local frame
    // Local bending DOFs: w=3i+0, θx=3i+1, θy=3i+2
    Eigen::MatrixXd Kb = bending_stiffness(frame.xl, frame.yl); // 9x9
    const int bend_map[3] = {2, 3, 4}; // local bend dof → offset in 6-DOF node block
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    Ke(6*i+bend_map[a], 6*j+bend_map[b]) += Kb(3*i+a, 3*j+b);

    // Drilling stiffness stabilization for θz DOF
    double drill = 1e-6 * Ke.diagonal().maxCoeff();
    if (drill < 1e-10) drill = 1.0;
    for (int i = 0; i < 3; ++i)
        Ke(6*i+5, 6*i+5) += drill;

    // Transform local stiffness to global frame: K_global = T^T * K_local * T
    return frame.T.transpose() * Ke * frame.T;
}

LocalKe CTria3::mass_matrix() const {
    LocalKe Me = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
    const Mat1& mat = material();
    const double rho = mat.rho;
    if (rho == 0.0) return Me;
    const double t = thickness();

    // Closed-form consistent mass matrix for CST triangle.
    // Nodal mass integrals: integral(Li^2 dA) = A/6, integral(Li*Lj dA) = A/12 (i≠j)
    // Compute actual 3D area via cross product (handles arbitrary orientation).
    auto g = node_coords();
    TriaFrame frame = compute_tria_frame(g, eid_);
    Vec3 v12 = g[1] - g[0];
    Vec3 v13 = g[2] - g[0];
    double A = 0.5 * v12.cross(v13).norm();

    double m_diag = rho * t * A / 6.0;
    double m_off  = rho * t * A / 12.0;

    double r_diag = rho * t * t * t / 12.0 * A / 6.0;
    double r_off  = rho * t * t * t / 12.0 * A / 12.0;
    double drill  = rho * t * t * t / 1200.0 * A / 6.0; // small drilling mass

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            double m_trans = (a == b) ? m_diag : m_off;
            double m_rot   = (a == b) ? r_diag : r_off;
            // Translational (T1,T2,T3): indices 6a+0,1,2
            for (int d = 0; d < 3; ++d)
                Me(6*a+d, 6*b+d) = m_trans;
            // Rotational bending (R1,R2): indices 6a+3,4
            for (int d = 0; d < 2; ++d)
                Me(6*a+3+d, 6*b+3+d) = m_rot;
        }
        // Drilling (R3): diagonal only
        Me(6*a+5, 6*a+5) = drill;
    }
    // Transform local mass matrix to global frame: M_global = T^T * M_local * T
    return frame.T.transpose() * Me * frame.T;
}

LocalFe CTria3::thermal_load(std::span<const double> temperatures, double t_ref) const {
    LocalFe fe = LocalFe::Zero(NUM_DOFS);
    const double alpha = material().A;
    if (alpha == 0.0) return fe;

    auto g = node_coords();
    TriaFrame frame = compute_tria_frame(g, eid_);

    double x1=frame.xl[0], y1=frame.yl[0];
    double x2=frame.xl[1], y2=frame.yl[1];
    double x3=frame.xl[2], y3=frame.yl[2];

    double A2 = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1);
    // Use signed area for consistent B matrix sign (matches membrane_stiffness)
    double A  = 0.5 * A2;
    if (std::abs(A2) < 1e-15) return fe;

    // Average temperature over element
    double T_avg = (temperatures[0] + temperatures[1] + temperatures[2]) / 3.0;
    double dT = T_avg - t_ref;
    if (std::abs(dT) < 1e-15) return fe;

    double b1 = y2-y3, b2 = y3-y1, b3 = y1-y2;
    double d1 = x3-x2, d2 = x1-x3, d3 = x2-x1;

    Eigen::MatrixXd Bm(3, 6);
    Bm.setZero();
    Bm(0,0)=b1; Bm(0,2)=b2; Bm(0,4)=b3;
    Bm(1,1)=d1; Bm(1,3)=d2; Bm(1,5)=d3;
    Bm(2,0)=d1; Bm(2,1)=b1;
    Bm(2,2)=d2; Bm(2,3)=b2;
    Bm(2,4)=d3; Bm(2,5)=b3;
    Bm /= (2.0 * A);  // same sign as membrane_stiffness()

    Eigen::Matrix3d D = plane_stress_D();
    double t = thickness();
    Eigen::Vector3d eps_th(alpha*dT, alpha*dT, 0.0);
    // Integration weight = |A| (physical area, always positive)
    Eigen::VectorXd fe_mem = t * std::abs(A) * Bm.transpose() * D * eps_th;

    for (int n = 0; n < 3; ++n) {
        fe(6*n+0) += fe_mem(2*n+0);
        fe(6*n+1) += fe_mem(2*n+1);
    }
    // Transform local load vector to global frame: f_global = T^T * f_local
    return frame.T.transpose() * fe;
}

CTria3::PlateResponse
CTria3::recover_response(std::span<const double> global_displacements,
                         const double xi, const double eta,
                         const double temperature, const double t_ref) const {
    if (global_displacements.size() != static_cast<size_t>(NUM_DOFS)) {
        throw SolverError(std::format(
            "CTRIA3 {}: expected {} displacement entries for recovery, got {}",
            eid_.value, NUM_DOFS, global_displacements.size()));
    }
    const TriaFrame frame = compute_tria_frame(node_coords(), eid_);
    Eigen::Matrix<double, 18, 1> ug;
    for (int i = 0; i < NUM_DOFS; ++i)
        ug(i) = global_displacements[static_cast<size_t>(i)];
    const Eigen::Matrix<double, 18, 1> ul = frame.T * ug;

    const auto& xl = frame.xl;
    const auto& yl = frame.yl;
    const double A2 = (xl[1]-xl[0])*(yl[2]-yl[0]) - (xl[2]-xl[0])*(yl[1]-yl[0]);
    const double b1 = yl[1]-yl[2], b2 = yl[2]-yl[0], b3 = yl[0]-yl[1];
    const double d1 = xl[2]-xl[1], d2 = xl[0]-xl[2], d3 = xl[1]-xl[0];

    PlateResponse r;
    // Membrane (CST): constant strain, mechanical stress after the free
    // thermal strain.
    Eigen::Matrix<double, 3, 6> Bm = Eigen::Matrix<double, 3, 6>::Zero();
    Bm(0,0)=b1; Bm(0,2)=b2; Bm(0,4)=b3;
    Bm(1,1)=d1; Bm(1,3)=d2; Bm(1,5)=d3;
    Bm(2,0)=d1; Bm(2,1)=b1; Bm(2,2)=d2; Bm(2,3)=b2; Bm(2,4)=d3; Bm(2,5)=b3;
    Bm /= A2;
    Eigen::Matrix<double, 6, 1> u_mem;
    for (int n = 0; n < 3; ++n) {
        u_mem(2*n) = ul(6*n);
        u_mem(2*n + 1) = ul(6*n + 1);
    }
    const double alpha = material().A;
    const Eigen::Vector3d eps_th(alpha * (temperature - t_ref),
                                 alpha * (temperature - t_ref), 0.0);
    r.membrane_stress = plane_stress_D() * (Bm * u_mem - eps_th);
    r.membrane_resultant = thickness() * r.membrane_stress;

    // Bending (DKT).
    Eigen::Matrix<double, 9, 1> u_b;
    for (int n = 0; n < 3; ++n) {
        u_b(3*n) = ul(6*n + 2);
        u_b(3*n + 1) = ul(6*n + 3);
        u_b(3*n + 2) = ul(6*n + 4);
    }
    const Eigen::Matrix3d Db = bending_D();
    r.curvature = dkt_curvature_B(xl, yl, xi, eta) * u_b;
    r.bending_moment = Db * r.curvature;

    // DKT is a Kirchhoff element: there is no transverse shear strain, and
    // differentiating its element-wise linear moment field gives a
    // mesh-independent bias (it is not complete for cubic deflections), so
    // transverse shear resultants are left at zero rather than reported
    // inaccurately.
    return r;
}

std::vector<EqIndex> CTria3::global_dof_indices(const DofMap& dof_map) const {
    std::vector<EqIndex> result;
    result.reserve(NUM_DOFS);
    for (NodeId nid : nodes_) {
        const auto& blk = dof_map.block(nid);
        for (int d = 0; d < 6; ++d)
            result.push_back(blk.eq[d]);
    }
    return result;
}

} // namespace vibestran
