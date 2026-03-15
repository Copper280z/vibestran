// src/elements/solid_elements.cpp
// CHEXA8: 8-node trilinear hexahedron, 2x2x2 Gauss
// CTETRA4: 4-node linear tetrahedron, closed-form

#include "elements/solid_elements.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <format>

namespace nastran {

// ── Shared helpers ────────────────────────────────────────────────────────────

/// 3-D isotropic constitutive matrix D [6x6]
static Eigen::Matrix<double,6,6> isotropic_D(double E, double nu) {
    double lam = E * nu / ((1+nu)*(1-2*nu));
    double mu  = E / (2*(1+nu));
    Eigen::Matrix<double,6,6> D;
    D.setZero();
    D(0,0)=lam+2*mu; D(0,1)=lam;      D(0,2)=lam;
    D(1,0)=lam;      D(1,1)=lam+2*mu; D(1,2)=lam;
    D(2,0)=lam;      D(2,1)=lam;      D(2,2)=lam+2*mu;
    D(3,3)=mu; D(4,4)=mu; D(5,5)=mu;
    return D;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHEXA8
// ═══════════════════════════════════════════════════════════════════════════════

CHexa8::CHexa8(ElementId eid, PropertyId pid,
               std::array<NodeId, 8> node_ids,
               const Model& model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {}

const PSolid& CHexa8::psolid() const {
    const auto& prop = model_.property(pid_);
    if (!std::holds_alternative<PSolid>(prop))
        throw SolverError(std::format("CHEXA8 {}: property {} is not PSOLID", eid_.value, pid_.value));
    return std::get<PSolid>(prop);
}

const Mat1& CHexa8::material() const {
    return model_.material(psolid().mid);
}

Eigen::Matrix<double,6,6> CHexa8::constitutive_D() const {
    const Mat1& m = material();
    return isotropic_D(m.E, m.nu);
}

std::array<Vec3, 8> CHexa8::node_coords() const {
    std::array<Vec3, 8> c;
    for (int i = 0; i < 8; ++i)
        c[i] = model_.node(nodes_[i]).position;
    return c;
}

CHexa8::ShapeData CHexa8::shape_functions(double xi, double eta, double zeta) noexcept {
    // Trilinear shape functions for 8-node hex
    // Node ordering: (-1,-1,-1),(+1,-1,-1),(+1,+1,-1),(-1,+1,-1),
    //                (-1,-1,+1),(+1,-1,+1),(+1,+1,+1),(-1,+1,+1)
    const double xi_n[8]   = {-1,+1,+1,-1,-1,+1,+1,-1};
    const double eta_n[8]  = {-1,-1,+1,+1,-1,-1,+1,+1};
    const double zeta_n[8] = {-1,-1,-1,-1,+1,+1,+1,+1};

    ShapeData s;
    for (int i = 0; i < 8; ++i) {
        double xni = xi_n[i], eni = eta_n[i], zni = zeta_n[i];
        s.N[i]      = 0.125*(1+xni*xi)*(1+eni*eta)*(1+zni*zeta);
        s.dNdxi[i]  = 0.125*xni*(1+eni*eta)*(1+zni*zeta);
        s.dNdeta[i] = 0.125*eni*(1+xni*xi)*(1+zni*zeta);
        s.dNdzeta[i]= 0.125*zni*(1+xni*xi)*(1+eni*eta);
    }
    return s;
}

LocalKe CHexa8::stiffness_matrix() const {
    LocalKe Ke = LocalKe::Zero(NUM_DOFS, NUM_DOFS);
    auto coords = node_coords();
    Eigen::Matrix<double,6,6> D = constitutive_D();

    // Selective Reduced Integration (SRI) to eliminate volumetric locking:
    // D = D_dev + D_vol, where D_vol = lam * e*e^T (e = [1,1,1,0,0,0]^T)
    // D_dev uses full 2x2x2 Gauss; D_vol uses 1-point centroidal integration.
    // This prevents over-stiffness in constrained states without losing stability.
    double lam = D(0,1);  // D(i,j) = lam for i,j in {0,1,2}, since D(0,0)=lam+2mu
    Eigen::Matrix<double,6,6> D_vol = Eigen::Matrix<double,6,6>::Zero();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            D_vol(i,j) = lam;
    Eigen::Matrix<double,6,6> D_dev = D - D_vol;

    // 2x2x2 Gauss quadrature for deviatoric part
    const double gp = 1.0/std::sqrt(3.0);
    const double gpts[2] = {-gp, gp};
    const double gwts[2] = {1.0, 1.0};

    auto build_B = [&](const ShapeData& sd, const Eigen::Matrix3d& Jinv,
                       Eigen::MatrixXd& B) {
        B.setZero();
        for (int n = 0; n < 8; ++n) {
            double dnx = Jinv(0,0)*sd.dNdxi[n] + Jinv(0,1)*sd.dNdeta[n] + Jinv(0,2)*sd.dNdzeta[n];
            double dny = Jinv(1,0)*sd.dNdxi[n] + Jinv(1,1)*sd.dNdeta[n] + Jinv(1,2)*sd.dNdzeta[n];
            double dnz = Jinv(2,0)*sd.dNdxi[n] + Jinv(2,1)*sd.dNdeta[n] + Jinv(2,2)*sd.dNdzeta[n];
            int c0 = 3*n;
            B(0,c0+0)=dnx;
            B(1,c0+1)=dny;
            B(2,c0+2)=dnz;
            B(3,c0+0)=dny; B(3,c0+1)=dnx;
            B(4,c0+1)=dnz; B(4,c0+2)=dny;
            B(5,c0+0)=dnz; B(5,c0+2)=dnx;
        }
    };

    for (int gi = 0; gi < 2; ++gi)
    for (int gj = 0; gj < 2; ++gj)
    for (int gk = 0; gk < 2; ++gk) {
        double xi   = gpts[gi], eta  = gpts[gj], zeta = gpts[gk];
        double wi   = gwts[gi],  wj   = gwts[gj],  wk   = gwts[gk];

        auto sd = shape_functions(xi, eta, zeta);

        // Jacobian [3x3]
        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 8; ++n) {
            J(0,0)+=sd.dNdxi[n]  *coords[n].x; J(0,1)+=sd.dNdxi[n]  *coords[n].y; J(0,2)+=sd.dNdxi[n]  *coords[n].z;
            J(1,0)+=sd.dNdeta[n] *coords[n].x; J(1,1)+=sd.dNdeta[n] *coords[n].y; J(1,2)+=sd.dNdeta[n] *coords[n].z;
            J(2,0)+=sd.dNdzeta[n]*coords[n].x; J(2,1)+=sd.dNdzeta[n]*coords[n].y; J(2,2)+=sd.dNdzeta[n]*coords[n].z;
        }
        double detJ = J.determinant();
        if (detJ <= 0)
            throw SolverError(std::format("CHEXA8 {}: non-positive Jacobian det={:.6g}", eid_.value, detJ));
        Eigen::Matrix3d Jinv = J.inverse();

        Eigen::MatrixXd B(6, NUM_DOFS);
        build_B(sd, Jinv, B);

        Ke += B.transpose() * D_dev * B * detJ * wi * wj * wk;
    }

    // Volumetric part: 1-point centroidal integration (weight = 2*2*2 = 8)
    {
        auto sd0 = shape_functions(0.0, 0.0, 0.0);
        Eigen::Matrix3d J0 = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 8; ++n) {
            J0(0,0)+=sd0.dNdxi[n]  *coords[n].x; J0(0,1)+=sd0.dNdxi[n]  *coords[n].y; J0(0,2)+=sd0.dNdxi[n]  *coords[n].z;
            J0(1,0)+=sd0.dNdeta[n] *coords[n].x; J0(1,1)+=sd0.dNdeta[n] *coords[n].y; J0(1,2)+=sd0.dNdeta[n] *coords[n].z;
            J0(2,0)+=sd0.dNdzeta[n]*coords[n].x; J0(2,1)+=sd0.dNdzeta[n]*coords[n].y; J0(2,2)+=sd0.dNdzeta[n]*coords[n].z;
        }
        double detJ0 = J0.determinant();
        if (detJ0 <= 0)
            throw SolverError(std::format("CHEXA8 {}: non-positive centroidal Jacobian det={:.6g}", eid_.value, detJ0));
        Eigen::Matrix3d Jinv0 = J0.inverse();

        Eigen::MatrixXd B0(6, NUM_DOFS);
        build_B(sd0, Jinv0, B0);

        Ke += B0.transpose() * D_vol * B0 * detJ0 * 8.0;
    }

    return Ke;
}

LocalFe CHexa8::thermal_load(std::span<const double> temperatures, double t_ref) const {
    LocalFe fe = LocalFe::Zero(NUM_DOFS);
    const double alpha = material().A;
    if (alpha == 0.0) return fe;

    auto coords = node_coords();
    Eigen::Matrix<double,6,6> D = constitutive_D();

    const double gp = 1.0/std::sqrt(3.0);
    const double gpts[2] = {-gp, gp};

    for (int gi = 0; gi < 2; ++gi)
    for (int gj = 0; gj < 2; ++gj)
    for (int gk = 0; gk < 2; ++gk) {
        double xi=gpts[gi], eta=gpts[gj], zeta=gpts[gk];

        auto sd = shape_functions(xi, eta, zeta);

        double T = 0;
        for (int n = 0; n < 8; ++n) T += sd.N[n] * temperatures[n];
        double dT = T - t_ref;
        if (std::abs(dT) < 1e-15) continue;

        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        for (int n = 0; n < 8; ++n) {
            J(0,0)+=sd.dNdxi[n]  *coords[n].x; J(0,1)+=sd.dNdxi[n]  *coords[n].y; J(0,2)+=sd.dNdxi[n]  *coords[n].z;
            J(1,0)+=sd.dNdeta[n] *coords[n].x; J(1,1)+=sd.dNdeta[n] *coords[n].y; J(1,2)+=sd.dNdeta[n] *coords[n].z;
            J(2,0)+=sd.dNdzeta[n]*coords[n].x; J(2,1)+=sd.dNdzeta[n]*coords[n].y; J(2,2)+=sd.dNdzeta[n]*coords[n].z;
        }
        double detJ = J.determinant();
        Eigen::Matrix3d Jinv = J.inverse();

        Eigen::MatrixXd dNdx(3, 8);
        for (int n = 0; n < 8; ++n) {
            dNdx(0,n) = Jinv(0,0)*sd.dNdxi[n] + Jinv(0,1)*sd.dNdeta[n] + Jinv(0,2)*sd.dNdzeta[n];
            dNdx(1,n) = Jinv(1,0)*sd.dNdxi[n] + Jinv(1,1)*sd.dNdeta[n] + Jinv(1,2)*sd.dNdzeta[n];
            dNdx(2,n) = Jinv(2,0)*sd.dNdxi[n] + Jinv(2,1)*sd.dNdeta[n] + Jinv(2,2)*sd.dNdzeta[n];
        }

        Eigen::MatrixXd B(6, NUM_DOFS);
        B.setZero();
        for (int n = 0; n < 8; ++n) {
            double dnx=dNdx(0,n), dny=dNdx(1,n), dnz=dNdx(2,n);
            int c0=3*n;
            B(0,c0+0)=dnx; B(1,c0+1)=dny; B(2,c0+2)=dnz;
            B(3,c0+0)=dny; B(3,c0+1)=dnx;
            B(4,c0+1)=dnz; B(4,c0+2)=dny;
            B(5,c0+0)=dnz; B(5,c0+2)=dnx;
        }

        Eigen::Matrix<double,6,1> eps_th;
        eps_th << alpha*dT, alpha*dT, alpha*dT, 0, 0, 0;
        fe += B.transpose() * D * eps_th * detJ;
    }
    return fe;
}

std::vector<EqIndex> CHexa8::global_dof_indices(const DofMap& dof_map) const {
    std::vector<EqIndex> result;
    result.reserve(NUM_DOFS);
    static constexpr int solid_dofs[3] = {0,1,2}; // T1,T2,T3 only
    for (NodeId nid : nodes_)
        for (int d : solid_dofs)
            result.push_back(dof_map.eq_index(nid, d));
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CTETRA4
// ═══════════════════════════════════════════════════════════════════════════════

CTetra4::CTetra4(ElementId eid, PropertyId pid,
                 std::array<NodeId, 4> node_ids,
                 const Model& model)
    : eid_(eid), pid_(pid), nodes_(node_ids), model_(model) {}

const PSolid& CTetra4::psolid() const {
    const auto& prop = model_.property(pid_);
    if (!std::holds_alternative<PSolid>(prop))
        throw SolverError(std::format("CTETRA4 {}: property {} is not PSOLID", eid_.value, pid_.value));
    return std::get<PSolid>(prop);
}

const Mat1& CTetra4::material() const {
    return model_.material(psolid().mid);
}

Eigen::Matrix<double,6,6> CTetra4::constitutive_D() const {
    const Mat1& m = material();
    return isotropic_D(m.E, m.nu);
}

std::array<Vec3, 4> CTetra4::node_coords() const {
    return {model_.node(nodes_[0]).position,
            model_.node(nodes_[1]).position,
            model_.node(nodes_[2]).position,
            model_.node(nodes_[3]).position};
}

LocalKe CTetra4::stiffness_matrix() const {
    auto c = node_coords();

    // CST tetrahedron: constant strain → B is constant
    // Volume via determinant of coordinate matrix
    // [x2-x1, x3-x1, x4-x1]
    // [y2-y1, y3-y1, y4-y1]
    // [z2-z1, z3-z1, z4-z1]
    Eigen::Matrix3d CM;
    for (int j = 0; j < 3; ++j) {
        CM(j,0) = (j==0?c[1].x:j==1?c[1].y:c[1].z) - (j==0?c[0].x:j==1?c[0].y:c[0].z);
        CM(j,1) = (j==0?c[2].x:j==1?c[2].y:c[2].z) - (j==0?c[0].x:j==1?c[0].y:c[0].z);
        CM(j,2) = (j==0?c[3].x:j==1?c[3].y:c[3].z) - (j==0?c[0].x:j==1?c[0].y:c[0].z);
    }
    // Cleaner version:
    double x1=c[0].x,y1=c[0].y,z1=c[0].z;
    double x2=c[1].x,y2=c[1].y,z2=c[1].z;
    double x3=c[2].x,y3=c[2].y,z3=c[2].z;
    double x4=c[3].x,y4=c[3].y,z4=c[3].z;

    Eigen::Matrix4d A4;
    A4 << 1,x1,y1,z1,
          1,x2,y2,z2,
          1,x3,y3,z3,
          1,x4,y4,z4;
    double V6 = A4.determinant(); // = 6 * volume
    double V  = V6 / 6.0;
    if (V <= 0)
        throw SolverError(std::format("CTETRA4 {}: non-positive volume={:.6g}", eid_.value, V));

    // Shape fn coefficients: N_i = (a_i + b_i*x + c_i*y + d_i*z) / (6V)
    // Computed via cofactors of A4
    // dN_i/dx = b_i / (6V), etc.
    // Cofactor matrix (signed minors of A4 transposed)
    Eigen::Matrix4d cofA = Eigen::Matrix4d::Zero();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            // Minor by removing row j, col i (transposed for cofactor)
            Eigen::Matrix3d minor3;
            int ri = 0;
            for (int r = 0; r < 4; ++r) {
                if (r == j) continue;
                int ci_ = 0;
                for (int cc = 0; cc < 4; ++cc) {
                    if (cc == i) continue;
                    minor3(ri, ci_++) = A4(r, cc);
                }
                ri++;
            }
            cofA(i,j) = std::pow(-1.0, i+j) * minor3.determinant();
        }

    // dN_i/dx = cofA(i,1) / V6
    // dN_i/dy = cofA(i,2) / V6
    // dN_i/dz = cofA(i,3) / V6
    Eigen::MatrixXd B(6, NUM_DOFS);
    B.setZero();
    for (int n = 0; n < 4; ++n) {
        double bx = cofA(1,n) / V6;
        double by = cofA(2,n) / V6;
        double bz = cofA(3,n) / V6;
        int c0 = 3*n;
        B(0,c0+0)=bx;
        B(1,c0+1)=by;
        B(2,c0+2)=bz;
        B(3,c0+0)=by; B(3,c0+1)=bx;
        B(4,c0+1)=bz; B(4,c0+2)=by;
        B(5,c0+0)=bz; B(5,c0+2)=bx;
    }

    Eigen::Matrix<double,6,6> D = constitutive_D();
    return V * B.transpose() * D * B;
}

LocalFe CTetra4::thermal_load(std::span<const double> temperatures, double t_ref) const {
    LocalFe fe = LocalFe::Zero(NUM_DOFS);
    const double alpha = material().A;
    if (alpha == 0.0) return fe;

    // For CST tetrahedron, B is constant → same computation as stiffness
    // fe = V * Bᵀ * D * eps_th
    auto c = node_coords();
    double x1=c[0].x,y1=c[0].y,z1=c[0].z;
    double x2=c[1].x,y2=c[1].y,z2=c[1].z;
    double x3=c[2].x,y3=c[2].y,z3=c[2].z;
    double x4=c[3].x,y4=c[3].y,z4=c[3].z;

    Eigen::Matrix4d A4;
    A4 << 1,x1,y1,z1, 1,x2,y2,z2, 1,x3,y3,z3, 1,x4,y4,z4;
    double V6 = A4.determinant();
    double V  = V6 / 6.0;

    Eigen::Matrix4d cofA = Eigen::Matrix4d::Zero();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            Eigen::Matrix3d minor3;
            int ri = 0;
            for (int r = 0; r < 4; ++r) {
                if (r == j) continue;
                int ci_ = 0;
                for (int cc = 0; cc < 4; ++cc) {
                    if (cc == i) continue;
                    minor3(ri, ci_++) = A4(r, cc);
                }
                ri++;
            }
            cofA(i,j) = std::pow(-1.0, i+j) * minor3.determinant();
        }

    Eigen::MatrixXd B(6, NUM_DOFS);
    B.setZero();
    for (int n = 0; n < 4; ++n) {
        double bx = cofA(1,n)/V6, by = cofA(2,n)/V6, bz = cofA(3,n)/V6;
        int c0=3*n;
        B(0,c0+0)=bx; B(1,c0+1)=by; B(2,c0+2)=bz;
        B(3,c0+0)=by; B(3,c0+1)=bx;
        B(4,c0+1)=bz; B(4,c0+2)=by;
        B(5,c0+0)=bz; B(5,c0+2)=bx;
    }

    double T_avg = (temperatures[0]+temperatures[1]+temperatures[2]+temperatures[3]) / 4.0;
    double dT = T_avg - t_ref;
    if (std::abs(dT) < 1e-15) return fe;

    Eigen::Matrix<double,6,6> D = constitutive_D();
    Eigen::Matrix<double,6,1> eps_th;
    eps_th << alpha*dT, alpha*dT, alpha*dT, 0, 0, 0;
    fe = V * B.transpose() * D * eps_th;
    return fe;
}

std::vector<EqIndex> CTetra4::global_dof_indices(const DofMap& dof_map) const {
    std::vector<EqIndex> result;
    result.reserve(NUM_DOFS);
    static constexpr int solid_dofs[3] = {0,1,2};
    for (NodeId nid : nodes_)
        for (int d : solid_dofs)
            result.push_back(dof_map.eq_index(nid, d));
    return result;
}

} // namespace nastran
