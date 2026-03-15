// tests/unit/test_bdf_parser.cpp
// Tests for BDF field parsing: numeric formats, card splitting, model population.

#include <gtest/gtest.h>
#include "io/bdf_parser.hpp"
#include "core/model.hpp"

using namespace nastran;

// ── Numeric format parsing ────────────────────────────────────────────────────

TEST(BdfParser, ParseSimpleBdf) {
    // Minimal valid BDF: one node, one material, one property, one element, one SPC
    const std::string bdf = R"(
SOL 101
BEGIN BULK
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,1.0,1.0,0.0
GRID,4,,0.0,1.0,0.0
MAT1,1,2.0E7,,0.3
PSHELL,1,1,0.1
CQUAD4,1,1,1,2,3,4
SPC1,1,123456,1,2,3,4
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    EXPECT_EQ(m.nodes.size(), 4u);
    EXPECT_EQ(m.elements.size(), 1u);
    EXPECT_EQ(m.materials.size(), 1u);
    EXPECT_EQ(m.properties.size(), 1u);
    EXPECT_EQ(m.spcs.size(), 4u);
}

TEST(BdfParser, SmallFieldGridCard) {
    const std::string bdf = R"(
BEGIN BULK
GRID           1       0     1.5     2.5     3.5
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.nodes.size(), 1u);
    auto& n = m.nodes.at(NodeId{1});
    EXPECT_DOUBLE_EQ(n.position.x, 1.5);
    EXPECT_DOUBLE_EQ(n.position.y, 2.5);
    EXPECT_DOUBLE_EQ(n.position.z, 3.5);
}

TEST(BdfParser, FreeFieldWithCommas) {
    const std::string bdf = R"(
BEGIN BULK
GRID,10,,3.14,2.72,1.41
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.nodes.size(), 1u);
    auto& n = m.nodes.at(NodeId{10});
    EXPECT_NEAR(n.position.x, 3.14, 1e-10);
    EXPECT_NEAR(n.position.y, 2.72, 1e-10);
    EXPECT_NEAR(n.position.z, 1.41, 1e-10);
}

TEST(BdfParser, NastranImplicitExponent) {
    // "1.5+3" is Nastran shorthand for 1.5E3 = 1500
    const std::string bdf = R"(
BEGIN BULK
MAT1,1,2.9+7,,3.0-1
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.materials.size(), 1u);
    const Mat1& mat = m.materials.at(MaterialId{1});
    EXPECT_NEAR(mat.E,  2.9e7, 1.0);
    EXPECT_NEAR(mat.nu, 0.30,  1e-10);
}

TEST(BdfParser, Mat1DerivedShearModulus) {
    // If G is blank, G = E / (2*(1+nu))
    const std::string bdf = R"(
BEGIN BULK
MAT1,1,2.0E7,,0.25
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    const Mat1& mat = m.materials.at(MaterialId{1});
    double G_expected = 2.0e7 / (2.0 * 1.25);
    EXPECT_NEAR(mat.G, G_expected, 1.0);
}

TEST(BdfParser, ForceCard) {
    const std::string bdf = R"(
BEGIN BULK
GRID,1,,0,0,0
FORCE,1,1,0,1000.0,0.0,0.0,1.0
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    ASSERT_EQ(m.loads.size(), 1u);
    const ForceLoad& f = std::get<ForceLoad>(m.loads[0]);
    EXPECT_EQ(f.sid.value, 1);
    EXPECT_EQ(f.node.value, 1);
    EXPECT_DOUBLE_EQ(f.scale, 1000.0);
    EXPECT_DOUBLE_EQ(f.direction.z, 1.0);
}

TEST(BdfParser, SPC1WithThruRange) {
    const std::string bdf = R"(
BEGIN BULK
GRID,1,,0,0,0
GRID,2,,1,0,0
GRID,3,,2,0,0
SPC1,1,123,1,THRU,3
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    // Should create 3 SPC entries (one per node)
    EXPECT_EQ(m.spcs.size(), 3u);
    for (const auto& spc : m.spcs) {
        EXPECT_EQ(spc.sid.value, 1);
        EXPECT_TRUE(spc.dofs.has(1) && spc.dofs.has(2) && spc.dofs.has(3));
    }
}

TEST(BdfParser, TempCard) {
    const std::string bdf = R"(
BEGIN BULK
GRID,1,,0,0,0
GRID,2,,1,0,0
TEMP,1,1,100.0,2,200.0
ENDDATA
)";
    Model m = BdfParser::parse_string(bdf);
    EXPECT_EQ(m.loads.size(), 2u);
    const TempLoad& t0 = std::get<TempLoad>(m.loads[0]);
    const TempLoad& t1 = std::get<TempLoad>(m.loads[1]);
    EXPECT_DOUBLE_EQ(t0.temperature, 100.0);
    EXPECT_DOUBLE_EQ(t1.temperature, 200.0);
}

TEST(BdfParser, ModelValidationCatchesMissingNode) {
    Model m;
    ElementData e;
    e.id   = ElementId{1};
    e.pid  = PropertyId{1};
    e.type = ElementType::CQUAD4;
    e.nodes = {NodeId{1},NodeId{2},NodeId{3},NodeId{4}}; // nodes not in model
    m.elements.push_back(e);
    EXPECT_THROW(m.validate(), SolverError);
}

// ── Model accessor methods (only used in tests) ───────────────────────────────

static Model make_simple_model() {
    const std::string bdf = R"(
SOL 101
BEGIN BULK
GRID,1,,0.0,0.0,0.0
GRID,2,,1.0,0.0,0.0
GRID,3,,1.0,1.0,0.0
GRID,4,,0.0,1.0,0.0
MAT1,1,2.0E7,,0.3
PSHELL,1,1,0.1
CQUAD4,1,1,1,2,3,4
SPC1,1,123456,1
FORCE,1,2,0,500.0,1.0,0.0,0.0
ENDDATA
)";
    return BdfParser::parse_string(bdf);
}

TEST(Model, NodeAccessorReturnsCorrectPosition) {
    Model m = make_simple_model();
    const GridPoint& gp = m.node(NodeId{3});
    EXPECT_NEAR(gp.position.x, 1.0, 1e-10);
    EXPECT_NEAR(gp.position.y, 1.0, 1e-10);
}

TEST(Model, NodeAccessorThrowsForMissingNode) {
    Model m = make_simple_model();
    EXPECT_THROW(m.node(NodeId{999}), SolverError);
}

TEST(Model, MaterialAccessorReturnsCorrectE) {
    Model m = make_simple_model();
    const Mat1& mat = m.material(MaterialId{1});
    EXPECT_NEAR(mat.E, 2.0e7, 1.0);
    EXPECT_NEAR(mat.nu, 0.3, 1e-10);
}

TEST(Model, MaterialAccessorThrowsForMissingMaterial) {
    Model m = make_simple_model();
    EXPECT_THROW(m.material(MaterialId{999}), SolverError);
}

TEST(Model, PropertyAccessorReturnsPShell) {
    Model m = make_simple_model();
    const Property& prop = m.property(PropertyId{1});
    const PShell& ps = std::get<PShell>(prop);
    EXPECT_NEAR(ps.t, 0.1, 1e-10);
}

TEST(Model, PropertyAccessorThrowsForMissingProperty) {
    Model m = make_simple_model();
    EXPECT_THROW(m.property(PropertyId{999}), SolverError);
}

TEST(Model, LoadsForSetReturnsMatchingLoads) {
    Model m = make_simple_model();
    auto loads = m.loads_for_set(LoadSetId{1});
    ASSERT_EQ(loads.size(), 1u);
    const ForceLoad& f = std::get<ForceLoad>(*loads[0]);
    EXPECT_EQ(f.node.value, 2);
}

TEST(Model, LoadsForSetReturnsEmptyForUnknownSet) {
    Model m = make_simple_model();
    auto loads = m.loads_for_set(LoadSetId{999});
    EXPECT_TRUE(loads.empty());
}

TEST(Model, SpcsForSetReturnsMatchingSpcs) {
    Model m = make_simple_model();
    auto spcs = m.spcs_for_set(SpcSetId{1});
    ASSERT_EQ(spcs.size(), 1u);
    EXPECT_EQ(spcs[0]->node.value, 1);
}

TEST(Model, SpcsForSetReturnsEmptyForUnknownSet) {
    Model m = make_simple_model();
    auto spcs = m.spcs_for_set(SpcSetId{999});
    EXPECT_TRUE(spcs.empty());
}
