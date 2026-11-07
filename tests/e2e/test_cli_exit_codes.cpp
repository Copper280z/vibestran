// Black-box checks of the exit code and the F06 a run leaves behind.
//
// Meson sets VIBESTRAN (the executable).
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <regex>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <system_error>

namespace fs = std::filesystem;

namespace {

constexpr const char *kRod =
    "SOL 101\nCEND\nSPC = 1\nLOAD = 1\nDISPLACEMENT = ALL\nBEGIN BULK\nMAT1,1,1.+7,,0.3\n"
    "PROD,1,1,2.,PROD_J\nGRID,1,,0.,0.,0.\nGRID,2,,10.,0.,0.\nCROD,1,1,1,2\nSPC1,1,123,1\nSPC1,1,23,2\n"
    "FORCE,1,2,0,100.,1.,0.,0.\nEXTRA\nENDDATA\n";

std::string deck(const std::string &j, const std::string &extra) {
    std::string text = kRod;
    text.replace(text.find("PROD_J"), 6, j);
    text.replace(text.find("EXTRA"), 5, extra);
    return text;
}

std::string read(const fs::path &path) {
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

class CliExit : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::getenv("VIBESTRAN"))
            FAIL() << "VIBESTRAN is not set; run this test through meson test";
        dir_ = fs::temp_directory_path() / ("vibestran_cli_" + std::to_string(::getpid()));
        fs::create_directories(dir_);
    }
    ~CliExit() override {
        std::error_code ignored;
        fs::remove_all(dir_, ignored);
    }

    int run(const std::string &text, const std::string &arguments = "", const std::string &env = "") {
        const fs::path input = dir_ / "model.bdf";
        std::ofstream(input) << text;
        const std::string command = env + " '" + std::getenv("VIBESTRAN") + "' " + arguments + " '" +
                                    input.string() + "' '" + (dir_ / "model.f06").string() + "' > /dev/null 2>&1";
        const int status = std::system(command.c_str());
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
    std::string f06() const { return read(dir_ / "model.f06"); }
    bool op2_written() const { return fs::exists(dir_ / "model.op2"); }

    fs::path dir_;
};

} // namespace

TEST_F(CliExit, AHealthyModelExitsZeroWithoutAFatalMessage) {
    // No J, so no torsion to leave free.
    EXPECT_EQ(run(deck("0.", "")), 0);
    EXPECT_EQ(f06().find("FATAL"), std::string::npos);
    EXPECT_TRUE(op2_written());
}

TEST_F(CliExit, ASingularModelWritesItsResultsUnderAFatalMessageAndExitsNonzero) {
    EXPECT_EQ(run(deck("1.", "MOMENT,1,2,0,5.,1.,0.,0.")), 5);
    const std::string text = f06();
    const auto fatal = text.find("*** FATAL: Subcase 1:");
    ASSERT_NE(fatal, std::string::npos) << text.substr(0, 400);
    EXPECT_LT(fatal, text.find(" RUN"));
    EXPECT_NE(text.find("D I S P L A C E M E N T"), std::string::npos);
    EXPECT_TRUE(op2_written());
}

TEST_F(CliExit, BailoutExitsZeroWithWarnings) {
    EXPECT_EQ(run(deck("1.", "PARAM,BAILOUT,-1\nMOMENT,1,2,0,5.,1.,0.,0.")), 0);
    const std::string text = f06();
    EXPECT_EQ(text.find("FATAL"), std::string::npos);
    EXPECT_NE(text.find("*** WARNING:"), std::string::npos);
}

TEST_F(CliExit, TheDefaultBackendIsTheLibraryAndTheOtherStaysAvailable) {
    EXPECT_EQ(run(deck("0.", "")), 0);
    EXPECT_NE(f06().find("LDL^T library"), std::string::npos);
    EXPECT_EQ(run(deck("0.", ""), "--backend=cpu-eigen"), 0);
    EXPECT_EQ(f06().find("LDL^T library"), std::string::npos);
}

TEST_F(CliExit, OmpNumThreadsOverridesTheCoreCount) {
    EXPECT_EQ(run(deck("0.", ""), "", "OMP_NUM_THREADS=3"), 0);
    EXPECT_TRUE(std::regex_search(f06(), std::regex(R"(CPU threads[ .]+3\b)")));
}
