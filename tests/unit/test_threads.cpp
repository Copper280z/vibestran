// tests/unit/test_threads.cpp
#include "core/threads.hpp"
#include <gtest/gtest.h>
#include <thread>
#if defined(__linux__)
#include <fstream>
#include <sched.h>
#include <set>
#include <sstream>
#include <string>
#endif

using namespace vibestran;

TEST(PhysicalCores, IsNeverMoreThanTheLogicalCpus) {
    const int cores = physical_core_count();
    EXPECT_GE(cores, 1);
    EXPECT_LE(static_cast<unsigned>(cores), std::max(1u, std::thread::hardware_concurrency()));
}

#if defined(__linux__)
namespace {

std::set<int> cpu_list(const std::string &path) {
    std::ifstream in(path);
    std::string text;
    in >> text;
    std::set<int> cpus;
    std::stringstream parts(text);
    std::string part;
    while (std::getline(parts, part, ',')) {
        const auto dash = part.find('-');
        const int first = std::stoi(part.substr(0, dash));
        const int last = dash == std::string::npos ? first : std::stoi(part.substr(dash + 1));
        for (int cpu = first; cpu <= last; ++cpu)
            cpus.insert(cpu);
    }
    return cpus;
}

/// Runs the count under an affinity mask and restores the original.
int count_on(const std::set<int> &cpus) {
    cpu_set_t original, restricted;
    EXPECT_EQ(sched_getaffinity(0, sizeof(original), &original), 0);
    CPU_ZERO(&restricted);
    for (int cpu : cpus)
        CPU_SET(cpu, &restricted);
    EXPECT_EQ(sched_setaffinity(0, sizeof(restricted), &restricted), 0);
    const int cores = physical_core_count();
    EXPECT_EQ(sched_setaffinity(0, sizeof(original), &original), 0);
    return cores;
}

} // namespace

TEST(PhysicalCores, HyperthreadSiblingsCountOnceUnderAnAffinityMask) {
    // What taskset does: only the cores the process may use count.
    cpu_set_t allowed;
    ASSERT_EQ(sched_getaffinity(0, sizeof(allowed), &allowed), 0);
    int first = 0;
    while (first < CPU_SETSIZE && !CPU_ISSET(first, &allowed))
        ++first;
    const auto siblings =
        cpu_list("/sys/devices/system/cpu/cpu" + std::to_string(first) + "/topology/thread_siblings_list");
    if (siblings.empty())
        GTEST_SKIP() << "this system does not show its CPU topology";
    std::set<int> usable;
    for (int cpu : siblings)
        if (CPU_ISSET(cpu, &allowed))
            usable.insert(cpu);
    EXPECT_EQ(count_on(usable), 1);

    int other = first + 1;
    while (other < CPU_SETSIZE && (!CPU_ISSET(other, &allowed) || siblings.contains(other)))
        ++other;
    if (other == CPU_SETSIZE)
        GTEST_SKIP() << "only one core is available";
    usable.insert(other);
    EXPECT_EQ(count_on(usable), 2);
}
#endif
