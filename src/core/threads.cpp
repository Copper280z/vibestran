// src/core/threads.cpp
#include "core/threads.hpp"
#include <algorithm>
#include <thread>
#if defined(__linux__)
#include <fstream>
#include <sched.h>
#include <set>
#include <string>
#include <utility>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace vibestran {

int physical_core_count() {
  const int logical = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
#if defined(__linux__)
  cpu_set_t allowed;
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
    return logical;
  std::set<std::pair<int, int>> cores;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &allowed))
      continue;
    const std::string topology = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    std::ifstream core_file(topology + "core_id"), package_file(topology + "physical_package_id");
    int core = 0, package = 0;
    if (!(core_file >> core) || !(package_file >> package))
      return std::max(1, CPU_COUNT(&allowed));
    cores.emplace(package, core);
  }
  return std::max(1, static_cast<int>(cores.size()));
#elif defined(__APPLE__)
  int cores = 0;
  size_t size = sizeof(cores);
  if (sysctlbyname("hw.physicalcpu", &cores, &size, nullptr, 0) == 0 && cores > 0)
    return cores;
  return logical;
#else
  return logical;
#endif
}

} // namespace vibestran
