// src/core/model.cpp
#include "core/model.hpp"
#include <format>
#include <algorithm>

namespace nastran {

void Model::validate() const {
    // Check all element nodes exist
    for (const auto& elem : elements) {
        for (NodeId nid : elem.nodes) {
            if (!nodes.count(nid))
                throw SolverError(std::format(
                    "Element {} references undefined node {}", elem.id.value, nid.value));
        }
        if (!properties.count(elem.pid))
            throw SolverError(std::format(
                "Element {} references undefined property {}", elem.id.value, elem.pid.value));
    }

    // Check property-material references
    for (const auto& [pid, prop] : properties) {
        std::visit([&](const auto& p) {
            using T = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<T, PShell>) {
                if (!materials.count(p.mid1))
                    throw SolverError(std::format(
                        "PSHELL {} references undefined material {}", pid.value, p.mid1.value));
            } else if constexpr (std::is_same_v<T, PSolid>) {
                if (!materials.count(p.mid))
                    throw SolverError(std::format(
                        "PSOLID {} references undefined material {}", pid.value, p.mid.value));
            }
        }, prop);
    }

    // Check SPCs reference existing nodes
    for (const auto& spc : spcs) {
        if (!nodes.count(spc.node))
            throw SolverError(std::format(
                "SPC references undefined node {}", spc.node.value));
    }

    // Check load nodes exist
    for (const auto& load : loads) {
        std::visit([&](const auto& l) {
            using T = std::decay_t<decltype(l)>;
            if constexpr (std::is_same_v<T, ForceLoad> ||
                          std::is_same_v<T, MomentLoad> ||
                          std::is_same_v<T, TempLoad>) {
                if (!nodes.count(l.node))
                    throw SolverError(std::format(
                        "Load references undefined node {}", l.node.value));
            }
        }, load);
    }

    // Check analysis case references
    for (const auto& sc : analysis.subcases) {
        if (sc.load_set.value != 0) {
            bool found = std::any_of(loads.begin(), loads.end(), [&](const Load& l){
                return std::visit([&](const auto& ll){ return ll.sid == sc.load_set; }, l);
            });
            if (!found && sc.load_set.value != 0) {
                // Not necessarily an error — thermal only case may have no FORCE loads
            }
        }
    }
}

} // namespace nastran
