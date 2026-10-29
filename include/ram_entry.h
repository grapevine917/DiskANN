#pragma once
#include <cstdint>
#include <vector>
namespace diskann {
// Opt-in RAM-navigation handoff. Null preserves the original medoid entry.
inline thread_local const std::vector<uint32_t>* ram_entry_seeds=nullptr;
}
