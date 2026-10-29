#pragma once
#include "online_gate.h"
#include <mutex>
#include <unordered_set>
namespace diskann {
// Training/audit only: no future information enters the inference feature vector.
struct TerminationBoundary {
    uint32_t node, completed;
    std::array<float,17> state;
    float score;
};
struct TerminationTrace {
    std::vector<TerminationBoundary> boundaries;
    std::vector<uint32_t> scored_nodes;
};
inline thread_local const OnlineGate* online_termination=nullptr;
inline thread_local TerminationTrace* termination_trace=nullptr;
inline void termination_write(const std::string& path,uint32_t query,
                              const TerminationTrace& trace,const uint64_t* ids,size_t k) {
    std::unordered_set<uint32_t> remaining;
    for(size_t i=0;i<k;++i) remaining.insert(uint32_t(ids[i]));
    std::vector<float> rows;
    size_t completed=0;
    for(const auto& b:trace.boundaries) {
        while(completed<b.completed) remaining.erase(trace.scored_nodes.at(completed++));
        // 21 float32 columns: query, candidate node, state[17], safe-to-stop label, score.
        rows.push_back(float(query)); rows.push_back(float(b.node));
        rows.insert(rows.end(),b.state.begin(),b.state.end());
        rows.push_back(float(remaining.empty())); rows.push_back(b.score);
    }
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::ofstream f(path,std::ios::binary|std::ios::app);
    if(!f) throw std::runtime_error("Cannot open termination trace");
    f.write(reinterpret_cast<const char*>(rows.data()),rows.size()*sizeof(float));
    if(!f) throw std::runtime_error("Cannot write termination trace");
}
}
