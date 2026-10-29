#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace diskann {
struct GateExpansion { uint32_t node, degree, fresh, admitted; float exact; uint32_t cached; };
struct GateEdge { uint32_t parent, child; float pq; uint32_t flags; };
struct GateEdgeV2 {
 uint32_t query_id, step_id, beam_step, current_node_id, neighbor_id;
 float current_distance_exact, current_distance_pq, neighbor_distance_pq, neighbor_distance_exact;
 float current_best_distance, current_best_exact_distance;
 uint32_t neighbor_rank, neighbor_position, queue_size, visited_count, flags;
 uint32_t neighbor_was_inserted, neighbor_was_expanded, neighbor_led_to_improvement;
 uint32_t neighbor_in_final_topk, neighbor_improved_best_pq, neighbor_improved_best_exact;
};
static_assert(sizeof(GateExpansion)==24 && sizeof(GateEdge)==16 && sizeof(GateEdgeV2)==88,"trace layout");
inline bool gate_v2_requested() { const char* v=std::getenv("DISKANN_TRACE_VERSION"); return v && std::string(v)=="2"; }
struct GateTrace {
 std::vector<GateExpansion> expansions;
 std::vector<GateEdge> edges;
 std::vector<GateEdgeV2> edges_v2;
 uint32_t visited=0, query_id=0, beam_step=0;
 bool v2=false;
 float current_pq=std::numeric_limits<float>::quiet_NaN();
 float best_exact=std::numeric_limits<float>::infinity();
};
inline thread_local GateTrace* gate_trace=nullptr;
inline thread_local bool gate_skip_entire_node=false;
inline thread_local const std::unordered_set<uint32_t>* gate_skip=nullptr;
inline bool gate_should_skip(uint32_t id) {return gate_skip && gate_skip->count(id);}
inline void gate_begin(uint32_t id,uint32_t degree,float exact,bool cached,
                       float pq=std::numeric_limits<float>::quiet_NaN(),uint32_t beam=0) {
 if(!gate_trace) return;
 gate_trace->expansions.push_back({id,degree,0,0,exact,uint32_t(cached)});
 gate_trace->current_pq=pq; gate_trace->beam_step=beam;
 gate_trace->best_exact=std::min(gate_trace->best_exact,exact);
}
inline std::vector<uint32_t> gate_neighbor_ranks(const uint32_t* ids,const float* ds,size_t n) {
 if(!gate_trace || !gate_trace->v2) return {};
 std::vector<uint32_t> order(n),rank(n); std::iota(order.begin(),order.end(),0);
 std::sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b) {
  return ds[a]<ds[b] || (ds[a]==ds[b] && (ids[a]<ids[b] || (ids[a]==ids[b] && a<b)));
 });
 for(uint32_t i=0;i<n;++i) rank[order[i]]=i;
 return rank;
}
inline void gate_edge(uint32_t child,float pq,bool fresh,bool admitted,uint32_t rank=0,
                      uint32_t position=0,uint32_t queue_size=0,uint32_t visited_before=0,
                      float best_pq=std::numeric_limits<float>::quiet_NaN()) {
 if(!gate_trace) return;
 auto &t=*gate_trace; auto &x=t.expansions.back(); x.fresh+=fresh; x.admitted+=admitted;
 uint32_t flags=uint32_t(fresh)|(uint32_t(admitted)<<1);
 if(!t.v2) {t.edges.push_back({x.node,child,pq,flags}); return;}
 t.edges_v2.push_back({t.query_id,uint32_t(t.expansions.size()-1),t.beam_step,x.node,child,
  x.exact,t.current_pq,pq,std::numeric_limits<float>::quiet_NaN(),best_pq,t.best_exact,
  rank,position,queue_size,visited_before,flags,uint32_t(admitted),0,0,0,
  uint32_t(admitted && pq<best_pq),0});
}
inline void gate_finalize(GateTrace& t,const uint64_t* result_ids,size_t k) {
 if(!t.v2) return;
 std::unordered_set<uint32_t> expanded,improved,final_ids,led;
 std::unordered_map<uint32_t,uint32_t> parent;
 float best=std::numeric_limits<float>::infinity();
 for(size_t i=0;i<t.expansions.size();++i) {
  const auto &x=t.expansions[i]; expanded.insert(x.node);
  if(i>0 && x.exact<best) improved.insert(x.node);
  best=std::min(best,x.exact);
 }
 for(const auto &x:t.edges_v2) if(x.flags&1) parent.emplace(x.neighbor_id,x.current_node_id);
 for(auto node:improved) {
  while(led.insert(node).second) {
   auto it=parent.find(node); if(it==parent.end()) break; node=it->second;
  }
 }
 for(size_t i=0;i<k;++i) final_ids.insert(uint32_t(result_ids[i]));
 for(auto &x:t.edges_v2) {
  x.neighbor_was_expanded=uint32_t(expanded.count(x.neighbor_id)!=0);
  x.neighbor_in_final_topk=uint32_t(final_ids.count(x.neighbor_id)!=0);
  x.neighbor_improved_best_exact=uint32_t((x.flags&1) && improved.count(x.neighbor_id));
  x.neighbor_led_to_improvement=uint32_t((x.flags&1) && led.count(x.neighbor_id));
 }
}
inline void gate_write(const std::string& prefix,uint32_t q,uint32_t L,const GateTrace& t) {
 static std::mutex m; std::lock_guard<std::mutex> lock(m);
 std::ofstream f(prefix+"_"+std::to_string(L)+".trace",std::ios::binary|std::ios::app);
 if(!f) throw std::runtime_error("Cannot open trace");
 uint32_t h[]={q,L,t.visited,uint32_t(t.expansions.size()),uint32_t(t.v2?t.edges_v2.size():t.edges.size())};
 f.write(reinterpret_cast<const char*>(h),sizeof(h));
 f.write(reinterpret_cast<const char*>(t.expansions.data()),t.expansions.size()*sizeof(GateExpansion));
 if(t.v2) f.write(reinterpret_cast<const char*>(t.edges_v2.data()),t.edges_v2.size()*sizeof(GateEdgeV2));
 else f.write(reinterpret_cast<const char*>(t.edges.data()),t.edges.size()*sizeof(GateEdge));
 if(!f) throw std::runtime_error("Trace write failed");
}
inline std::vector<std::unordered_set<uint32_t>> gate_load_masks(const std::string& prefix,uint32_t L,size_t n) {
 std::vector<std::unordered_set<uint32_t>> masks(n);
 if(prefix.empty()) return masks;
 std::ifstream f(prefix+"_"+std::to_string(L)+".mask");
 if(!f) throw std::runtime_error("Cannot open skip mask");
 uint32_t q,id; while(f>>q>>id) {if(q>=n) throw std::runtime_error("Bad mask query"); masks[q].insert(id);}
 return masks;
}
}
