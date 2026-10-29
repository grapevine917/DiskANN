// Counterfactual opportunity estimator: no trained selection or production oracle.
#include "common_includes.h"
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "gate_trace.h"
#include "online_gate.h"
#include "linux_aligned_file_reader.h"
#include <array>
#include <fstream>
#include <mutex>
#include <set>
struct Run { diskann::GateTrace trace; diskann::QueryStats stats{}; std::array<uint64_t,10> ids{}; std::array<float,10> ds{}; };
int main(int argc,char**argv) {
 if(argc!=9) {std::cerr<<"index query gt model threshold output-prefix L cache\n";return 2;}
 float* queries=nullptr;uint32_t* truth=nullptr;float* td=nullptr;size_t nq,dim,adim,ng,dg;
 diskann::load_aligned_bin<float>(argv[2],queries,nq,dim,adim);
 diskann::load_truthset(argv[3],truth,td,ng,dg);
 if(nq!=ng || dg<10) throw std::runtime_error("truth shape");
 const bool features_only=std::getenv("CF_FEATURES_ONLY")!=nullptr;
 const unsigned L=std::stoul(argv[7]),cache=std::stoul(argv[8]);
 std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
 diskann::PQFlashIndex<float,uint32_t> index(reader,diskann::Metric::L2);
 if(index.load(4,argv[1]))return 3;
 std::vector<uint32_t> nodes;index.cache_bfs_levels(cache,nodes);index.load_cache_list(nodes);
 diskann::OnlineGate model(argv[4],"",std::stof(argv[5]));
 std::ofstream cand(std::string(argv[6])+"_candidates.csv"),joint(std::string(argv[6])+"_queries.csv");
 cand<<"query,node,cache_miss,fresh,admitted,in_result,singleton_recall_ok,singleton_exact,singleton_reads,singleton_delta\n";
 joint<<"query,reference,policy,baseline_reads,reads,baseline_expansions,expansions,baseline_hits,baseline_recall,recall,exact,mask_nodes,omitted_reads,new_reads\n";
 std::ofstream features(std::string(argv[6])+"_features.csv");
 features<<"query,node";for(int j=0;j<17;++j)features<<",s"<<j;features<<"\n";
 model.threshold=0;
 std::mutex lock;
 #pragma omp parallel for schedule(dynamic,1) num_threads(4)
 for(size_t q=0;q<nq;++q) {
  mkl_set_num_threads_local(1);
  auto run=[&](const std::unordered_set<uint32_t>&mask,bool pruning) {
   Run r;diskann::gate_trace=&r.trace;diskann::gate_skip=&mask;diskann::gate_skip_entire_node=true;
   diskann::online_gate=pruning?&model:nullptr;
   index.cached_beam_search(queries+q*adim,10,L,r.ids.data(),r.ds.data(),4,std::numeric_limits<uint32_t>::max(),false,&r.stats);
   diskann::gate_trace=nullptr;diskann::gate_skip=nullptr;diskann::gate_skip_entire_node=false;diskann::online_gate=nullptr;
   if(r.stats.n_ios!=std::count_if(r.trace.expansions.begin(),r.trace.expansions.end(),[](auto&e){return !e.cached;}))throw std::runtime_error("read/expansion accounting mismatch");
   return r;
  };
  auto recall=[&](const Run&r) {unsigned n=0;for(auto id:r.ids)for(unsigned j=0;j<10;++j)if(id==truth[q*dg+j]){++n;break;}return n;};
  std::unordered_set<uint32_t> empty;
  std::vector<float> audit;
  diskann::online_audit=&audit;
  Run base_audit=run(empty,true);
  diskann::online_audit=nullptr;
  std::ostringstream fs;
  for(size_t j=0;j<audit.size();j+=19) {fs<<q<<','<<uint32_t(audit[j]);for(int k=0;k<17;++k)fs<<','<<std::setprecision(9)<<audit[j+1+k];fs<<'\n';}
  if(features_only){std::lock_guard<std::mutex> guard(lock);features<<fs.str();continue;}
  std::ostringstream cs,js;
  for(bool pruning:{false}) {
   const Run base=run(empty,pruning); const unsigned quality=recall(base);
   auto emit=[&](const char*policy,const Run&r,size_t masksize) {
    std::set<uint32_t>b,a;for(auto&e:base.trace.expansions)if(!e.cached)b.insert(e.node);for(auto&e:r.trace.expansions)if(!e.cached)a.insert(e.node);
    unsigned omitted=0,added=0;for(auto id:b)omitted+=!a.count(id);for(auto id:a)added+=!b.count(id);
    js<<q<<','<<(pruning?"pruning":"original")<<','<<policy<<','<<base.stats.n_ios<<','<<r.stats.n_ios<<','<<base.stats.n_expanded<<','<<r.stats.n_expanded<<','<<base.stats.n_cache_hits<<','<<quality<<','<<recall(r)<<','<<(r.ids==base.ids && r.ds==base.ds)<<','<<masksize<<','<<omitted<<','<<added<<'\n';
   };
   emit("baseline",base,0);
   std::unordered_set<uint32_t> zero,singles;
   for(auto&e:base.trace.expansions) {
    bool final=std::find(base.ids.begin(),base.ids.end(),e.node)!=base.ids.end();
    if(!e.admitted && !final)zero.insert(e.node);
    // Exhaustive singleton removal, including cache hits and final result nodes.
    Run s=run({e.node},pruning);bool ok=recall(s)>=quality;
    if(ok && !e.cached && s.stats.n_ios<base.stats.n_ios) singles.insert(e.node);
    if(!pruning)cs<<q<<','<<e.node<<','<<!e.cached<<','<<e.fresh<<','<<e.admitted<<','<<final<<','<<ok<<','<<(s.ids==base.ids && s.ds==base.ds)<<','<<s.stats.n_ios<<','<<int(base.stats.n_ios)-int(s.stats.n_ios)<<'\n';
   }
  }
  std::lock_guard<std::mutex> guard(lock);cand<<cs.str();joint<<js.str();features<<fs.str();
 }
 diskann::aligned_free(queries);delete[] truth;delete[] td;
}
