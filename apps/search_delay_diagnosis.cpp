// Opt-in delayed expansion and single-node protection on frozen classifiers.
#include "common_includes.h"
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "online_gate.h"
#include "linux_aligned_file_reader.h"
#include <array>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_set>
struct DelayRun {diskann::QueryStats stats{};std::array<uint64_t,10> ids{};std::array<float,10> distances{};std::vector<float> audit;};
int main(int argc,char**argv){
 if(argc!=9){std::cerr<<"index query gt model tau1 tau2 tau3 output-prefix\n";return 2;}
 float* queries=nullptr;uint32_t* truth=nullptr;float* gd=nullptr;size_t nq,dim,adim,ng,dg;
 diskann::load_aligned_bin<float>(argv[2],queries,nq,dim,adim);diskann::load_truthset(argv[3],truth,gd,ng,dg);
 if(nq!=ng || dg<10)throw std::runtime_error("truth shape");
 std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());diskann::PQFlashIndex<float,uint32_t> index(reader,diskann::Metric::L2);
 if(index.load(4,argv[1]))return 3;std::vector<uint32_t> cache;index.cache_bfs_levels(10000,cache);index.load_cache_list(cache);
 std::vector<std::unique_ptr<diskann::OnlineGate>> models;for(int j=0;j<3;++j)models.emplace_back(new diskann::OnlineGate(argv[4],"",std::stof(argv[5+j])));
 std::ofstream results(std::string(argv[8])+"_queries.csv"),events(std::string(argv[8])+"_skips.csv"),causal(std::string(argv[8])+"_protection.csv");
 results<<"query,threshold_index,mode,baseline_hits,permanent_hits,hits,baseline_reads,reads,expansions,skipped,deferred,rescued,exact_baseline,latency_us,protected_nodes\n";
 events<<"query,threshold_index,node,retain_score,completed_expansions,candidate_rank,pq_over_worst,pq_over_best,beam_round,baseline_hits,permanent_hits,in_baseline_result,in_gt10\n";
 causal<<"query,threshold_index,node,completed_expansions,candidate_rank,in_baseline_result,in_gt10,baseline_hits,permanent_hits,protected_hits,permanent_reads,protected_reads\n";
 const uint32_t priority=std::getenv("DEFER_PRIORITY")?std::stoul(std::getenv("DEFER_PRIORITY")):0;
 if(priority>2)throw std::runtime_error("Invalid deferred priority");
 const bool skip_protection=std::getenv("NO_PROTECTION")!=nullptr;
 const bool only_protection=std::getenv("ONLY_PROTECTION")!=nullptr;
 std::unordered_set<uint32_t> query_filter;
 if(const char* path=std::getenv("QUERY_FILTER")){std::ifstream f(path);if(!f)throw std::runtime_error("Query filter missing");uint32_t id;while(f>>id)query_filter.insert(id);}
 std::mutex mutex;
 #pragma omp parallel for schedule(dynamic,1) num_threads(4)
 for(size_t q=0;q<nq;++q){
  if(!query_filter.empty() && !query_filter.count(q))continue;
  mkl_set_num_threads_local(1);
  auto run=[&](const diskann::OnlineGate* model,uint32_t limit,const std::unordered_set<uint32_t>*protect,bool audit){
   DelayRun r;diskann::online_gate=model;diskann::online_defer_limit=limit;diskann::online_defer_priority=priority;diskann::online_protect=protect;diskann::online_audit=audit?&r.audit:nullptr;
   index.cached_beam_search(queries+q*adim,10,100,r.ids.data(),r.distances.data(),4,std::numeric_limits<uint32_t>::max(),false,&r.stats);
   diskann::online_gate=nullptr;diskann::online_defer_limit=0;diskann::online_defer_priority=0;diskann::online_protect=nullptr;diskann::online_audit=nullptr;return r;
  };
  auto hits=[&](const DelayRun&r){unsigned count=0;for(auto id:r.ids)for(unsigned j=0;j<10;++j)if(id==truth[q*dg+j]){++count;break;}return count;};
  DelayRun base=run(nullptr,0,nullptr,false);unsigned bh=hits(base);std::ostringstream rs,es,cs;
  for(int t=0;t<3;++t){auto model=models[t].get();DelayRun permanent=run(model,0,nullptr,true);unsigned ph=hits(permanent);
   auto emit=[&](const std::string&mode,const DelayRun&r,size_t protected_nodes=0){rs<<q<<','<<t<<','<<mode<<','<<bh<<','<<ph<<','<<hits(r)<<','<<base.stats.n_ios<<','<<r.stats.n_ios<<','<<r.stats.n_expanded<<','<<r.stats.n_gate_skipped<<','<<r.stats.n_deferred<<','<<r.stats.n_rescued<<','<<(r.ids==base.ids && r.distances==base.distances)<<','<<r.stats.total_us<<','<<protected_nodes<<'\n';};
   emit("permanent",permanent);
   if(!only_protection)for(uint32_t limit:{1u,4u,8u,16u,32u,std::numeric_limits<uint32_t>::max()}){
    DelayRun d=run(model,limit,nullptr,true);
    if(d.audit!=permanent.audit)throw std::runtime_error("Deferred phase altered primary decisions");
    if(d.stats.n_rescued>limit || d.stats.n_rescued>d.stats.n_deferred)throw std::runtime_error("Rescue budget accounting");
    emit(limit==std::numeric_limits<uint32_t>::max()?"defer_all":"defer_"+std::to_string(limit),d);
   }
   std::unordered_set<uint32_t> useful_protect;
   for(size_t j=0;j<permanent.audit.size();j+=19){auto&a=permanent.audit;uint32_t node=uint32_t(a[j]);if(a[j+18]>=model->threshold)continue;
    bool result=std::find(base.ids.begin(),base.ids.end(),node)!=base.ids.end();bool gt=false;for(unsigned z=0;z<10;++z)gt|=truth[q*dg+z]==node;
    es<<q<<','<<t<<','<<node<<','<<std::setprecision(9)<<a[j+18]<<','<<a[j+8]<<','<<a[j+4]<<','<<a[j+15]<<','<<a[j+14]<<','<<a[j+9]<<','<<bh<<','<<ph<<','<<result<<','<<gt<<'\n';
    if(ph<bh && !skip_protection){std::unordered_set<uint32_t> protect{node};DelayRun r=run(model,0,&protect,false);if(hits(r)>ph)useful_protect.insert(node);cs<<q<<','<<t<<','<<node<<','<<a[j+8]<<','<<a[j+4]<<','<<result<<','<<gt<<','<<bh<<','<<ph<<','<<hits(r)<<','<<permanent.stats.n_ios<<','<<r.stats.n_ios<<'\n';}
   }
   if(ph<bh && !skip_protection){
    DelayRun joint=run(model,0,&useful_protect,false);emit("protect_causal_joint",joint,useful_protect.size());
   }
  }
  std::lock_guard<std::mutex> guard(mutex);results<<rs.str();events<<es.str();causal<<cs.str();
 }
 diskann::aligned_free(queries);delete[]truth;delete[]gd;
}
