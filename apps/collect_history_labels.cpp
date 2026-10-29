#include "common_includes.h"
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "online_gate.h"
#include "gate_trace.h"
#include "linux_aligned_file_reader.h"
#include <array>
#include <fstream>
#include <mutex>
struct HistoryRun { diskann::QueryStats stats{};std::array<uint64_t,10> ids{};std::array<float,10> ds{}; };
int main(int argc,char**argv) {
 if(argc!=7){std::cerr<<"index query gt frozen-LR threshold output\n";return 2;}
 float* queries=nullptr;uint32_t* truth=nullptr;float* td=nullptr;size_t nq,dim,adim,ng,dg;
 diskann::load_aligned_bin<float>(argv[2],queries,nq,dim,adim);diskann::load_truthset(argv[3],truth,td,ng,dg);
 if(nq!=ng || dg<10)throw std::runtime_error("truth shape");
 std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());diskann::PQFlashIndex<float,uint32_t> index(reader,diskann::Metric::L2);
 if(index.load(4,argv[1]))return 3;
 std::vector<uint32_t> nodes;index.cache_bfs_levels(10000,nodes);index.load_cache_list(nodes);
 diskann::OnlineGate model(argv[4],"",std::stof(argv[5]));std::mutex lock;
 std::ofstream out(argv[6]);out<<"query,node,prior_skipped,keep_hits,skip_hits,keep_reads,skip_reads";for(int j=0;j<44;++j)out<<",s"<<j;out<<"\n";
 #pragma omp parallel for schedule(dynamic,1) num_threads(4)
 for(size_t q=0;q<nq;++q) {
  mkl_set_num_threads_local(1);
  auto run=[&](const std::unordered_set<uint32_t>& mask,const std::unordered_set<uint32_t>& protect,std::vector<float>* audit){
   HistoryRun r;diskann::online_gate=&model;diskann::online_protect=&protect;diskann::gate_skip=&mask;diskann::gate_skip_entire_node=true;diskann::online_history_audit=audit;
   index.cached_beam_search(queries+q*adim,10,100,r.ids.data(),r.ds.data(),4,std::numeric_limits<uint32_t>::max(),false,&r.stats);
   diskann::online_gate=nullptr;diskann::online_protect=nullptr;diskann::gate_skip=nullptr;diskann::gate_skip_entire_node=false;diskann::online_history_audit=nullptr;return r;
  };
  auto hits=[&](const HistoryRun&r){unsigned n=0;for(auto id:r.ids)for(unsigned j=0;j<10;++j)if(id==truth[q*dg+j]){++n;break;}return n;};
  std::unordered_set<uint32_t> empty;std::vector<float> audit;auto base=run(empty,empty,&audit);std::ostringstream rows;
  std::unordered_set<uint32_t> unique;
  for(size_t j=0;j<audit.size();j+=46){uint32_t node=uint32_t(audit[j]);const float* state=audit.data()+j+1;if(state[10]!=0)continue;
   if(!unique.insert(node).second)throw std::runtime_error("Repeated candidate");
   bool skipped=audit[j+45]!=0;
   auto keep=skipped?run(empty,{node},nullptr):base;
   auto skip=skipped?base:run({node},empty,nullptr);
   rows<<q<<','<<node<<','<<skipped<<','<<hits(keep)<<','<<hits(skip)<<','<<keep.stats.n_ios<<','<<skip.stats.n_ios;
   for(int k=0;k<44;++k)rows<<','<<std::setprecision(9)<<state[k];rows<<'\n';
  }
  {std::lock_guard<std::mutex> guard(lock);out<<rows.str();out.flush();std::cout<<"query "<<q<<" done candidates "<<unique.size()<<std::endl;}
 }
 diskann::aligned_free(queries);delete[] truth;delete[] td;
}
