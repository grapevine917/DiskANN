#include "common_includes.h"
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "online_gate.h"
#include "linux_aligned_file_reader.h"
#include <array>
#include <chrono>
#include <fstream>
#include <sstream>
#include <numeric>
struct MemoResult {diskann::QueryStats stats{};std::array<uint64_t,10> ids{};std::array<float,10> ds{};uint32_t calls=0,reused=0,member=0,conflicts=0,hit=0,donor=0;double rho=0,total_us=0,lookup_us=0;};
struct MemoConfig {std::string key;int mode,rep;double rho;};
int main(int argc,char**argv){
 if(argc!=10){std::cerr<<"index bank-query model tau query gt configs output-prefix bank-file\n";return 2;}
 float *bank=nullptr,*query=nullptr;uint32_t*gt=nullptr;float*gd=nullptr;size_t nb,bd,ba,nq,dim,adim,ng,dg;
 diskann::load_aligned_bin<float>(argv[2],bank,nb,bd,ba);diskann::load_aligned_bin<float>(argv[5],query,nq,dim,adim);diskann::load_truthset(argv[6],gt,gd,ng,dg);
 if(bd!=dim || nq!=ng || dg<10 || nb!=1000)throw std::runtime_error("query shapes");
 diskann::OnlineGate model(argv[3],"",std::stof(argv[4]));
 std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());diskann::PQFlashIndex<float,uint32_t> index(reader,diskann::Metric::L2);
 if(index.load(4,argv[1]))return 3;std::vector<uint32_t> bfs;index.cache_bfs_levels(10000,bfs);index.load_cache_list(bfs);
 std::vector<std::vector<diskann::SkipMemoRecord>> entries(nb);std::vector<uint32_t> pool;
 auto search=[&](const float*q,MemoResult&r){
  diskann::online_gate=&model;diskann::online_miss_only=true;diskann::online_model_calls=&r.calls;
  index.cached_beam_search(q,10,100,r.ids.data(),r.ds.data(),4,std::numeric_limits<uint32_t>::max(),false,&r.stats);
  diskann::online_gate=nullptr;diskann::online_miss_only=false;diskann::online_model_calls=nullptr;
 };
 std::ifstream cached(argv[9],std::ios::binary);
 if(cached){
  char magic[8];uint32_t n,d,u;cached.read(magic,8);cached.read((char*)&n,4);cached.read((char*)&d,4);cached.read((char*)&u,4);
  if(!cached || std::string(magic,8)!="QSMEMO01" || n!=nb || d!=dim || u>1000000)throw std::runtime_error("cache header");
  pool.resize(u);cached.read((char*)pool.data(),u*4);
  for(auto&v:entries){uint32_t count;cached.read((char*)&count,4);if(!cached || count>1000000)throw std::runtime_error("cache count");v.resize(count);cached.read((char*)v.data(),count*sizeof(diskann::SkipMemoRecord));}
  if(!cached)throw std::runtime_error("truncated cache");
 }else{
  #pragma omp parallel for schedule(dynamic,1) num_threads(4)
  for(size_t q=0;q<nb;++q){mkl_set_num_threads_local(1);MemoResult r;diskann::online_skip_records=&entries[q];search(bank+q*ba,r);diskann::online_skip_records=nullptr;
   auto&v=entries[q];std::sort(v.begin(),v.end(),[](auto&a,auto&b){return a.node<b.node;});
   for(size_t j=1;j<v.size();++j)if(v[j-1].node==v[j].node)throw std::runtime_error("duplicate cache member");
  }
  for(auto&v:entries)for(auto&r:v)pool.push_back(r.node);std::sort(pool.begin(),pool.end());pool.erase(std::unique(pool.begin(),pool.end()),pool.end());pool.shrink_to_fit();
  for(auto&v:entries){for(auto&r:v)r.node=uint32_t(std::lower_bound(pool.begin(),pool.end(),r.node)-pool.begin());v.shrink_to_fit();}
  std::ofstream f(argv[9],std::ios::binary);f.write("QSMEMO01",8);uint32_t n=nb,d=dim,u=pool.size();f.write((char*)&n,4);f.write((char*)&d,4);f.write((char*)&u,4);f.write((char*)pool.data(),u*4);
  for(auto&v:entries){uint32_t count=v.size();f.write((char*)&count,4);f.write((char*)v.data(),v.size()*sizeof(diskann::SkipMemoRecord));}
 }
 if(!std::is_sorted(pool.begin(),pool.end()))throw std::runtime_error("pool ordering");
 uint64_t edges=0;for(auto&v:entries){edges+=v.size();for(size_t j=0;j<v.size();++j){if(v[j].node>=pool.size() || (j && v[j-1].node>=v[j].node))throw std::runtime_error("invalid shared references");}}
 std::vector<float> norm(nb);for(size_t i=0;i<nb;++i){double s=0;for(size_t j=0;j<dim;++j)s+=double(bank[i*ba+j])*bank[i*ba+j];norm[i]=float(s);}
 std::ofstream meta(std::string(argv[8])+".bank.json");meta<<"{\"cached_queries\":"<<nb<<",\"unique_nodes\":"<<pool.size()<<",\"query_node_references\":"<<edges<<",\"node_pool_bytes\":"<<pool.size()*4<<",\"query_reference_bytes\":"<<edges*4<<",\"state_guard_bytes\":"<<edges*8<<",\"query_vector_bytes\":"<<nb*ba*4<<",\"norm_bytes\":"<<nb*4<<",\"container_bytes\":"<<entries.size()*sizeof(entries[0])+sizeof(pool)+sizeof(entries)+sizeof(norm)<<",\"per_worker_scratch_bytes\":"<<nb*4<<",\"plain_node_id_lists_bytes\":"<<edges*4<<"}\n";meta.close();
 std::ifstream cfg(argv[7]);std::vector<MemoConfig> configs;MemoConfig c;while(cfg>>c.key>>c.mode>>c.rho>>c.rep)configs.push_back(c);
 if(configs.empty())throw std::runtime_error("empty configs");
 // Prime index/OS caches; no query-cache lookup or audit in warmup.
 #pragma omp parallel for schedule(dynamic,1) num_threads(4)
 for(size_t i=0;i<nq;++i){mkl_set_num_threads_local(1);MemoResult r;search(query+i*adim,r);}
 std::ofstream csv(std::string(argv[8])+".csv");csv<<"key,mode,rep,query,hits,reads,expansions,skips,model_calls,reused,membership_hits,conflicts,cache_hit,donor,rho,total_us,lookup_us\n";
 for(const auto&config:configs){
  std::vector<MemoResult> rows(nq);auto batch_start=std::chrono::steady_clock::now();
  #pragma omp parallel for schedule(dynamic,1) num_threads(4)
  for(size_t i=0;i<nq;++i){mkl_set_num_threads_local(1);auto&r=rows[i];diskann::SkipMemoView view;view.pool=&pool;view.guarded=config.mode==3 || config.mode==5;view.lookup_only=config.mode==1;view.audit=config.mode==4 || config.mode==5;
   auto begin=std::chrono::steady_clock::now();
   if(config.mode){
    std::vector<float> dot(nb);cblas_sgemv(CblasRowMajor,CblasNoTrans,nb,dim,-2,bank,ba,query+i*adim,1,0,dot.data(),1);
    size_t best=0;float dist=dot[0]+norm[0];for(size_t j=1;j<nb;++j)if(dot[j]+norm[j]<dist){best=j;dist=dot[j]+norm[j];}
    double qn=0,bn=0,d2=0;for(size_t j=0;j<dim;++j){double q=query[i*adim+j],b=bank[best*ba+j];qn+=q*q;bn+=b*b;d2+=(q-b)*(q-b);}
    r.donor=best;r.rho=d2/std::max({qn,bn,1e-12});r.hit=r.rho<=config.rho;
    view.records=&entries[best];if(r.hit)diskann::online_skip_memo=&view;
    r.lookup_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
   }
   search(query+i*adim,r);diskann::online_skip_memo=nullptr;
   r.total_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();r.reused=view.reused;r.member=view.membership_hits;r.conflicts=view.conflicts;
  }
  double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-batch_start).count();
  uint64_t hits=0,reads=0,calls=0,reused=0;double us=0;
  std::ofstream ids(std::string(argv[8])+"_"+config.key+"_r"+std::to_string(config.rep)+".ids",std::ios::binary),ds(std::string(argv[8])+"_"+config.key+"_r"+std::to_string(config.rep)+".dists",std::ios::binary);
  for(size_t i=0;i<nq;++i){auto&r=rows[i];unsigned h=0;for(auto id:r.ids)for(size_t j=0;j<10;++j)if(id==gt[i*dg+j]){++h;break;}hits+=h;reads+=r.stats.n_ios;calls+=r.calls;reused+=r.reused;us+=r.total_us;
   csv<<config.key<<','<<config.mode<<','<<config.rep<<','<<i<<','<<h<<','<<r.stats.n_ios<<','<<r.stats.n_expanded<<','<<r.stats.n_gate_skipped<<','<<r.calls<<','<<r.reused<<','<<r.member<<','<<r.conflicts<<','<<r.hit<<','<<r.donor<<','<<std::setprecision(12)<<r.rho<<','<<r.total_us<<','<<r.lookup_us<<'\n';ids.write((char*)r.ids.data(),80);ds.write((char*)r.ds.data(),40);
  }csv.flush();std::cout<<"MEMO "<<config.key<<" rep "<<config.rep<<" recall "<<double(hits)/nq*10<<" reads "<<double(reads)/nq<<" calls "<<double(calls)/nq<<" reused "<<double(reused)/nq<<" us "<<us/nq<<" qps "<<nq/wall<<std::endl;
 }
 diskann::aligned_free(bank);diskann::aligned_free(query);delete[]gt;delete[]gd;
}
