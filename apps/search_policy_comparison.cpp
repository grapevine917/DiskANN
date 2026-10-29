#include "common_includes.h"
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "online_gate.h"
#include "linux_aligned_file_reader.h"
#include <array>
#include <chrono>
#include <fstream>
#include <map>
struct PolicyBank {std::vector<uint32_t>pool;std::vector<std::vector<diskann::SkipMemoRecord>> records;};
struct Policy {std::string key;int kind,rep;const diskann::OnlineGate*first=nullptr,*second=nullptr;const diskann::OnlineNodeMarks*marks=nullptr;diskann::OnlineHeuristic rule{0,-1};const PolicyBank*bank=nullptr;double rho=0;bool guard=false;};
struct PolicyResult {diskann::QueryStats stats{};std::array<uint64_t,10>ids{};std::array<float,10>ds{};uint32_t calls=0,reused=0,cache_hit=0;double us=0,lookup_us=0;};
int main(int argc,char**argv){
 if(argc!=7){std::cerr<<"index bank-query query gt configs output-prefix\n";return 2;}
 float *bankq=nullptr,*query=nullptr;uint32_t*truth=nullptr;float*gd=nullptr;size_t nb,bd,ba,nq,dim,adim,ng,dg;
 diskann::load_aligned_bin<float>(argv[2],bankq,nb,bd,ba);diskann::load_aligned_bin<float>(argv[3],query,nq,dim,adim);diskann::load_truthset(argv[4],truth,gd,ng,dg);
 if(nb!=1000 || bd!=dim || nq!=ng || dg<10)throw std::runtime_error("query shape");
 std::shared_ptr<AlignedFileReader>reader(new LinuxAlignedFileReader());diskann::PQFlashIndex<float,uint32_t>index(reader,diskann::Metric::L2);
 if(index.load(4,argv[1]))return 3;std::vector<uint32_t>bfs;index.cache_bfs_levels(10000,bfs);index.load_cache_list(bfs);
 std::map<std::string,std::unique_ptr<diskann::OnlineGate>>models;std::map<std::string,std::unique_ptr<diskann::OnlineNodeMarks>>marks;std::map<std::string,PolicyBank>banks;
 auto gate=[&](const std::string&path,double tau)->const diskann::OnlineGate*{if(path=="-")return nullptr;std::string key=path+"_"+std::to_string(tau);if(!models.count(key))models[key].reset(new diskann::OnlineGate(path,"",float(tau)));return models[key].get();};
 std::ifstream input(argv[5]);std::vector<Policy>policies;std::string key,path,second,mark,bankpath;int kind,rank,progress,guard,rep;double tau,tau2,rho;
 while(input>>key>>kind>>path>>tau>>second>>tau2>>mark>>rank>>progress>>bankpath>>rho>>guard>>rep){Policy p;p.key=key;p.kind=kind;p.rep=rep;p.first=gate(path,tau);p.second=gate(second,tau2);p.rule={rank,progress};p.rho=rho;p.guard=guard;
  if(mark!="-"){if(!marks.count(mark))marks[mark].reset(new diskann::OnlineNodeMarks(mark));p.marks=marks[mark].get();}
  if(bankpath!="-"){
   if(!banks.count(bankpath)){auto&b=banks[bankpath];std::ifstream f(bankpath,std::ios::binary);char magic[8];uint32_t n,d,u;f.read(magic,8);f.read((char*)&n,4);f.read((char*)&d,4);f.read((char*)&u,4);if(!f || std::string(magic,8)!="QSMEMO01" || n!=nb || d!=dim || u>1000000)throw std::runtime_error("bank header");b.pool.resize(u);f.read((char*)b.pool.data(),u*4);b.records.resize(n);for(auto&v:b.records){uint32_t count;f.read((char*)&count,4);if(!f || count>1000000)throw std::runtime_error("bank count");v.resize(count);f.read((char*)v.data(),count*sizeof(diskann::SkipMemoRecord));}if(!f)throw std::runtime_error("bank truncated");}
   p.bank=&banks.at(bankpath);
  }policies.push_back(p);
 }
 if(policies.empty())throw std::runtime_error("no policies");
 std::vector<float>norm(nb);for(size_t i=0;i<nb;++i){double s=0;for(size_t j=0;j<dim;++j)s+=double(bankq[i*ba+j])*bankq[i*ba+j];norm[i]=float(s);}
 auto search=[&](const Policy&p,size_t row,PolicyResult&r){diskann::online_gate=p.first;diskann::online_joint_gate=p.second;diskann::online_heuristic=p.kind==4?&p.rule:nullptr;diskann::online_node_marks=p.marks;diskann::online_miss_only=true;diskann::online_model_calls=&r.calls;
  index.cached_beam_search(query+row*adim,10,100,r.ids.data(),r.ds.data(),4,std::numeric_limits<uint32_t>::max(),false,&r.stats);
  diskann::online_gate=nullptr;diskann::online_joint_gate=nullptr;diskann::online_heuristic=nullptr;diskann::online_node_marks=nullptr;diskann::online_miss_only=false;diskann::online_model_calls=nullptr;
 };
 Policy baseline;baseline.kind=0;
 #pragma omp parallel for schedule(dynamic,1) num_threads(4)
 for(size_t i=0;i<nq;++i){mkl_set_num_threads_local(1);PolicyResult r;search(baseline,i,r);}
 std::ofstream out(std::string(argv[6])+".csv");out<<"key,rep,query,hits,reads,expansions,evaluated,skips,calls,reused,cache_hit,total_us,lookup_us\n";
 for(const auto&p:policies){std::vector<PolicyResult>rows(nq);auto batch=std::chrono::steady_clock::now();
  #pragma omp parallel for schedule(dynamic,1) num_threads(4)
  for(size_t i=0;i<nq;++i){mkl_set_num_threads_local(1);auto&r=rows[i];diskann::SkipMemoView view;auto start=std::chrono::steady_clock::now();
   if(p.bank){std::vector<float>dot(nb);cblas_sgemv(CblasRowMajor,CblasNoTrans,nb,dim,-2,bankq,ba,query+i*adim,1,0,dot.data(),1);size_t best=0;float d=dot[0]+norm[0];for(size_t j=1;j<nb;++j)if(dot[j]+norm[j]<d){best=j;d=dot[j]+norm[j];}double qn=0,bn=0,d2=0;for(size_t j=0;j<dim;++j){double q=query[i*adim+j],b=bankq[best*ba+j];qn+=q*q;bn+=b*b;d2+=(q-b)*(q-b);}r.cache_hit=d2/std::max({qn,bn,1e-12})<=p.rho;view.pool=&p.bank->pool;view.records=&p.bank->records[best];view.guarded=p.guard;if(r.cache_hit)diskann::online_skip_memo=&view;r.lookup_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();}
   search(p,i,r);diskann::online_skip_memo=nullptr;r.reused=view.reused;r.us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
   if(p.kind==0){if(r.stats.n_gate_evaluated || r.calls)throw std::runtime_error("baseline gated");}
   else if(r.stats.n_expanded!=r.stats.n_gate_evaluated-r.stats.n_gate_skipped)throw std::runtime_error("gate accounting");
  }
  double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-batch).count();std::ofstream ids(std::string(argv[6])+"_"+p.key+"_r"+std::to_string(p.rep)+".ids",std::ios::binary),ds(std::string(argv[6])+"_"+p.key+"_r"+std::to_string(p.rep)+".dists",std::ios::binary);double us=0;uint64_t hits=0,reads=0,calls=0;
  for(size_t i=0;i<nq;++i){auto&r=rows[i];unsigned h=0;for(auto id:r.ids)for(size_t j=0;j<10;++j)if(id==truth[i*dg+j]){++h;break;}hits+=h;reads+=r.stats.n_ios;calls+=r.calls;us+=r.us;out<<p.key<<','<<p.rep<<','<<i<<','<<h<<','<<r.stats.n_ios<<','<<r.stats.n_expanded<<','<<r.stats.n_gate_evaluated<<','<<r.stats.n_gate_skipped<<','<<r.calls<<','<<r.reused<<','<<r.cache_hit<<','<<std::setprecision(12)<<r.us<<','<<r.lookup_us<<'\n';ids.write((char*)r.ids.data(),80);ds.write((char*)r.ds.data(),40);}out.flush();std::cout<<"POLICY "<<p.key<<" rep "<<p.rep<<" recall "<<double(hits)/nq*10<<" reads "<<double(reads)/nq<<" calls "<<double(calls)/nq<<" us "<<us/nq<<" qps "<<nq/wall<<std::endl;
 }
 diskann::aligned_free(bankq);diskann::aligned_free(query);delete[]truth;delete[]gd;
}
