#include "common_includes.h"
#include <mkl.h>
#include "disk_utils.h"
#include "pq_flash_index.h"
#include "neighbor.h"
#include "ram_entry.h"
#include "linux_aligned_file_reader.h"
#include "gperftools/malloc_extension.h"
#include <array>
#include <fstream>
#include <chrono>
#include <numeric>
#include <mutex>
#include <sys/resource.h>
#include <time.h>
size_t heap_bytes(){size_t value=0;if(!MallocExtension::instance()->GetNumericProperty("generic.current_allocated_bytes",&value))throw std::runtime_error("TCMalloc accounting unavailable");return value;}
template<class T>size_t allocated(const std::vector<T>&v){return v.capacity()?MallocExtension::instance()->GetAllocatedSize(v.data()):0;}
struct NavGraph {std::vector<uint32_t> ids,degree,edges;uint32_t root=0,R=0;bool original_edges=false,csr=false;uint32_t csr_edge_count=0;};
struct NavScratch {std::vector<uint8_t> visited;diskann::NeighborPriorityQueue queue;std::vector<diskann::Neighbor> best;std::vector<uint32_t> seeds;};
struct NavStats {uint32_t expanded=0,distances=0;double us=0,cpu_us=0;};
std::vector<uint32_t> read_ids(const std::string&path){std::ifstream f(path,std::ios::binary);uint32_t n=0;f.read((char*)&n,4);std::vector<uint32_t>v(n);f.read((char*)v.data(),n*4);if(!f)throw std::runtime_error("IDs missing");return v;}
NavGraph read_graph(const std::string&prefix){NavGraph g;g.ids=read_ids(prefix+".ids");std::ifstream f(prefix+".edges",std::ios::binary);uint32_t n;f.read((char*)&n,4);f.read((char*)&g.R,4);f.read((char*)&g.root,4);g.csr=(g.R&0x80000000u)!=0;g.R&=0x7fffffffu;if(n!=g.ids.size()||g.R!=16||g.root>=n)throw std::runtime_error("Navigation shape");g.degree.resize(n);g.edges.resize(size_t(n)*g.R);f.read((char*)g.degree.data(),n*4);f.read((char*)g.edges.data(),n*g.R*4);if(!f)throw std::runtime_error("Navigation missing");if(g.csr){g.csr_edge_count=std::find(g.edges.begin(),g.edges.end(),UINT32_MAX)-g.edges.begin();if(g.degree[0]!=0 || !std::is_sorted(g.degree.begin(),g.degree.end()) || g.degree.back()>g.csr_edge_count)throw std::runtime_error("CSR offsets");}
else for(auto d:g.degree)if(d>g.R)throw std::runtime_error("Navigation degree");return g;}
NavStats navigate(diskann::PQFlashIndex<float,uint32_t>&index,const float*q,NavGraph&g,NavScratch&s,uint32_t budget,std::vector<uint32_t>*path=nullptr){
 auto start=std::chrono::steady_clock::now();struct timespec cpu_start{},cpu_end{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&cpu_start);NavStats stats;
 std::fill(s.visited.begin(),s.visited.end(),0);s.queue.clear();s.best.clear();s.seeds.clear();
 const uint32_t dimension=index.get_data_dim();
 auto distance=[&](uint32_t slot){const float*p=index.cached_coordinates(g.ids[slot]);if(!p)throw std::runtime_error("Uncached navigation vector");float sum=0;for(uint32_t j=0;j<dimension;++j){float d=q[j]-p[j];sum+=d*d;}++stats.distances;return sum;};
 s.visited[g.root]=1;s.queue.insert(diskann::Neighbor(g.root,distance(g.root)));
 while(s.queue.has_unexpanded_node() && stats.expanded<budget){auto node=s.queue.closest_unexpanded();++stats.expanded;if(path)path->push_back(node.id);
  s.best.push_back(node);std::sort(s.best.begin(),s.best.end());if(s.best.size()>4)s.best.pop_back();
  auto consider=[&](uint32_t next){if(next>=g.ids.size())throw std::runtime_error("Navigation edge");if(!s.visited[next]){s.visited[next]=1;s.queue.insert(diskann::Neighbor(next,distance(next)));}};
  if(g.original_edges){auto neighborhood=index.cached_neighbors(g.ids[node.id]);for(uint32_t j=0;j<neighborhood.first;++j){uint32_t next=index.cached_slot(neighborhood.second[j]);if(next!=UINT32_MAX)consider(next);}}
  else if(g.csr){uint32_t end=node.id+1<g.ids.size()?g.degree[node.id+1]:g.csr_edge_count;for(uint32_t j=g.degree[node.id];j<end;++j)consider(g.edges[j]);}
  else for(uint32_t j=0;j<g.degree[node.id];++j)consider(g.edges[size_t(node.id)*g.R+j]);
 }
 for(auto&b:s.best)s.seeds.push_back(g.ids[b.id]);
 clock_gettime(CLOCK_THREAD_CPUTIME_ID,&cpu_end);stats.cpu_us=(cpu_end.tv_sec-cpu_start.tv_sec)*1e6+(cpu_end.tv_nsec-cpu_start.tv_nsec)/1e3;stats.us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();return stats;
}
int main(int argc,char**argv){
 if(argc<4){std::cerr<<"probe index output | run index graph-or-B1 query gt output nav-budget repeat | profile index graph queries output nav-budget\n";return 2;}
 std::string mode=argv[1],prefix=argv[2];std::shared_ptr<AlignedFileReader>reader(new LinuxAlignedFileReader());diskann::PQFlashIndex<float,uint32_t>index(reader,diskann::Metric::L2);if(index.load(4,prefix.c_str()))return 3;
 if(mode=="probe"){
  std::vector<uint32_t>ids;index.cache_bfs_levels(10000,ids);size_t before=heap_bytes();index.load_cache_list(ids);size_t after=heap_bytes();
  std::ofstream f(std::string(argv[3])+".ids",std::ios::binary);uint32_t n=ids.size();f.write((char*)&n,4);f.write((char*)ids.data(),n*4);
  std::ofstream meta(std::string(argv[3])+".json");meta<<"{\"bfs_native_cache_bytes\":"<<after-before+index.cache_metadata_object_bytes()<<",\"nodes\":"<<n<<",\"max_degree\":"<<index.get_max_degree()<<",\"aligned_dim\":"<<index.aligned_dimension()<<",\"root_id\":"<<index.original_medoid()<<",\"common_heap_before_cache\":"<<before-allocated(ids)<<"}\n";return 0;
 }
 bool profile=mode=="profile";if((!profile&&argc!=9)||(profile&&argc!=7))throw std::runtime_error("Arguments");
 bool b1=std::string(argv[3])=="B1";NavGraph graph;std::vector<uint32_t> b1ids;
 if(b1)index.cache_bfs_levels(10000,b1ids);else graph=read_graph(argv[3]);
 if(!b1)graph.original_edges=std::getenv("RAM_USE_ORIGINAL_EDGES")!=nullptr;
 std::vector<uint32_t>&ids=b1?b1ids:graph.ids;
 // Warm reader registration before isolating cache allocations.
 std::vector<uint32_t>warm{index.original_medoid()};std::vector<float*>coord(1,nullptr);std::vector<uint32_t>neighbors(index.get_max_degree()+1);std::vector<std::pair<uint32_t,uint32_t*>>nbr{{0,neighbors.data()}};index.read_nodes(warm,coord,nbr);
 size_t before=heap_bytes();index.load_cache_list(ids);size_t cache_bytes=heap_bytes()-before+index.cache_metadata_object_bytes();
 if(b1)std::vector<uint32_t>().swap(b1ids);
 uint32_t queue_capacity=64,queue_storage=64;
 if(const char*v=std::getenv("RAM_NAV_QUEUE_CAPACITY"))queue_capacity=std::stoul(v);
 if(const char*v=std::getenv("RAM_NAV_QUEUE_STORAGE"))queue_storage=std::stoul(v);
 if(queue_capacity<4||queue_capacity>queue_storage||queue_storage>4096)throw std::runtime_error("RAM navigation queue capacity/storage");
 std::array<NavScratch,4>scratch;
 size_t scratch_before=heap_bytes();if(!b1)for(auto&s:scratch){s.visited.resize(graph.ids.size());s.queue.reserve(queue_storage);s.queue.reserve(queue_capacity);s.best.reserve(5);s.seeds.reserve(4);}size_t scratch_bytes=heap_bytes()-scratch_before+(b1?0:sizeof(scratch));
 size_t graph_bytes=b1?0:allocated(graph.ids)+allocated(graph.degree)+allocated(graph.edges)+sizeof(graph);
 size_t footprint=cache_bytes+graph_bytes+scratch_bytes;
 float*queries=nullptr;size_t nq,dim,adim;diskann::load_aligned_bin<float>(argv[4],queries,nq,dim,adim);
 uint32_t*truth=nullptr;float*gd=nullptr;size_t ng=0,dg=0;std::string output=profile?argv[5]:argv[6];uint32_t nav_budget=std::stoul(profile?argv[6]:argv[7]);
 if(!profile){diskann::load_truthset(argv[5],truth,gd,ng,dg);if(ng!=nq||dg<10)throw std::runtime_error("truth");}
 std::ofstream meta(output+".memory.json");meta<<"{\"native_cache_bytes\":"<<cache_bytes<<",\"cache_vector_allocated_bytes\":"<<MallocExtension::instance()->GetAllocatedSize(index.cached_coordinate_storage())<<",\"cache_original_adjacency_allocated_bytes\":"<<MallocExtension::instance()->GetAllocatedSize(index.cached_adjacency_storage())<<",\"navigation_graph_bytes\":"<<graph_bytes<<",\"navigation_thread_scratch_bytes\":"<<scratch_bytes<<",\"cache_and_navigation_bytes\":"<<footprint<<",\"navigation_queue_capacity\":"<<queue_capacity<<",\"navigation_queue_storage_capacity\":"<<queue_storage<<",\"nodes\":"<<(b1?10000:graph.ids.size())<<",\"aligned_dim\":"<<index.aligned_dimension()<<",\"max_degree\":"<<index.get_max_degree()<<",\"heap_with_input_queries\":"<<heap_bytes()<<"}\n";meta.close();
 if(profile){std::ofstream f(output+".profile",std::ios::binary);std::mutex mutex;
  #pragma omp parallel for schedule(dynamic,1) num_threads(4)
  for(size_t q=0;q<nq;++q){std::vector<uint32_t>path;path.reserve(nav_budget);auto&s=scratch[omp_get_thread_num()];navigate(index,queries+q*adim,graph,s,nav_budget,&path);uint32_t h[]={uint32_t(q),uint32_t(path.size())};std::lock_guard<std::mutex>guard(mutex);f.write((char*)h,8);f.write((char*)path.data(),path.size()*4);}diskann::aligned_free(queries);return 0;
 }
 uint32_t repetitions=std::stoul(argv[8]);std::ofstream f(output+".csv");f<<"repeat,query,hits,submitted_requests,expansions,cache_hits,nav_us,ssd_search_us,end_to_end_us,nav_expansions,nav_distances,nav_cpu_us\n";
 // Warm a fixed prefix without changing the static cache.
 for(size_t q=0;q<std::min(size_t(32),nq);++q){std::array<uint64_t,10>rr{};std::array<float,10>dd{};if(!b1 && nav_budget){navigate(index,queries+q*adim,graph,scratch[0],nav_budget);diskann::ram_entry_seeds=&scratch[0].seeds;}index.cached_beam_search(queries+q*adim,10,100,rr.data(),dd.data(),4,UINT32_MAX,false,nullptr);diskann::ram_entry_seeds=nullptr;}
 struct rusage usage{};getrusage(RUSAGE_SELF,&usage);std::ofstream warm_memory(output+".warm_memory.json");warm_memory<<"{\"heap_after_warmup\":"<<heap_bytes()<<",\"max_rss_kib\":"<<usage.ru_maxrss<<"}\n";warm_memory.close();
 std::ofstream timing(output+".timing.csv");timing<<"repeat,wall_seconds,qps\n";
 for(uint32_t rep=0;rep<repetitions;++rep){std::vector<std::string>lines(nq);auto wall=std::chrono::steady_clock::now();
  #pragma omp parallel for schedule(dynamic,1) num_threads(4)
  for(size_t q=0;q<nq;++q){mkl_set_num_threads_local(1);auto start=std::chrono::steady_clock::now();auto&s=scratch[omp_get_thread_num()];NavStats nav;if(!b1 && nav_budget){nav=navigate(index,queries+q*adim,graph,s,nav_budget);diskann::ram_entry_seeds=&s.seeds;}
   diskann::QueryStats stats{};std::array<uint64_t,10>result{};std::array<float,10>dist{};index.cached_beam_search(queries+q*adim,10,100,result.data(),dist.data(),4,UINT32_MAX,false,&stats);diskann::ram_entry_seeds=nullptr;
   double end_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();unsigned hits=0;for(auto id:result)for(unsigned j=0;j<10;++j)if(id==truth[q*dg+j]){++hits;break;}
   std::ostringstream row;row<<rep<<','<<q<<','<<hits<<','<<stats.n_ios<<','<<stats.n_expanded<<','<<stats.n_cache_hits<<','<<nav.us<<','<<stats.total_us<<','<<end_us<<','<<nav.expanded<<','<<nav.distances<<','<<nav.cpu_us<<'\n';lines[q]=row.str();
  }
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall).count();timing<<rep<<','<<seconds<<','<<nq/seconds<<'\n';for(auto&line:lines)f<<line;
 }
 diskann::aligned_free(queries);delete[]truth;delete[]gd;return 0;
}
