#pragma once
// App-only passive capture. No core search or model code is changed.
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
namespace decision_trace {
inline std::string mode="search", prefix, ids_file;
inline bool enabled=true, overwrite=false;
inline bool active() { return mode=="decision_trace" && enabled; }
inline std::string quoted(const std::string &s) {
    std::string r="\""; for(unsigned char c:s) {
        if(c=='"'||c=='\\') {r+='\\';r+=c;}
        else if(c<32) { const char *h="0123456789abcdef";r+="\\u00";r+=h[c>>4];r+=h[c&15]; }
        else r+=c;
    } return r+'"';
}
struct Record {
    uint32_t query_id,node_id,decision_step,search_step;
    float state[17],score;
    uint8_t cache_hit,model_invoked,skip,reserved;
};
static_assert(sizeof(Record)==92,"decision trace wire format");
struct Writer {
    std::string base; std::vector<uint32_t> ids,ls; uint32_t n,d;
    template<class V> static void put(std::ofstream &f,const V &v) {f.write(reinterpret_cast<const char*>(&v),sizeof(v));}
    std::string records(uint32_t l) const {return base+".L"+std::to_string(l)+".records.bin";}
    Writer(const std::string &fallback,const float *q,uint32_t count,uint32_t dim,uint32_t stride,
           const std::vector<uint32_t>& lists):base(prefix.empty()?fallback+".decision_trace":prefix),ls(lists),n(count),d(dim) {
        ids.resize(n);for(uint32_t i=0;i<n;i++)ids[i]=i;
        if(!ids_file.empty()) {
            std::ifstream in(ids_file); if(!in)throw std::runtime_error("Cannot read decision_trace_query_ids");
            uint64_t x;size_t i=0;std::set<uint32_t> unique;
            while(in>>x){if(i>=n||x>UINT32_MAX||!unique.insert(uint32_t(x)).second)throw std::runtime_error("Invalid/duplicate trace query ID");ids[i++]=uint32_t(x);}
            if(i!=n||!in.eof())throw std::runtime_error("Trace query ID count/format mismatch");
        }
        std::vector<std::string> paths={base+".manifest.json",base+".query_vectors.fbin",base+".query_ids.u32"};
        std::set<uint32_t> seen;for(auto l:ls){if(!seen.insert(l).second)throw std::runtime_error("Duplicate L in trace");paths.push_back(records(l));}
        for(auto &p:paths)if(!overwrite&&std::filesystem::exists(p))throw std::runtime_error("Trace exists (use --decision_trace_overwrite true): "+p);
        auto parent=std::filesystem::path(base).parent_path();if(!parent.empty())std::filesystem::create_directories(parent);
        std::ofstream f(base+".query_vectors.fbin",std::ios::binary);put(f,n);put(f,d);
        for(uint32_t i=0;i<n;i++)f.write(reinterpret_cast<const char*>(q+size_t(i)*stride),d*sizeof(float));
        if(!f)throw std::runtime_error("Cannot write trace query vectors");
        std::ofstream g(base+".query_ids.u32",std::ios::binary);g.write(reinterpret_cast<const char*>(ids.data()),n*4);
        if(!g)throw std::runtime_error("Cannot write trace query IDs");
        manifest(false);
    }
    void manifest(bool complete) {
        std::ofstream f(base+".manifest.json");
        f<<"{\"schema\":\"DECTR001\",\"complete\":"<<(complete?"true":"false")<<",\"query_count\":"<<n<<",\"dimension\":"<<d
         <<",\"record_bytes\":92,\"score_semantics\":\"retain probability; lower means Skip\",\"search_step\":\"state_features[8] (native hop counter)\",\"decision_step\":\"zero-based candidate decision ordinal\",\"cache_policy\":\"unchanged native fixed node cache\",\"query_vectors\":"<<quoted(base+".query_vectors.fbin")<<",\"query_ids\":"<<quoted(base+".query_ids.u32")<<",\"records\":[";
        for(size_t i=0;i<ls.size();i++){if(i)f<<',';f<<"{\"L\":"<<ls[i]<<",\"path\":"<<quoted(records(ls[i]))<<'}';}
        f<<"]}\n";if(!f)throw std::runtime_error("Cannot write trace manifest");
    }
    void write(uint32_t l,const std::vector<std::vector<float>>&scores,const std::vector<std::vector<float>>&history) {
        std::ofstream f(records(l),std::ios::binary);f.write("DECTR001",8);put(f,n);uint32_t bytes=92;put(f,bytes);
        for(uint32_t i=0;i<n;i++){
            auto &a=scores[i];auto &b=history[i];
            if(a.size()%19||b.size()%46||a.size()/19!=b.size()/46)throw std::runtime_error("Trace audit lengths disagree");
            uint32_t count=a.size()/19;put(f,ids[i]);put(f,i);put(f,count);
            for(uint32_t j=0;j<count;j++){
                const float *s=a.data()+19*j,*h=b.data()+46*j;
                for(int k=0;k<18;k++)if(s[k]!=h[k])throw std::runtime_error("Trace score/decision alignment mismatch");
                Record r{};r.query_id=ids[i];r.node_id=uint32_t(s[0]);r.decision_step=j;
                std::copy(s+1,s+18,r.state);r.search_step=uint32_t(r.state[8]);r.score=s[18];
                r.cache_hit=r.state[10]!=0;r.model_invoked=1;r.skip=h[45]!=0;put(f,r);
            }
        }if(!f)throw std::runtime_error("Cannot write trace records");
    }
};
} // namespace decision_trace
