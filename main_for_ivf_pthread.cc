#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include "pq_fastscan_ivf_pthread.h" 
#include <unistd.h>
#include <limits.h>
#include <sys/stat.h>
#include <algorithm>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <queue> 

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("Failed to open data file: " + data_path);
    }
    uint32_t n_32, d_32;
    fin.read(reinterpret_cast<char*>(&n_32), sizeof(n_32));
    fin.read(reinterpret_cast<char*>(&d_32), sizeof(d_32));
    if (!fin || fin.gcount() != sizeof(d_32)) {
         fin.close();
         throw std::runtime_error("Failed to read dimensions from: " + data_path);
    }
    if (n_32 == 0 || d_32 == 0 || n_32 > 20000000 || d_32 > 20000) {
         std::cerr << "Invalid dimensions read from: " << data_path << std::endl;
         std::cerr << "n: " << n_32 << ", d: " << d_32 << std::endl;
         fin.close();
         throw std::runtime_error("Invalid dimensions in file: " + data_path);
    }
    n = static_cast<size_t>(n_32);
    d = static_cast<size_t>(d_32);

    T* data = nullptr;
    size_t num_elements = n * d;
    if (num_elements == 0) {
        std::cerr << "Warning: Requesting 0 elements to load." << std::endl;
        return nullptr;
    }
    try {
        data = new T[num_elements];
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed for " << n << "x" << d << " elements: " << e.what() << std::endl;
        fin.close();
        throw;
    }

    int sz = sizeof(T);
    std::cerr << "Attempting to read " << n << " vectors of dimension " << d << " (element size: " << sz << ")" << std::endl;

    fin.seekg(0, std::ios::end);
    long long file_size = fin.tellg();
    fin.seekg(8, std::ios::beg);
    long long expected_data_size = static_cast<long long>(num_elements) * sz;
    long long expected_total_size = 8 + expected_data_size;

    if (file_size < expected_total_size) {
        std::cerr << "Error: File size " << file_size << " is too small for " << n << "x" << d << " elements. Expected at least " << expected_total_size << " bytes." << std::endl;
         delete[] data;
         fin.close();
         throw std::runtime_error("File size mismatch: " + data_path);
    }
     if (file_size > expected_total_size) {
         std::cerr << "Warning: File size " << file_size << " is larger than expected (" << expected_total_size << " bytes). Reading only the expected amount." << std::endl;
     }

    size_t bytes_to_read = num_elements * sz;
    char* buffer = reinterpret_cast<char*>(data);
    fin.read(buffer, bytes_to_read);
    if (fin.gcount() != static_cast<std::streamsize>(bytes_to_read)) {
         std::cerr << "Error reading data block from " << data_path << ". Read " << fin.gcount() << " bytes, expected " << bytes_to_read << std::endl;
         delete[] data;
         fin.close();
         throw std::runtime_error("Incomplete read from file: " + data_path);
    }

    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

//SearchResult结构体
struct SearchResult
{
    float recall;
    int64_t latency;
};

void ensure_files_directory() {
    struct stat st = {0};
    if (stat("files", &st) == -1) {
        if (mkdir("files", 0755) != 0) {
            perror("Error creating files directory");
        } else {
             std::cerr << "Created 'files' directory." << std::endl;
        }
    } else if (!S_ISDIR(st.st_mode)) {
             std::cerr << "'files' exists but is not a directory!" << std::endl;
    }
}

inline float exact_l2_distance_sq(const float* vec1, const float* vec2, size_t dim) {
    float dist_sq = 0;
#ifdef __ARM_NEON__
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t d = 0;
    for (; d + 3 < dim; d += 4) {
        float32x4_t v1 = vld1q_f32(vec1 + d);
        float32x4_t v2 = vld1q_f32(vec2 + d);
        float32x4_t diff = vsubq_f32(v1, v2);
        sum_vec = vfmaq_f32(sum_vec, diff, diff); 
    }
    float32x2_t sum_pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
    dist_sq = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
    for (; d < dim; ++d) {
        float diff = vec1[d] - vec2[d];
        dist_sq += diff * diff;
    }
#else
    for (size_t d = 0; d < dim; ++d) {
        float diff = vec1[d] - vec2[d];
        dist_sq += diff * diff;
    }
#endif
    return dist_sq;
}

int main(int argc, char *argv[])
{
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
       std::cerr << "Current working dir: " << cwd << std::endl;
    } else {
       perror("getcwd() error");
    }

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    size_t query_dim_check = 0, base_dim_check = 0;

    std::string data_path = "/anndata/"; 

    std::unique_ptr<float[]> test_query_ptr;
    std::unique_ptr<int[]> test_gt_ptr;
    std::unique_ptr<float[]> base_ptr; 

    try {
        float* raw_test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, query_dim_check);
        test_query_ptr.reset(raw_test_query);
        vecdim = query_dim_check;

        size_t gt_n_check = 0;
        int* raw_test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", gt_n_check, test_gt_d);
        test_gt_ptr.reset(raw_test_gt);
        if (gt_n_check != test_number) {
             std::cerr << "Warning: Query count (" << test_number << ") and ground truth count (" << gt_n_check << ") mismatch. Using minimum." << std::endl;
             test_number = std::min(test_number, gt_n_check);
        }

        float* raw_base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, base_dim_check);
        base_ptr.reset(raw_base); 
        if (query_dim_check != base_dim_check) {
            std::cerr << "Error: Query dimension (" << query_dim_check
                      << ") does not match base dimension (" << base_dim_check << ")!" << std::endl;
            return 1;
        }
        vecdim = query_dim_check;

        if (vecdim == 0 || base_number == 0 || test_number == 0) {
             std::cerr << "Error: Zero dimension or zero vectors loaded." << std::endl;
             return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Data loading error: " << e.what() << std::endl;
        return 1;
    }

    const float* test_query = test_query_ptr.get();
    const int* test_gt = test_gt_ptr.get();
    const float* base = base_ptr.get();

    size_t num_queries_to_test = std::min((size_t)2000, test_number);
     if (num_queries_to_test == 0) {
         std::cerr << "No queries to test." << std::endl; 
         return 0;
     } 

    // IVF-PQ参数配置
    const size_t k = 10;         
    const size_t k_rerank = 350;   
    size_t num_subvectors = 8;   
    size_t num_clusters = 256;    // IVF聚类数
    size_t nprobe = 10;           // 搜索时检查的聚类数
    
    std::cerr << "IVF-PQ Configuration: k=" << k << ", k_rerank=" << k_rerank 
              << ", M=" << num_subvectors << ", clusters=" << num_clusters 
              << ", nprobe=" << nprobe << std::endl;

    std::cerr << "Initializing IVF-PQ Index..." << std::endl;
    if (vecdim % num_subvectors != 0) {
         std::cerr << "Error: Dimension " << vecdim << " is not divisible by num_subvectors " << num_subvectors << std::endl;
         return 1;
    }

    std::unique_ptr<IVFPQIndex> ivf_pq_index;

    try {
        // 创建IVF-PQ索引
        ivf_pq_index.reset(new IVFPQIndex(vecdim, num_clusters, num_subvectors));

        size_t train_size = std::min(base_number, (size_t)100000); 
        std::cerr << "Training IVF-PQ index on " << train_size << " vectors..." << std::endl;
        
        // 训练和构建索引
        ivf_pq_index->train_and_build(base, train_size);

        std::cerr << "IVF-PQ Index initialization complete." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "IVF-PQ Index Initialization Error: " << e.what() << std::endl;
        return 1;
    }

    std::vector<SearchResult> results(num_queries_to_test);

    std::cerr << "Starting query test for " << num_queries_to_test << " queries using IVF-PQ + Re-ranking..." << std::endl;

    const unsigned long Converter = 1000 * 1000; 

    for(size_t i = 0; i < num_queries_to_test; ++i) {
        const float* current_query = test_query + i * vecdim;

        struct timeval val_start, val_end;
        gettimeofday(&val_start, NULL); 

        // 使用IVF-PQ搜索
        auto initial_res_queue = ivf_pq_search(*ivf_pq_index, current_query, k_rerank, nprobe);

        std::vector<std::pair<float, uint32_t>> candidates;
        candidates.reserve(initial_res_queue.size()); 
        while (!initial_res_queue.empty()) {
            candidates.push_back(initial_res_queue.top());
            initial_res_queue.pop();
        }
        
        // 精确重排序
        std::priority_queue<std::pair<float, uint32_t>> final_results_pq;

        for (const auto& candidate_pair : candidates) {
            uint32_t candidate_id = candidate_pair.second;
            if (candidate_id >= base_number) {
                 std::cerr << "Warning: Invalid candidate ID " << candidate_id << " encountered for query " << i << std::endl;
                 continue;
            }
            const float* candidate_vector = base + (size_t)candidate_id * vecdim; 

            float exact_dist_sq = exact_l2_distance_sq(current_query, candidate_vector, vecdim);

            if (final_results_pq.size() < k) {
                final_results_pq.push({exact_dist_sq, candidate_id});
            } else if (exact_dist_sq < final_results_pq.top().first) { 
                final_results_pq.pop();
                final_results_pq.push({exact_dist_sq, candidate_id});
            }
        }

        gettimeofday(&val_end, NULL); 
        int64_t diff = (val_end.tv_sec * Converter + val_end.tv_usec) - (val_start.tv_sec * Converter + val_start.tv_usec);

        // 计算召回率
        std::set<uint32_t> gtset;
        size_t gt_start_index = i * test_gt_d;
        size_t num_gt_to_consider = std::min(k, test_gt_d);

        if (gt_start_index + num_gt_to_consider > test_number * test_gt_d) {
             num_gt_to_consider = (test_number * test_gt_d > gt_start_index) ? (test_number * test_gt_d - gt_start_index) : 0;
        }

        for(size_t j = 0; j < num_gt_to_consider; ++j){
            int t = test_gt[gt_start_index + j];
            if (t >= 0) { 
                 gtset.insert(static_cast<uint32_t>(t));
            }
        }
        if (test_gt_d == 0 && num_queries_to_test > 0 && i==0) {
             std::cerr << "Warning: Ground truth dimension is 0." << std::endl;
        }

        size_t acc = 0;
        std::vector<uint32_t> final_result_indices;
        final_result_indices.reserve(final_results_pq.size());
        while (!final_results_pq.empty()) {
            final_result_indices.push_back(final_results_pq.top().second); 
            final_results_pq.pop();
        }
        if (!gtset.empty()) {
            for(uint32_t found_idx : final_result_indices) {
                if(gtset.count(found_idx)){
                    ++acc;
                }
            }
        }

        float recall = 0.0f;
        if (k > 0) {
             recall = static_cast<float>(acc) / k;
        }
        results[i] = {recall, diff};
    } 
    
    std::cerr << "Query test finished." << std::endl;

    float avg_recall = 0;
    double avg_latency = 0;
    if (num_queries_to_test > 0) {
        for(size_t i = 0; i < num_queries_to_test; ++i) {
            avg_recall += results[i].recall;
            avg_latency += results[i].latency;
        }
        avg_recall /= num_queries_to_test;
        avg_latency /= num_queries_to_test;
    }

    std::cout << "average recall: "<< std::fixed << std::setprecision(6) << avg_recall <<"\n";
    std::cout << "average latency (us): "<< std::fixed << std::setprecision(2) << avg_latency <<"\n";

    return 0;
}
