#include <vector>
#include <queue>
#include <utility>
#include <cstdint>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <pthread.h>
#include <thread>
#include <functional>
#include <future>
#include <unordered_map>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

// 优化的线程池实现
class ThreadPool {
private:
    std::vector<pthread_t> threads;
    std::vector<bool> thread_busy;
    size_t num_threads;
    bool shutdown;
    pthread_mutex_t queue_mutex;
    pthread_cond_t condition;
    std::queue<std::function<void()>> tasks;

public:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) 
        : num_threads(num_threads), shutdown(false) {
        threads.resize(num_threads);
        thread_busy.resize(num_threads, false);
        pthread_mutex_init(&queue_mutex, nullptr);
        pthread_cond_init(&condition, nullptr);
        
        for (size_t i = 0; i < num_threads; ++i) {
            pthread_create(&threads[i], nullptr, worker_thread, this);
        }
    }

    ~ThreadPool() {
        {
            pthread_mutex_lock(&queue_mutex);
            shutdown = true;
            pthread_mutex_unlock(&queue_mutex);
        }
        pthread_cond_broadcast(&condition);
        
        for (pthread_t& thread : threads) {
            pthread_join(thread, nullptr);
        }
        
        pthread_mutex_destroy(&queue_mutex);
        pthread_cond_destroy(&condition);
    }

    template<typename F>
    void enqueue(F&& f) {
        pthread_mutex_lock(&queue_mutex);
        tasks.emplace(std::forward<F>(f));
        pthread_mutex_unlock(&queue_mutex);
        pthread_cond_signal(&condition);
    }

    void wait_all() {
        while (true) {
            pthread_mutex_lock(&queue_mutex);
            bool all_done = tasks.empty();
            for (bool busy : thread_busy) {
                if (busy) {
                    all_done = false;
                    break;
                }
            }
            pthread_mutex_unlock(&queue_mutex);
            
            if (all_done) break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    size_t get_num_threads() const { return num_threads; }

private:
    static void* worker_thread(void* arg) {
        ThreadPool* pool = static_cast<ThreadPool*>(arg);
        return pool->worker_loop();
    }

    void* worker_loop() {
        size_t thread_id = 0;
        for (size_t i = 0; i < threads.size(); ++i) {
            if (pthread_equal(threads[i], pthread_self())) {
                thread_id = i;
                break;
            }
        }

        while (true) {
            std::function<void()> task;
            
            pthread_mutex_lock(&queue_mutex);
            while (tasks.empty() && !shutdown) {
                pthread_cond_wait(&condition, &queue_mutex);
            }
            
            if (shutdown && tasks.empty()) {
                pthread_mutex_unlock(&queue_mutex);
                break;
            }
            
            if (!tasks.empty()) {
                task = std::move(tasks.front());
                tasks.pop();
                thread_busy[thread_id] = true;
            }
            pthread_mutex_unlock(&queue_mutex);
            
            if (task) {
                task();
                pthread_mutex_lock(&queue_mutex);
                thread_busy[thread_id] = false;
                pthread_mutex_unlock(&queue_mutex);
            }
        }
        return nullptr;
    }
};

// 倒排列表项 - 优化内存布局
struct InvertedListItem {
    uint32_t vector_id;
    std::vector<uint8_t> pq_codes;
    
    InvertedListItem(uint32_t id, const std::vector<uint8_t>& codes) 
        : vector_id(id), pq_codes(codes) {}
};

using InvertedList = std::vector<InvertedListItem>;

// 优化的乘积量化器
class ProductQuantizer {
private:
    size_t m_dim;
    size_t m_num_subvectors;
    size_t m_subvec_dim;
    size_t m_ks;
    std::vector<float> m_codebooks;
    static std::unique_ptr<ThreadPool> thread_pool;
    static pthread_mutex_t print_mutex;
    static bool initialized;

    static void initialize_pool() {
        if (!initialized) {
            thread_pool.reset(new ThreadPool());  // 替换 make_unique
            initialized = true;
        }
    }

    // 优化的SIMD点积计算
    inline float simd_dot_product(const float* a, const float* b, size_t dim) const {
        float result = 0.0f;
        
#ifdef __ARM_NEON__
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        size_t d = 0;
        for (; d + 3 < dim; d += 4) {
            float32x4_t a_vec = vld1q_f32(a + d);
            float32x4_t b_vec = vld1q_f32(b + d);
            sum_vec = vfmaq_f32(sum_vec, a_vec, b_vec);
        }
        
        float32x2_t sum_pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
        result = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
        
        for (; d < dim; d++) {
            result += a[d] * b[d];
        }
#else
        for (size_t d = 0; d < dim; d++) {
            result += a[d] * b[d];
        }
#endif
        return result;
    }

    // 优化的SIMD平方距离计算
    inline float simd_squared_distance(const float* a, const float* b, size_t dim) const {
        float result = 0.0f;
        
#ifdef __ARM_NEON__
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        size_t d = 0;
        for (; d + 3 < dim; d += 4) {
            float32x4_t a_vec = vld1q_f32(a + d);
            float32x4_t b_vec = vld1q_f32(b + d);
            float32x4_t diff_vec = vsubq_f32(a_vec, b_vec);
            sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
        }
        
        float32x2_t sum_pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
        result = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
        
        for (; d < dim; d++) {
            float diff = a[d] - b[d];
            result += diff * diff;
        }
#else
        for (size_t d = 0; d < dim; d++) {
            float diff = a[d] - b[d];
            result += diff * diff;
        }
#endif
        return result;
    }

public:
    ProductQuantizer(size_t dim, size_t num_subvectors, size_t ks = 256)
        : m_dim(dim), m_num_subvectors(num_subvectors), m_ks(ks) {
        initialize_pool();
        
        if (dim == 0 || num_subvectors == 0 || ks == 0) {
             throw std::invalid_argument("Dimensions, subvectors, and ks must be non-zero.");
        }
        if (dim % num_subvectors != 0) {
            throw std::invalid_argument("Dimension must be divisible by num_subvectors.");
        }
        m_subvec_dim = dim / num_subvectors;
        try {
            m_codebooks.resize(m_num_subvectors * m_ks * m_subvec_dim);
        } catch (const std::bad_alloc& e) {
             std::cerr << "Failed to allocate memory for codebooks: " << e.what() << std::endl;
             throw;
        }
    }

    // 改进的K-means++初始化训练
    void train(const float* data, size_t n, int max_iter = 30) {
         if (n == 0) {
             std::cerr << "Warning: Trying to train PQ on 0 data points." << std::endl;
             return;
        }
        
        pthread_mutex_lock(&print_mutex);
        std::cerr << "开始优化PQ训练，数据量: " << n << ", 子向量数: " << m_num_subvectors << ", 维度: " << m_dim << std::endl;
        pthread_mutex_unlock(&print_mutex);
        
        for (size_t subvec_idx = 0; subvec_idx < m_num_subvectors; subvec_idx++) {
            thread_pool->enqueue([this, data, n, max_iter, subvec_idx]() {
                this->train_subvector_optimized(data, n, max_iter, subvec_idx);
            });
        }
        
        thread_pool->wait_all();
    }

    void encode(const float* vec, uint8_t* code) const {
        for (size_t subvec_idx = 0; subvec_idx < m_num_subvectors; subvec_idx++) {
            size_t subvec_offset = subvec_idx * m_subvec_dim;
            const float* current_codebook = m_codebooks.data() + subvec_idx * m_ks * m_subvec_dim;
            const float* current_subvector = vec + subvec_offset;

            float min_dist_sq = std::numeric_limits<float>::max();
            uint8_t best_centroid = 0;

            for (size_t k = 0; k < m_ks; k++) {
                const float* centroid = current_codebook + k * m_subvec_dim;
                float dist_sq = simd_squared_distance(current_subvector, centroid, m_subvec_dim);

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    best_centroid = static_cast<uint8_t>(k);
                }
            }
            code[subvec_idx] = best_centroid;
        }
    }

    // 优化的距离表计算 - 使用L2距离而不是内积
    void compute_distance_table(const float* query, float* table) const {
        for (size_t subvec_idx = 0; subvec_idx < m_num_subvectors; subvec_idx++) {
            thread_pool->enqueue([this, query, table, subvec_idx]() {
                this->compute_distance_table_subvector(query, table, subvec_idx);
            });
        }
        thread_pool->wait_all();
    }

    // 保持兼容性的内积表计算
    void compute_ip_table(const float* query, float* table) const {
        compute_distance_table(query, table);
    }

    size_t get_dim() const { return m_dim; }
    size_t get_num_subvectors() const { return m_num_subvectors; }
    size_t get_subvec_dim() const { return m_subvec_dim; }
    size_t get_ks() const { return m_ks; }
    const std::vector<float>& get_codebooks() const { return m_codebooks; }

private:
    // 使用K-means++改进初始化的训练
    void train_subvector_optimized(const float* data, size_t n, int max_iter, size_t subvec_idx) {
        size_t subvec_offset = subvec_idx * m_subvec_dim;
        float* current_codebook = m_codebooks.data() + subvec_idx * m_ks * m_subvec_dim;
        std::vector<float> subvec_data(n * m_subvec_dim);
        
        if (subvec_data.empty() && (n * m_subvec_dim > 0)) {
            pthread_mutex_lock(&print_mutex);
            std::cerr << "Error: Failed to allocate memory for subvector data in thread for subvec " << subvec_idx << std::endl;
            pthread_mutex_unlock(&print_mutex);
            return;
        }

        // 提取子向量数据
        for (size_t i = 0; i < n; i++) {
            std::memcpy(subvec_data.data() + i * m_subvec_dim,
                       data + i * m_dim + subvec_offset,
                       m_subvec_dim * sizeof(float));
        }

        // K-means++初始化
        std::random_device rd;
        std::mt19937 gen(rd() + subvec_idx);
        std::uniform_int_distribution<> dis(0, static_cast<int>(n - 1));
        
        // 选择第一个中心点
        size_t first_idx = dis(gen);
        std::memcpy(current_codebook, 
                   subvec_data.data() + first_idx * m_subvec_dim,
                   m_subvec_dim * sizeof(float));
        
        // K-means++选择剩余中心点
        for (size_t k = 1; k < m_ks && k < n; k++) {
            std::vector<float> distances(n);
            float total_dist = 0;
            
            for (size_t i = 0; i < n; i++) {
                float min_dist = std::numeric_limits<float>::max();
                const float* point = subvec_data.data() + i * m_subvec_dim;
                
                for (size_t j = 0; j < k; j++) {
                    const float* center = current_codebook + j * m_subvec_dim;
                    float dist = simd_squared_distance(point, center, m_subvec_dim);
                    min_dist = std::min(min_dist, dist);
                }
                distances[i] = min_dist;
                total_dist += min_dist;
            }
            
            if (total_dist > 0) {
                std::uniform_real_distribution<float> prob_dis(0, total_dist);
                float target = prob_dis(gen);
                float cumsum = 0;
                size_t selected = 0;
                
                for (size_t i = 0; i < n; i++) {
                    cumsum += distances[i];
                    if (cumsum >= target) {
                        selected = i;
                        break;
                    }
                }
                
                std::memcpy(current_codebook + k * m_subvec_dim,
                           subvec_data.data() + selected * m_subvec_dim,
                           m_subvec_dim * sizeof(float));
            } else {
                // 随机选择
                size_t idx = dis(gen);
                std::memcpy(current_codebook + k * m_subvec_dim,
                           subvec_data.data() + idx * m_subvec_dim,
                           m_subvec_dim * sizeof(float));
            }
        }

        // 如果数据点少于聚类数，随机填充剩余中心
        for (size_t k = std::min(m_ks, n); k < m_ks; k++) {
            size_t idx = dis(gen);
            std::memcpy(current_codebook + k * m_subvec_dim,
                       subvec_data.data() + idx * m_subvec_dim,
                       m_subvec_dim * sizeof(float));
        }

        // K-means迭代优化
        std::vector<size_t> assignments(n);
        std::vector<size_t> counts(m_ks);
        std::vector<float> new_centroids(m_ks * m_subvec_dim);
        
        float prev_inertia = std::numeric_limits<float>::max();
        int early_stop_count = 0;

        for (int iter = 0; iter < max_iter; iter++) {
            std::fill(counts.begin(), counts.end(), 0);
            std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
            float current_inertia = 0;

            // 分配步骤
            for (size_t i = 0; i < n; i++) {
                float min_dist_sq = std::numeric_limits<float>::max();
                size_t best_centroid = 0;
                const float* current_point = subvec_data.data() + i * m_subvec_dim;

                for (size_t k = 0; k < m_ks; k++) {
                    const float* centroid = current_codebook + k * m_subvec_dim;
                    float dist_sq = simd_squared_distance(current_point, centroid, m_subvec_dim);
                    
                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        best_centroid = k;
                    }
                }
                
                assignments[i] = best_centroid;
                counts[best_centroid]++;
                current_inertia += min_dist_sq;
                
                float* target_centroid_sum = new_centroids.data() + best_centroid * m_subvec_dim;
                for (size_t d = 0; d < m_subvec_dim; d++) {
                    target_centroid_sum[d] += current_point[d];
                }
            }

            // 更新步骤
            bool changed = false;
            for (size_t k = 0; k < m_ks; k++) {
                 float* target_centroid = current_codebook + k * m_subvec_dim;
                if (counts[k] > 0) {
                    float* source_sum = new_centroids.data() + k * m_subvec_dim;
                    float inv_count = 1.0f / counts[k];
                    for (size_t d = 0; d < m_subvec_dim; d++) {
                        float new_val = source_sum[d] * inv_count;
                        if (std::abs(target_centroid[d] - new_val) > 1e-6f) {
                            target_centroid[d] = new_val;
                            changed = true;
                        }
                    }
                } else {
                     // 重新初始化空聚类
                     size_t idx = dis(gen);
                     std::memcpy(target_centroid, subvec_data.data() + idx * m_subvec_dim, m_subvec_dim * sizeof(float));
                     changed = true;
                }
            }
            
            // 早停机制
            if (std::abs(prev_inertia - current_inertia) < 1e-6f * prev_inertia) {
                early_stop_count++;
                if (early_stop_count >= 3) break;
            } else {
                early_stop_count = 0;
            }
            prev_inertia = current_inertia;
            
             if (!changed && iter > 0) break;
        }

        pthread_mutex_lock(&print_mutex);
        std::cerr << "优化子空间 " << subvec_idx << " 码本训练完成" << std::endl;
        pthread_mutex_unlock(&print_mutex);
    }

    // 优化的距离表计算 - 使用L2距离
    void compute_distance_table_subvector(const float* query, float* table, size_t subvec_idx) const {
        size_t subvec_offset = subvec_idx * m_subvec_dim;
        const float* current_codebook = m_codebooks.data() + subvec_idx * m_ks * m_subvec_dim;
        float* current_table = table + subvec_idx * m_ks;
        const float* query_subvector = query + subvec_offset;

        for (size_t k = 0; k < m_ks; k++) {
            const float* centroid = current_codebook + k * m_subvec_dim;
            float squared_dist = simd_squared_distance(query_subvector, centroid, m_subvec_dim);
            current_table[k] = squared_dist;  // 直接使用L2距离
        }
    }
};

// 优化的IVF-PQ索引类
class IVFPQIndex {
private:
    size_t m_dim;
    size_t m_num_clusters;
    size_t m_num_subvectors;
    std::vector<float> m_cluster_centers;  
    std::vector<InvertedList> m_inverted_lists;  
    std::unique_ptr<ProductQuantizer> m_pq;
    std::vector<size_t> m_cluster_sizes;  // 追踪聚类大小
    static std::unique_ptr<ThreadPool> thread_pool;
    static pthread_mutex_t print_mutex;
    static bool initialized;

    static void initialize_pool() {
        if (!initialized) {
            thread_pool.reset(new ThreadPool());  // 替换 make_unique
            initialized = true;
        }
    }

public:
    IVFPQIndex(size_t dim, size_t num_clusters, size_t num_subvectors, size_t pq_ks = 256)
        : m_dim(dim), m_num_clusters(num_clusters), m_num_subvectors(num_subvectors) {
        initialize_pool();
        
        if (dim == 0 || num_clusters == 0 || num_subvectors == 0) {
            throw std::invalid_argument("Invalid parameters for IVF-PQ index");
        }
        
        m_cluster_centers.resize(num_clusters * dim);
        m_inverted_lists.resize(num_clusters);
        m_cluster_sizes.resize(num_clusters, 0);
        m_pq.reset(new ProductQuantizer(dim, num_subvectors, pq_ks));  // 替换 make_unique
    }

    // 优化的训练和构建过程
    void train_and_build(const float* data, size_t n, int kmeans_iter = 30, int pq_iter = 30) {
        if (n == 0) {
            std::cerr << "Warning: No data to train IVF-PQ index" << std::endl;
            return;
        }

        pthread_mutex_lock(&print_mutex);
        std::cerr << "开始优化IVF-PQ训练，数据量: " << n << ", 聚类数: " << m_num_clusters << std::endl;
        pthread_mutex_unlock(&print_mutex);

        // 1. 使用K-means++训练聚类中心
        train_clusters_optimized(data, n, kmeans_iter);

        // 2. 分配数据到聚类
        std::vector<std::vector<uint32_t>> cluster_assignments(m_num_clusters);
        assign_to_clusters_optimized(data, n, cluster_assignments);

        // 3. 计算残差向量并训练PQ
        std::vector<float> residual_data;
        prepare_residual_data_optimized(data, n, cluster_assignments, residual_data);
        
        if (!residual_data.empty()) {
            m_pq->train(residual_data.data(), residual_data.size() / m_dim, pq_iter);
        }

        // 4. 构建倒排列表
        build_inverted_lists_optimized(data, n, cluster_assignments);

        pthread_mutex_lock(&print_mutex);
        std::cerr << "优化IVF-PQ索引构建完成" << std::endl;
        pthread_mutex_unlock(&print_mutex);
    }

    // 优化的搜索算法
    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, 
        size_t k, 
        size_t nprobe = 10) const {
        
        std::priority_queue<std::pair<float, uint32_t>> result_queue;
        
        // 1. 动态nprobe调整 - 基于聚类大小分布
        std::vector<std::pair<float, uint32_t>> cluster_distances;
        for (size_t i = 0; i < m_num_clusters; i++) {
            if (m_cluster_sizes[i] == 0) continue;  // 跳过空聚类
            
            const float* center = m_cluster_centers.data() + i * m_dim;
            float dist = 0;
            
#ifdef __ARM_NEON__
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            size_t d = 0;
            for (; d + 3 < m_dim; d += 4) {
                float32x4_t q_vec = vld1q_f32(query + d);
                float32x4_t c_vec = vld1q_f32(center + d);
                float32x4_t diff = vsubq_f32(q_vec, c_vec);
                sum_vec = vfmaq_f32(sum_vec, diff, diff);
            }
            float32x2_t sum_pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
            dist = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
            
            for (; d < m_dim; d++) {
                float diff = query[d] - center[d];
                dist += diff * diff;
            }
#else
            for (size_t d = 0; d < m_dim; d++) {
                float diff = query[d] - center[d];
                dist += diff * diff;
            }
#endif
            cluster_distances.push_back({dist, static_cast<uint32_t>(i)});
        }
        
        std::sort(cluster_distances.begin(), cluster_distances.end());
        
        // 动态调整nprobe基于k和聚类分布
        size_t effective_nprobe = std::min(nprobe, cluster_distances.size());
        size_t adaptive_nprobe = std::max(effective_nprobe, k / 50 + 1);  // 自适应调整
        adaptive_nprobe = std::min(adaptive_nprobe, cluster_distances.size());

        // 2. 在选中的聚类中并行搜索
        std::vector<float> residual_query(m_dim);
        std::vector<float> distance_table(m_num_subvectors * m_pq->get_ks());
        
        for (size_t probe_idx = 0; probe_idx < adaptive_nprobe; probe_idx++) {
            uint32_t cluster_id = cluster_distances[probe_idx].second;
            const InvertedList& inv_list = m_inverted_lists[cluster_id];
            
            if (inv_list.empty()) continue;
            
            // 计算残差查询向量
            const float* center = m_cluster_centers.data() + cluster_id * m_dim;
            for (size_t d = 0; d < m_dim; d++) {
                residual_query[d] = query[d] - center[d];
            }
            
            // 计算残差查询的距离表
            m_pq->compute_distance_table(residual_query.data(), distance_table.data());
            
            // 搜索当前聚类
            for (const auto& item : inv_list) {
                float total_dist = 0;
                
                // 使用距离表快速计算PQ距离
                for (size_t subvec_idx = 0; subvec_idx < m_num_subvectors; subvec_idx++) {
                    uint8_t code = item.pq_codes[subvec_idx];
                    total_dist += distance_table[subvec_idx * m_pq->get_ks() + code];
                }
                
                // 维护top-k结果
                if (result_queue.size() < k) {
                    result_queue.push({total_dist, item.vector_id});
                } else if (total_dist < result_queue.top().first) {
                    result_queue.pop();
                    result_queue.push({total_dist, item.vector_id});
                }
            }
        }
        
        return result_queue;
    }
    
    size_t get_dim() const { return m_dim; }
    size_t get_num_clusters() const { return m_num_clusters; }
    const std::vector<size_t>& get_cluster_sizes() const { return m_cluster_sizes; }

private:
    // 优化的K-means++聚类训练 
        // 优化的K-means++聚类训练 
    void train_clusters_optimized(const float* data, size_t n, int max_iter) {
        pthread_mutex_lock(&print_mutex);
        std::cerr << "开始优化聚类中心训练..." << std::endl;
        pthread_mutex_unlock(&print_mutex);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, static_cast<int>(n - 1));
        
        // K-means++初始化
        size_t first_idx = dis(gen);
        std::memcpy(m_cluster_centers.data(), 
                   data + first_idx * m_dim,
                   m_dim * sizeof(float));
        
        for (size_t k = 1; k < m_num_clusters && k < n; k++) {
            std::vector<float> distances(n);
            float total_dist = 0;
            
            // 并行计算距离
            const size_t chunk_size = std::max((size_t)1, n / thread_pool->get_num_threads());
            std::vector<std::future<void>> futures;
            
            for (size_t start = 0; start < n; start += chunk_size) {
                size_t end = std::min(start + chunk_size, n);
                futures.push_back(std::async(std::launch::async, [&, start, end, k]() {
                    for (size_t i = start; i < end; i++) {
                        float min_dist = std::numeric_limits<float>::max();
                        const float* point = data + i * m_dim;
                        
                        for (size_t j = 0; j < k; j++) {
                            const float* center = m_cluster_centers.data() + j * m_dim;
                            float dist = 0;
                            
#ifdef __ARM_NEON__
                            float32x4_t sum_vec = vdupq_n_f32(0.0f);
                            size_t d = 0;
                            for (; d + 3 < m_dim; d += 4) {
                                float32x4_t p_vec = vld1q_f32(point + d);
                                float32x4_t c_vec = vld1q_f32(center + d);
                                float32x4_t diff = vsubq_f32(p_vec, c_vec);
                                sum_vec = vfmaq_f32(sum_vec, diff, diff);
                            }
                            float32x2_t sum_pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
                            dist = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
                            
                            for (; d < m_dim; d++) {
                                float diff = point[d] - center[d];
                                dist += diff * diff;
                            }
#else
                            for (size_t d = 0; d < m_dim; d++) {
                                float diff = point[d] - center[d];
                                dist += diff * diff;
                            }
#endif
                            min_dist = std::min(min_dist, dist);
                        }
                        distances[i] = min_dist;
                    }
                }));
            }
            
            for (auto& future : futures) {
                future.wait();
            }
            
            for (size_t i = 0; i < n; i++) {
                total_dist += distances[i];
            }
            
            if (total_dist > 0) {
                std::uniform_real_distribution<float> prob_dis(0, total_dist);
                float target = prob_dis(gen);
                float cumsum = 0;
                size_t selected = 0;
                
                for (size_t i = 0; i < n; i++) {
                    cumsum += distances[i];
                    if (cumsum >= target) {
                        selected = i;
                        break;
                    }
                }
                
                std::memcpy(m_cluster_centers.data() + k * m_dim,
                           data + selected * m_dim,
                           m_dim * sizeof(float));
            } else {
                size_t idx = dis(gen);
                std::memcpy(m_cluster_centers.data() + k * m_dim,
                           data + idx * m_dim,
                           m_dim * sizeof(float));
            }
        }
        
        // 如果数据点少于聚类数，复制现有中心
        for (size_t k = std::min(m_num_clusters, n); k < m_num_clusters; k++) {
            size_t src_idx = k % std::min(m_num_clusters, n);
            std::memcpy(m_cluster_centers.data() + k * m_dim,
                       m_cluster_centers.data() + src_idx * m_dim,
                       m_dim * sizeof(float));
        }
        
        // K-means迭代优化
        std::vector<size_t> assignments(n);
        std::vector<size_t> counts(m_num_clusters);
        std::vector<std::vector<float>> new_centers(m_num_clusters, std::vector<float>(m_dim, 0));
        
        for (int iter = 0; iter < max_iter; iter++) {
            // 重置
            for (auto& center : new_centers) {
                std::fill(center.begin(), center.end(), 0);
            }
            std::fill(counts.begin(), counts.end(), 0);
            
            // 分配步骤 - 并行化
            const size_t chunk_size = std::max((size_t)1, n / thread_pool->get_num_threads());
            std::vector<std::future<void>> assign_futures;
            std::vector<std::vector<std::vector<float>>> thread_centers(thread_pool->get_num_threads(), 
                std::vector<std::vector<float>>(m_num_clusters, std::vector<float>(m_dim, 0)));
            std::vector<std::vector<size_t>> thread_counts(thread_pool->get_num_threads(), 
                std::vector<size_t>(m_num_clusters, 0));
            
            for (size_t t = 0; t < thread_pool->get_num_threads(); t++) {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, n);
                if (start >= n) break;
                
                assign_futures.push_back(std::async(std::launch::async, [&, t, start, end]() {
                    for (size_t i = start; i < end; i++) {
                        float min_dist = std::numeric_limits<float>::max();
                        size_t best_cluster = 0;
                        const float* point = data + i * m_dim;
                        
                        for (size_t k = 0; k < m_num_clusters; k++) {
                            const float* center = m_cluster_centers.data() + k * m_dim;
                            float dist = 0;
                            
#ifdef __ARM_NEON__
                            float32x4_t sum_vec = vdupq_n_f32(0.0f);
                            size_t d = 0;
                            for (; d + 3 < m_dim; d += 4) {
                                float32x4_t p_vec = vld1q_f32(point + d);
                                float32x4_t c_vec = vld1q_f32(center + d);
                                float32x4_t diff = vsubq_f32(p_vec, c_vec);
                                sum_vec = vfmaq_f32(sum_vec, diff, diff);
                            }
                            float32x2_t sum_pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
                            dist = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
                            
                            for (; d < m_dim; d++) {
                                float diff = point[d] - center[d];
                                dist += diff * diff;
                            }
#else
                            for (size_t d = 0; d < m_dim; d++) {
                                float diff = point[d] - center[d];
                                dist += diff * diff;
                            }
#endif
                            if (dist < min_dist) {
                                min_dist = dist;
                                best_cluster = k;
                            }
                        }
                        
                        assignments[i] = best_cluster;
                        thread_counts[t][best_cluster]++;
                        for (size_t d = 0; d < m_dim; d++) {
                            thread_centers[t][best_cluster][d] += point[d];
                        }
                    }
                }));
            }
            
            for (auto& future : assign_futures) {
                future.wait();
            }
            
            // 合并线程结果
            for (size_t t = 0; t < thread_pool->get_num_threads(); t++) {
                for (size_t k = 0; k < m_num_clusters; k++) {
                    counts[k] += thread_counts[t][k];
                    for (size_t d = 0; d < m_dim; d++) {
                        new_centers[k][d] += thread_centers[t][k][d];
                    }
                }
            }
            
            // 更新中心
            bool converged = true;
            for (size_t k = 0; k < m_num_clusters; k++) {
                float* center = m_cluster_centers.data() + k * m_dim;
                if (counts[k] > 0) {
                    float inv_count = 1.0f / counts[k];
                    for (size_t d = 0; d < m_dim; d++) {
                        float new_val = new_centers[k][d] * inv_count;
                        if (std::abs(center[d] - new_val) > 1e-6f) {
                            converged = false;
                        }
                        center[d] = new_val;
                    }
                }
            }
            
            if (converged) break;
        }
        
        pthread_mutex_lock(&print_mutex);
        std::cerr << "聚类中心训练完成" << std::endl;
        pthread_mutex_unlock(&print_mutex);
    }
    
    // 优化的数据分配
    void assign_to_clusters_optimized(const float* data, size_t n, 
                                    std::vector<std::vector<uint32_t>>& cluster_assignments) {
        
        std::fill(m_cluster_sizes.begin(), m_cluster_sizes.end(), 0);
        
        const size_t chunk_size = std::max((size_t)1, n / thread_pool->get_num_threads());
        std::vector<std::future<void>> futures;
        std::vector<std::vector<std::vector<uint32_t>>> thread_assignments(
            thread_pool->get_num_threads(), 
            std::vector<std::vector<uint32_t>>(m_num_clusters));
        
        for (size_t t = 0; t < thread_pool->get_num_threads(); t++) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, n);
            if (start >= n) break;
            
            futures.push_back(std::async(std::launch::async, [&, t, start, end]() {
                for (size_t i = start; i < end; i++) {
                    float min_dist = std::numeric_limits<float>::max();
                    size_t best_cluster = 0;
                    const float* point = data + i * m_dim;
                    
                    for (size_t k = 0; k < m_num_clusters; k++) {
                        const float* center = m_cluster_centers.data() + k * m_dim;
                        float dist = 0;
                        
                        for (size_t d = 0; d < m_dim; d++) {
                            float diff = point[d] - center[d];
                            dist += diff * diff;
                        }
                        
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_cluster = k;
                        }
                    }
                    
                    thread_assignments[t][best_cluster].push_back(static_cast<uint32_t>(i));
                }
            }));
        }
        
        for (auto& future : futures) {
            future.wait();
        }
        
        // 合并结果
        for (size_t t = 0; t < thread_pool->get_num_threads(); t++) {
            for (size_t k = 0; k < m_num_clusters; k++) {
                cluster_assignments[k].insert(cluster_assignments[k].end(),
                    thread_assignments[t][k].begin(), thread_assignments[t][k].end());
                m_cluster_sizes[k] += thread_assignments[t][k].size();
            }
        }
    }
    
    // 优化的残差数据准备 
    void prepare_residual_data_optimized(const float* data, size_t n,
                                       const std::vector<std::vector<uint32_t>>& cluster_assignments,
                                       std::vector<float>& residual_data) {
        
        size_t total_points = 0;
        for (const auto& cluster : cluster_assignments) {
            total_points += cluster.size();
        }
        
        residual_data.resize(total_points * m_dim);
        
        pthread_mutex_lock(&print_mutex);
        std::cerr << "准备残差数据，总点数: " << total_points << std::endl;
        pthread_mutex_unlock(&print_mutex);
        
        size_t offset = 0;
        for (size_t k = 0; k < m_num_clusters; k++) {
            const auto& cluster = cluster_assignments[k];
            if (cluster.empty()) continue;
            
            const float* center = m_cluster_centers.data() + k * m_dim;
            
            for (uint32_t point_idx : cluster) {
                const float* point = data + point_idx * m_dim;
                float* residual = residual_data.data() + offset * m_dim;
                
                for (size_t d = 0; d < m_dim; d++) {
                    residual[d] = point[d] - center[d];  // 计算残差
                }
                offset++;
            }
        }
    }
    
    // 优化的倒排列表构建
    void build_inverted_lists_optimized(const float* data, size_t n,
                                       const std::vector<std::vector<uint32_t>>& cluster_assignments) {
        
        pthread_mutex_lock(&print_mutex);
        std::cerr << "构建倒排列表..." << std::endl;
        pthread_mutex_unlock(&print_mutex);
        
        for (size_t k = 0; k < m_num_clusters; k++) {
            const auto& cluster = cluster_assignments[k];
            if (cluster.empty()) continue;
            
            const float* center = m_cluster_centers.data() + k * m_dim;
            
            for (uint32_t point_idx : cluster) {
                const float* point = data + point_idx * m_dim;
                std::vector<float> residual(m_dim);
                
                // 计算残差
                for (size_t d = 0; d < m_dim; d++) {
                    residual[d] = point[d] - center[d];
                }
                
                // PQ编码
                std::vector<uint8_t> pq_codes(m_num_subvectors);
                m_pq->encode(residual.data(), pq_codes.data());
                
                // 添加到倒排列表
                m_inverted_lists[k].emplace_back(point_idx, pq_codes);
            }
        }
        
        pthread_mutex_lock(&print_mutex);
        std::cerr << "倒排列表构建完成" << std::endl;
        pthread_mutex_unlock(&print_mutex);
    }
};

// 静态成员初始化
std::unique_ptr<ThreadPool> ProductQuantizer::thread_pool;
pthread_mutex_t ProductQuantizer::print_mutex = PTHREAD_MUTEX_INITIALIZER;
bool ProductQuantizer::initialized = false;

std::unique_ptr<ThreadPool> IVFPQIndex::thread_pool;
pthread_mutex_t IVFPQIndex::print_mutex = PTHREAD_MUTEX_INITIALIZER;  
bool IVFPQIndex::initialized = false;

// 主搜索函数 - 改进版本
std::priority_queue<std::pair<float, uint32_t>> ivf_pq_search(
    const IVFPQIndex& index,
    const float* query,
    size_t k,
    size_t nprobe = 10) {
    
    // 动态调整nprobe基于数据分布
    const auto& cluster_sizes = index.get_cluster_sizes();
    size_t non_empty_clusters = 0;
    for (size_t size : cluster_sizes) {
        if (size > 0) non_empty_clusters++;
    }
    
    // 自适应nprobe调整策略 
    size_t adaptive_nprobe = std::min(nprobe, non_empty_clusters);
    if (k > 50) {
        adaptive_nprobe = std::min(adaptive_nprobe * 2, non_empty_clusters);
    }
    
    return index.search(query, k, adaptive_nprobe);
}

