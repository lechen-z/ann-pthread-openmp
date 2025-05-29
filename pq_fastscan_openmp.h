#include <vector>
#include <queue>
#include <utility>
#include <cstdint>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>
#include <memory>
#include <omp.h>
#include <iostream>
#include <stdexcept>
#include <cstring>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

class ProductQuantizer {
private:
    size_t m_dim;
    size_t m_num_subvectors;
    size_t m_subvec_dim;
    size_t m_ks;
    std::vector<float> m_codebooks;

public:
    ProductQuantizer(size_t dim, size_t num_subvectors, size_t ks = 256)
        : m_dim(dim), m_num_subvectors(num_subvectors), m_ks(ks) {
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

    void train(const float* data, size_t n, int max_iter = 20) {
         if (n == 0) {
             std::cerr << "Warning: Trying to train PQ on 0 data points." << std::endl;
             return;
        }
        std::cerr << "开始PQ训练，数据量: " << n << ", 子向量数: " << m_num_subvectors << ", 维度: " << m_dim << std::endl;
        
        #pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
        for (size_t subvec_idx = 0; subvec_idx < m_num_subvectors; subvec_idx++) {
            size_t subvec_offset = subvec_idx * m_subvec_dim;
            float* current_codebook = m_codebooks.data() + subvec_idx * m_ks * m_subvec_dim;
            std::vector<float> subvec_data(n * m_subvec_dim);
             if (subvec_data.empty() && (n * m_subvec_dim > 0)) {
                 #pragma omp critical
                 { std::cerr << "Error: Failed to allocate memory for subvector data in thread for subvec " << subvec_idx << std::endl; }
                 continue;
             }

            for (size_t i = 0; i < n; i++) {
                std::memcpy(subvec_data.data() + i * m_subvec_dim,
                           data + i * m_dim + subvec_offset,
                           m_subvec_dim * sizeof(float));
            }

            std::random_device rd;
            std::mt19937 gen(rd() + subvec_idx);
            std::uniform_int_distribution<> dis(0, static_cast<int>(n - 1));
            std::vector<size_t> initial_indices(m_ks);
            for (size_t i = 0; i < m_ks; ++i) initial_indices[i] = dis(gen);
            std::sort(initial_indices.begin(), initial_indices.end());
            initial_indices.erase(std::unique(initial_indices.begin(), initial_indices.end()), initial_indices.end());
            while(initial_indices.size() < m_ks && initial_indices.size() < n) {
                size_t idx = dis(gen);
                if(std::find(initial_indices.begin(), initial_indices.end(), idx) == initial_indices.end()) {
                    initial_indices.push_back(idx);
                }
            }
            if (initial_indices.size() < m_ks) {
                 #pragma omp critical
                 { std::cerr << "Warning: Could not find " << m_ks << " unique initial points for subvector " << subvec_idx << ". Using duplicates." << std::endl; }
                 while(initial_indices.size() < m_ks) initial_indices.push_back(dis(gen));
            }

            for (size_t i = 0; i < m_ks; i++) {
                 size_t idx = initial_indices[i];
                 std::memcpy(current_codebook + i * m_subvec_dim,
                             subvec_data.data() + idx * m_subvec_dim,
                             m_subvec_dim * sizeof(float));
            }

            std::vector<size_t> assignments(n);
            std::vector<size_t> counts(m_ks);
            std::vector<float> new_centroids(m_ks * m_subvec_dim);

            for (int iter = 0; iter < max_iter; iter++) {
                std::fill(counts.begin(), counts.end(), 0);
                std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);

                for (size_t i = 0; i < n; i++) {
                    float min_dist_sq = std::numeric_limits<float>::max();
                    size_t best_centroid = 0;
                    const float* current_point = subvec_data.data() + i * m_subvec_dim;

                    for (size_t k = 0; k < m_ks; k++) {
                        float dist_sq = 0;
                        const float* centroid = current_codebook + k * m_subvec_dim;
                        
                        #pragma omp simd reduction(+:dist_sq)
                        for (size_t d = 0; d < m_subvec_dim; d++) {
                            float diff = current_point[d] - centroid[d];
                            dist_sq += diff * diff;
                        }
                        if (dist_sq < min_dist_sq) {
                            min_dist_sq = dist_sq;
                            best_centroid = k;
                        }
                    }
                    assignments[i] = best_centroid;
                    counts[best_centroid]++;
                    float* target_centroid_sum = new_centroids.data() + best_centroid * m_subvec_dim;
                    
                    #pragma omp simd
                    for (size_t d = 0; d < m_subvec_dim; d++) {
                        target_centroid_sum[d] += current_point[d];
                    }
                }

                bool changed = false;
                for (size_t k = 0; k < m_ks; k++) {
                     float* target_centroid = current_codebook + k * m_subvec_dim;
                    if (counts[k] > 0) {
                        float* source_sum = new_centroids.data() + k * m_subvec_dim;
                        float inv_count = 1.0f / counts[k];
                        
                        #pragma omp simd
                        for (size_t d = 0; d < m_subvec_dim; d++) {
                            float new_val = source_sum[d] * inv_count;
                            if (target_centroid[d] != new_val) {
                                target_centroid[d] = new_val;
                                changed = true;
                            }
                        }
                    } else {
                         size_t idx = dis(gen);
                         std::memcpy(target_centroid, subvec_data.data() + idx * m_subvec_dim, m_subvec_dim * sizeof(float));
                         changed = true;
                    }
                }
                 if (!changed && iter > 0) break;
            }

            #pragma omp critical
            { std::cerr << "子空间 " << subvec_idx << " 码本训练完成" << std::endl; }
        }
    }

    void encode(const float* vec, uint8_t* code) const {
        #pragma omp parallel for if(m_num_subvectors >= 4)
        for (size_t subvec_idx = 0; subvec_idx < m_num_subvectors; subvec_idx++) {
            size_t subvec_offset = subvec_idx * m_subvec_dim;
            const float* current_codebook = m_codebooks.data() + subvec_idx * m_ks * m_subvec_dim;
            const float* current_subvector = vec + subvec_offset;

            float min_dist_sq = std::numeric_limits<float>::max();
            uint8_t best_centroid = 0;

            for (size_t k = 0; k < m_ks; k++) {
                float dist_sq = 0;
                const float* centroid = current_codebook + k * m_subvec_dim;
                
                #pragma omp simd reduction(+:dist_sq)
                for (size_t d = 0; d < m_subvec_dim; d++) {
                    float diff = current_subvector[d] - centroid[d];
                    dist_sq += diff * diff;
                }

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    best_centroid = static_cast<uint8_t>(k);
                }
            }
            code[subvec_idx] = best_centroid;
        }
    }

    std::vector<uint8_t> encode_dataset(const float* base, size_t n) const {
        if (n == 0) return {};
        std::vector<uint8_t> codes(n * m_num_subvectors);
         if (codes.empty() && (n * m_num_subvectors > 0)) {
             throw std::runtime_error("Failed to allocate memory for encoded dataset.");
         }
        std::cerr << "Encoding dataset with subvector-major layout..." << std::endl;

        #pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
        for (size_t i = 0; i < n; i++) {
            std::vector<uint8_t> temp_code(m_num_subvectors);
            encode(base + i * m_dim, temp_code.data());

            for (size_t j = 0; j < m_num_subvectors; j++) {
                codes[j * n + i] = temp_code[j];
            }
        }
        std::cerr << "Encoding complete." << std::endl;
        return codes;
    }

    void compute_ip_table(const float* query, float* table) const {
        #pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
        for (size_t subvec_idx = 0; subvec_idx < m_num_subvectors; subvec_idx++) {
            size_t subvec_offset = subvec_idx * m_subvec_dim;
            const float* current_codebook = m_codebooks.data() + subvec_idx * m_ks * m_subvec_dim;
            float* current_table = table + subvec_idx * m_ks;
            const float* query_subvector = query + subvec_offset;

            #pragma omp simd
            for (size_t k = 0; k < m_ks; k++) {
                const float* centroid = current_codebook + k * m_subvec_dim;
                float dot_product = 0;

#ifdef __ARM_NEON__
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                size_t d = 0;
                for (; d + 3 < m_subvec_dim; d += 4) {
                    float32x4_t query_vec = vld1q_f32(query_subvector + d);
                    float32x4_t centroid_vec = vld1q_f32(centroid + d);
                    sum_vec = vfmaq_f32(sum_vec, query_vec, centroid_vec);
                }

                float32x2_t sum_pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
                dot_product = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

                for (; d < m_subvec_dim; d++) {
                    dot_product += query_subvector[d] * centroid[d];
                }
#else
                #pragma omp simd reduction(+:dot_product)
                for (size_t d = 0; d < m_subvec_dim; d++) {
                    dot_product += query_subvector[d] * centroid[d];
                }
#endif
                current_table[k] = 1.0f - dot_product;
            }
        }
    }

    size_t get_dim() const { return m_dim; }
    size_t get_num_subvectors() const { return m_num_subvectors; }
    size_t get_subvec_dim() const { return m_subvec_dim; }
    size_t get_ks() const { return m_ks; }
    const std::vector<float>& get_codebooks() const { return m_codebooks; }
};

inline std::priority_queue<std::pair<float, uint32_t>>
pq_fastscan_search(const uint8_t* encoded_base_submajor,
                   const ProductQuantizer& pq,
                   const float* query,
                   size_t base_number,
                   size_t k)
{
    size_t num_subvectors = pq.get_num_subvectors();
    size_t ks = pq.get_ks();
    size_t vecdim = pq.get_dim();

    std::priority_queue<std::pair<float, uint32_t>> result_queue;

    std::vector<float> distance_table(num_subvectors * ks);
    pq.compute_ip_table(query, distance_table.data());

    const size_t batch_size = 1024;
    const size_t num_batches = (base_number + batch_size - 1) / batch_size;
    
    std::vector<std::vector<std::pair<float, uint32_t>>> batch_results(num_batches);

    #pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
    for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        size_t batch_start = batch_idx * batch_size;
        size_t current_batch_size = std::min(batch_size, base_number - batch_start);
        
        std::vector<float> partial_distances(current_batch_size, 0.0f);
        std::priority_queue<std::pair<float, uint32_t>> local_queue;

        for (size_t j = 0; j < num_subvectors; ++j) {
            const float* current_subvector_table = distance_table.data() + j * ks;
            const uint8_t* codes_ptr_base = encoded_base_submajor + j * base_number;
            const uint8_t* current_batch_codes = codes_ptr_base + batch_start;

#ifdef __ARM_NEON__
            size_t i = 0;
            const size_t step = 16;
            alignas(16) float dists_arr[step];

            for (; i + (step - 1) < current_batch_size; i += step) {
                uint8x16_t codes_vec = vld1q_u8(current_batch_codes + i);

                for(int lane = 0; lane < step; ++lane) {
                   uint8_t code_val = vgetq_lane_u8(codes_vec, lane);
                   dists_arr[lane] = current_subvector_table[code_val];
                }

                float32x4_t dists0 = vld1q_f32(dists_arr + 0);
                float32x4_t dists1 = vld1q_f32(dists_arr + 4);
                float32x4_t dists2 = vld1q_f32(dists_arr + 8);
                float32x4_t dists3 = vld1q_f32(dists_arr + 12);

                float32x4_t psum0 = vld1q_f32(partial_distances.data() + i + 0);
                float32x4_t psum1 = vld1q_f32(partial_distances.data() + i + 4);
                float32x4_t psum2 = vld1q_f32(partial_distances.data() + i + 8);
                float32x4_t psum3 = vld1q_f32(partial_distances.data() + i + 12);

                psum0 = vaddq_f32(psum0, dists0);
                psum1 = vaddq_f32(psum1, dists1);
                psum2 = vaddq_f32(psum2, dists2);
                psum3 = vaddq_f32(psum3, dists3);

                vst1q_f32(partial_distances.data() + i + 0, psum0);
                vst1q_f32(partial_distances.data() + i + 4, psum1);
                vst1q_f32(partial_distances.data() + i + 8, psum2);
                vst1q_f32(partial_distances.data() + i + 12, psum3);
            }

            for (; i < current_batch_size; ++i) {
                uint8_t code_val = current_batch_codes[i];
                partial_distances[i] += current_subvector_table[code_val];
            }
#else
            #pragma omp simd
            for (size_t i = 0; i < current_batch_size; ++i) {
                uint8_t code_val = current_batch_codes[i];
                partial_distances[i] += current_subvector_table[code_val];
            }
#endif
        }

        for (size_t i = 0; i < current_batch_size; ++i) {
            float final_distance = partial_distances[i];
            uint32_t current_index = static_cast<uint32_t>(batch_start + i);

            if (local_queue.size() < k) {
                local_queue.push({final_distance, current_index});
            } else if (final_distance < local_queue.top().first) {
                local_queue.pop();
                local_queue.push({final_distance, current_index});
            }
        }

        while (!local_queue.empty() && batch_results[batch_idx].size() < k) {
            batch_results[batch_idx].push_back(local_queue.top());
            local_queue.pop();
        }
    }

    for (const auto& batch_result : batch_results) {
        for (const auto& candidate : batch_result) {
            if (result_queue.size() < k) {
                result_queue.push(candidate);
            } else if (candidate.first < result_queue.top().first) {
                result_queue.pop();
                result_queue.push(candidate);
            }
        }
    }

    return result_queue;
}
