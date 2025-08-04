// spatial hash nudge-spatial-hash-grid.cu
// compile: nvcc -o nudge-spatial-hash-grid nudge-spatial-hash-grid.cu -lsfml-graphics -lsfml-window -lsfml-system

#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <random>
#include <array>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <curand_kernel.h>

const int width = 2560;
const int height = 1440;
const int total_groups = 5;
const int total_particles = 10000;
const int particles_per_group = total_particles / total_groups;
const int cell_size = 80;
const int grid_cols = width / cell_size;
const int grid_rows = height / cell_size;

struct Particle {
    float x, y, vx, vy;
    unsigned char r, g, b, a;
};

__device__ int clamp(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}

__global__ void update_particles_spatial(
    Particle* particles,
    int* bin_starts,
    int* bin_counts,
    int* bin_particles,
    int num_particles,
    const float* interaction_matrix,
    int* group_ids,
    float* out_fitness,
    int total_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    __shared__ float shared_score[256];

    curandState state;
    curand_init(clock64() + idx, 0, 0, &state);


    Particle& p1 = particles[idx];

    float fx = 0, fy = 0;

    int group_idx = group_ids[total_groups * 2];

    int cell_x = clamp((int)(p1.x / cell_size), 0, grid_cols - 1);
    int cell_y = clamp((int)(p1.y / cell_size), 0, grid_rows - 1);

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = cell_x + dx;
            int ny = cell_y + dy;
            if (nx < 0 || nx >= grid_cols || ny < 0 || ny >= grid_rows) continue;
            int cell_id = ny * grid_cols + nx;

            int start = bin_starts[cell_id];
            int count = bin_counts[cell_id];

            for (int i = 0; i < count; ++i) {
                int j = bin_particles[start + i];
                if (j == idx) continue;

                Particle p2 = particles[j];

                float dx = p1.x - p2.x;
                float dy = p1.y - p2.y;
                float dist2 = dx * dx + dy * dy;
                if (dist2 > 0 && dist2 < 6400) {
                    float dist = sqrtf(dist2);
                    int g2 = group_ids[j + (total_groups * 3)];
                    float interaction = interaction_matrix[group_idx * total_groups + g2];
                    float force = interaction / dist;
                    fx += force * dx;
                    fy += force * dy;
                }
            }
        }
    }

    p1.vx = (p1.vx + fx) * 0.95f + (curand_uniform(&state) - 0.5f) * 0.01f;
    p1.vy = (p1.vy + fy) * 0.95f + (curand_uniform(&state) - 0.5f) * 0.01f;

    p1.x += p1.vx;
    p1.y += p1.vy;

    if (p1.x <= 0 || p1.x >= width) p1.vx *= -1;
    if (p1.y <= 0 || p1.y >= height) p1.vy *= -1;
    p1.x = fminf(fmaxf(p1.x, 0), width);
    p1.y = fminf(fmaxf(p1.y, 0), height);

    shared_score[threadIdx.x] = expf(-sqrtf(p1.vx * p1.vx + p1.vy * p1.vy) * 0.1f);
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < blockDim.x; ++i) total += shared_score[i];
        atomicAdd(out_fitness, total);
    }
}

void create_particles(std::vector<Particle>& particles, int number, sf::Color color) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis_x(50, width - 50);
    std::uniform_real_distribution<float> dis_y(50, height - 50);
    std::uniform_real_distribution<float> dis_v(-1.0f, 1.0f);
    for (int i = 0; i < number; ++i) {
        Particle p;
        p.x = dis_x(gen);
        p.y = dis_y(gen);
        p.vx = dis_v(gen);
        p.vy = dis_v(gen);
        p.r = color.r;
        p.g = color.g;
        p.b = color.b;
        p.a = color.a;
        particles.push_back(p);
    }
}

int get_cell_id(float x, float y) {
    int cell_x = std::min((int)(x / cell_size), grid_cols - 1);
    int cell_y = std::min((int)(y / cell_size), grid_rows - 1);
    return cell_y * grid_cols + cell_x;
}

int main() {
    sf::RenderWindow window(sf::VideoMode(width, height), "CUDA Biogenesis Sim");

    std::vector<Particle> all_particles;
    std::vector<sf::Color> used_colors;
    std::vector<sf::Color> base_colors = { sf::Color::Green, sf::Color::Red, sf::Color::Yellow };

    auto gen_unique_color = [&](const std::vector<sf::Color>& used) {
        while (true) {
            sf::Color c(rand() % 256, rand() % 256, rand() % 256);
            bool dup = false;
            for (auto& u : used) if (c == u) { dup = true; break; }
            if (!dup) return c;
        }
    };

    std::vector<int> group_for_particle;
    for (int i = 0; i < total_groups; ++i) {
        sf::Color col;
        if (i < base_colors.size()) {
            col = base_colors[i];
        } else {
            col = gen_unique_color(used_colors);
        }
        used_colors.push_back(col);
        create_particles(all_particles, particles_per_group, col);
        for (int j = 0; j < particles_per_group; ++j) group_for_particle.push_back(i);
    }


    std::vector<Particle> original_particles = all_particles;

    std::vector<float> interaction_matrix(total_groups * total_groups, 0.0f);
    for (int g = 0; g < total_groups; ++g)
        interaction_matrix[g * total_groups + g] = 0.5f;

    float* device_matrix;
    cudaMalloc(&device_matrix, interaction_matrix.size() * sizeof(float));
    cudaMemcpy(device_matrix, interaction_matrix.data(), interaction_matrix.size() * sizeof(float), cudaMemcpyHostToDevice);

    Particle* device_particles;
    cudaMalloc(&device_particles, all_particles.size() * sizeof(Particle));

    float* device_fitness;
    cudaMalloc(&device_fitness, sizeof(float));

    std::vector<int> bin_starts(grid_cols * grid_rows, 0);
    std::vector<int> bin_counts(grid_cols * grid_rows, 0);
    std::vector<int> bin_particles(all_particles.size());

    int* d_bin_starts;
    int* d_bin_counts;
    int* d_bin_particles;

    cudaMalloc(&d_bin_starts, bin_starts.size() * sizeof(int));
    cudaMalloc(&d_bin_counts, bin_counts.size() * sizeof(int));
    cudaMalloc(&d_bin_particles, bin_particles.size() * sizeof(int));

    std::vector<int> group_ids(total_particles + total_groups * 3, 0);
    for (int i = 0; i < total_particles; ++i) group_ids[i + total_groups * 3] = group_for_particle[i];

    int* d_group_ids;
    cudaMalloc(&d_group_ids, group_ids.size() * sizeof(int));
    cudaMemcpy(d_group_ids, group_ids.data(), group_ids.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_particles, original_particles.data(), all_particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);

    int threads_per_block = 256;

    while (window.isOpen()) {
        sf::Event e;
        while (window.pollEvent(e)) if (e.type == sf::Event::Closed) window.close();

        cudaMemset(device_fitness, 0, sizeof(float));

        std::fill(bin_counts.begin(), bin_counts.end(), 0);
        for (int i = 0; i < total_particles; ++i) {
            int cell_id = get_cell_id(original_particles[i].x, original_particles[i].y);
            bin_counts[cell_id]++;
        }

        // compute bin_starts from bin_counts
        bin_starts[0] = 0;
        for (size_t i = 1; i < bin_starts.size(); ++i) {
            bin_starts[i] = bin_starts[i - 1] + bin_counts[i - 1];
        }

        // refill bin_counts to use as offsets
        std::fill(bin_counts.begin(), bin_counts.end(), 0);
        for (int i = 0; i < total_particles; ++i) {
            int cell_id = get_cell_id(original_particles[i].x, original_particles[i].y);
            int index = bin_starts[cell_id] + bin_counts[cell_id]++;
            bin_particles[index] = i;
        }


        cudaMemcpy(d_bin_particles, bin_particles.data(), total_particles * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bin_counts, bin_counts.data(), bin_counts.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bin_starts, bin_starts.data(), bin_starts.size() * sizeof(int), cudaMemcpyHostToDevice);

        int blocks = (total_particles + threads_per_block - 1) / threads_per_block;

        update_particles_spatial<<<blocks, threads_per_block>>>(
            device_particles, d_bin_starts, d_bin_counts, d_bin_particles,
            total_particles, device_matrix, d_group_ids, device_fitness, total_groups
        );

        cudaDeviceSynchronize();

        cudaMemcpy(all_particles.data(), device_particles, all_particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);

        window.clear();
        sf::VertexArray vertices(sf::Points, all_particles.size());
        for (size_t i = 0; i < all_particles.size(); ++i) {
            vertices[i].position = sf::Vector2f(all_particles[i].x, all_particles[i].y);
            vertices[i].color = sf::Color(all_particles[i].r, all_particles[i].g, all_particles[i].b, all_particles[i].a);
        }
        window.draw(vertices);
        window.display();
    }

    cudaFree(device_particles);
    cudaFree(device_matrix);
    cudaFree(device_fitness);
    cudaFree(d_bin_particles);
    cudaFree(d_bin_counts);
    cudaFree(d_bin_starts);
    cudaFree(d_group_ids);
}
