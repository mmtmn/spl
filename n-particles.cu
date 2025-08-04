// main.cu
// compile: nvcc -o main main.cu -lsfml-graphics -lsfml-window -lsfml-system

#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <random>
#include <array>
#include <iostream>
#include <fstream>

const int width = 2560;
const int height = 1440;

const int total_groups = 10;
const int total_particles = 10000;
const int particles_per_group = total_particles / total_groups;

struct Particle {
    float x, y, vx, vy;
    unsigned char r, g, b, a;
};

__global__ void update_particles_and_score(
    Particle* particles,
    int num_particles,
    const float* interaction_matrix,
    int group_offset,
    int group_size,
    int* group_ids,
    float* out_fitness,
    int total_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= group_size) return;

    __shared__ float shared_score[256];
    shared_score[threadIdx.x] = 0;

    Particle& p1 = particles[group_offset + idx];
    float fx = 0, fy = 0;

    int self_group = group_ids[total_groups * 2];  // stored at the end

    for (int g = 0; g < total_groups; ++g) {
        int target_offset = group_ids[g];
        int target_size = group_ids[g + total_groups];

        float interaction = interaction_matrix[self_group * total_groups + g];

        for (int j = 0; j < target_size; ++j) {
            Particle p2 = particles[target_offset + j];
            float dx = p1.x - p2.x;
            float dy = p1.y - p2.y;
            float dist2 = dx * dx + dy * dy;

            if (dist2 > 0 && dist2 < 6400) {
                float dist = sqrtf(dist2);
                float force = interaction / dist;
                fx += force * dx;
                fy += force * dy;
            }
        }
    }

    p1.vx = (p1.vx + fx) * 0.5f;
    p1.vy = (p1.vy + fy) * 0.5f;
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

int main() {
    sf::RenderWindow window(sf::VideoMode(width, height), "CUDA Biogenesis Sim");

    std::vector<Particle> all_particles;
    std::vector<sf::Color> used_colors = {
        sf::Color::Green, sf::Color::Red, sf::Color::Yellow
    };

    auto gen_unique_color = [&](std::vector<sf::Color>& used) {
        while (true) {
            sf::Color c(rand() % 256, rand() % 256, rand() % 256);
            bool dup = false;
            for (auto& u : used)
                if (c == u) { dup = true; break; }
            if (!dup) return c;
        }
    };

    // Create all groups
    for (int i = 0; i < total_groups; ++i) {
        sf::Color col;
        if (i < used_colors.size()) {
            col = used_colors[i];
        } else {
            col = gen_unique_color(used_colors);
            used_colors.push_back(col);
        }

        create_particles(all_particles, particles_per_group, col);
    }

    std::vector<Particle> original_particles = all_particles;

    // Interaction matrix (flattened total_groups x total_groups)
    int matrix_size = total_groups * total_groups;
    std::vector<float> interaction_matrix(matrix_size, 0.0f);
    std::mt19937 gen(std::random_device{}());
    for (int i = 0; i < matrix_size; ++i)
        interaction_matrix[i] = 0.0f;

    for (int g = 0; g < total_groups; ++g)
        interaction_matrix[g * total_groups + g] = 0.5f;



    float* device_matrix;
    cudaMalloc(&device_matrix, matrix_size * sizeof(float));
    cudaMemcpy(device_matrix, interaction_matrix.data(), matrix_size * sizeof(float), cudaMemcpyHostToDevice);

    // Setup group data
    std::vector<int> group_offsets, group_sizes;
    int offset = 0;
    for (int i = 0; i < total_groups; ++i) {
        group_offsets.push_back(offset);
        group_sizes.push_back(particles_per_group);
        offset += particles_per_group;
    }

    // Flattened group_ids: [offsets][sizes][current_group]
    std::vector<int> group_ids(total_groups * 2 + 1, 0);
    for (int i = 0; i < total_groups; ++i) {
        group_ids[i] = group_offsets[i];
        group_ids[i + total_groups] = group_sizes[i];
    }

    int* device_group_ids;
    cudaMalloc(&device_group_ids, group_ids.size() * sizeof(int));

    Particle* device_particles;
    cudaMalloc(&device_particles, all_particles.size() * sizeof(Particle));
    cudaMemcpy(device_particles, original_particles.data(), all_particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);

    float* device_fitness;
    cudaMalloc(&device_fitness, sizeof(float));

    int threads_per_block = 256;

    while (window.isOpen()) {
        sf::Event e;
        while (window.pollEvent(e)) {
            if (e.type == sf::Event::Closed)
                window.close();
        }

        cudaMemset(device_fitness, 0, sizeof(float));

        int steps = 10;
        for (int step = 0; step < steps; ++step) {
            for (int group = 0; group < total_groups; ++group) {
                group_ids[group_ids.size() - 1] = group;
                cudaMemcpy(device_group_ids, group_ids.data(), group_ids.size() * sizeof(int), cudaMemcpyHostToDevice);
                int offset = group_ids[group];
                int size = group_ids[group + total_groups];
                int blocks = (size + threads_per_block - 1) / threads_per_block;

                update_particles_and_score<<<blocks, threads_per_block>>>(
                    device_particles, all_particles.size(), device_matrix,
                    offset, size, device_group_ids, device_fitness, total_groups
                );
            }
        }

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

    cudaFree(device_matrix);
    cudaFree(device_particles);
    cudaFree(device_group_ids);
    cudaFree(device_fitness);
    return 0;
}
