// main.cu

// to compile: nvcc -o main main.cu -lsfml-graphics -lsfml-window -lsfml-system

#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <vector>
#include <random>
#include <array>
#include <iostream>
#include <fstream>

const int width = 2560;
const int height = 1440;
const int num_particles = 120000;
const int num_green = num_particles / 3;
const int num_red = num_particles / 3;
const int num_yellow = num_particles - num_green - num_red;
const int rule_count = 9;

struct Particle {
    float x, y, vx, vy;
    unsigned char r, g, b, a;
};

__device__ float fitness(Particle* particles, int num_particles) {
    float score = 0;
    for (int i = 0; i < num_particles - 1; i++) {
        float dx = particles[i + 1].x - particles[i].x;
        float dy = particles[i + 1].y - particles[i].y;
        float d = sqrtf(dx * dx + dy * dy);
        score += expf(-d * 0.01f);
    }
    return score / num_particles;
}

__global__ void update_particles_and_score(
    Particle* particles,
    int num_particles,
    const float* interaction_matrix,
    int group_offset,
    int group_size,
    int* group_ids,
    float* out_fitness
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= group_size) return;

    __shared__ float shared_score[256];
    shared_score[threadIdx.x] = 0;

    Particle& p1 = particles[group_offset + idx];
    float fx = 0, fy = 0;

    for (int g = 0; g < 3; g++) {
        int target_offset = group_ids[g];
        int target_size = group_ids[g + 3];

        for (int j = 0; j < target_size; ++j) {
            Particle p2 = particles[target_offset + j];
            float dx = p1.x - p2.x;
            float dy = p1.y - p2.y;
            float dist2 = dx * dx + dy * dy;

            if (dist2 > 0 && dist2 < 6400) {
                float dist = sqrtf(dist2);
                float interaction = interaction_matrix[group_ids[6] * 3 + g];
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
    create_particles(all_particles, num_green, sf::Color::Green);
    create_particles(all_particles, num_red, sf::Color::Red);
    create_particles(all_particles, num_yellow, sf::Color::Yellow);
    std::vector<Particle> original_particles = all_particles;
    // std::cout << "Particles created: " << all_particles.size() << std::endl;


    Particle* device_particles;
    cudaMalloc(&device_particles, all_particles.size() * sizeof(Particle));
    cudaMemcpy(device_particles, original_particles.data(), original_particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);

    float best_rules[rule_count];
    float best_score = -1;

    int group_ids[7] = {
        0, num_green, num_green + num_red,
        num_green, num_red, num_yellow,
        0
    };
    int* device_group_ids;
    cudaMalloc(&device_group_ids, sizeof(group_ids));

    float* device_fitness;
    cudaMalloc(&device_fitness, sizeof(float));

    int threads_per_block = 256;

    while (window.isOpen()) {
        sf::Event e;
        while (window.pollEvent(e)) {
            if (e.type == sf::Event::Closed)
                window.close();
        }

        float trial_rules[rule_count];
        std::mt19937 gen(std::random_device{}());
        std::ifstream infile("best_rules.txt");
        if (infile.is_open()) {
            float dummy_score;
            infile >> dummy_score;
            for (int i = 0; i < rule_count; ++i)
                infile >> trial_rules[i];
            infile.close();
            // std::cout << "Loaded rules from file" << std::endl;
        } else {
            std::uniform_real_distribution<float> dis(-1.5f, 1.5f);
            for (int i = 0; i < rule_count; ++i) trial_rules[i] = dis(gen);
        }


        float* device_rules;
        cudaMalloc(&device_rules, sizeof(trial_rules));
        cudaMemcpy(device_rules, trial_rules, sizeof(trial_rules), cudaMemcpyHostToDevice);
        cudaMemset(device_fitness, 0, sizeof(float));

        cudaMemcpy(device_particles, original_particles.data(), original_particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);


        int steps = 10; // or 200, tune as needed
        for (int step = 0; step < steps; ++step) {
            for (int group = 0; group < 3; ++group) {
                group_ids[6] = group;
                cudaMemcpy(device_group_ids, group_ids, sizeof(group_ids), cudaMemcpyHostToDevice);
                int offset = group_ids[group];
                int size = group_ids[group + 3];
                int blocks = (size + threads_per_block - 1) / threads_per_block;
                update_particles_and_score<<<blocks, threads_per_block>>>(
                    device_particles, all_particles.size(), device_rules, offset, size, device_group_ids, device_fitness
                );
            }
        }


        cudaDeviceSynchronize();
        // std::cout << "Kernel sync complete" << std::endl;


        float current_score = 0;
        cudaMemcpy(&current_score, device_fitness, sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Current score: " << current_score << ", Best score: " << best_score << std::endl;

        if (current_score > best_score) {
            best_score = current_score;
            std::copy(std::begin(trial_rules), std::end(trial_rules), std::begin(best_rules));
            std::cout << "New best score found: " << best_score << std::endl;
            std::ofstream out("best_rules.txt");
            out << best_score << std::endl;
            for (int i = 0; i < rule_count; ++i)
                out << best_rules[i] << (i < rule_count - 1 ? " " : "\n");
            out.close();
        }

        cudaMemcpy(all_particles.data(), device_particles, all_particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);
        window.clear();
        sf::VertexArray vertices(sf::Points, all_particles.size());
        for (size_t i = 0; i < all_particles.size(); ++i) {
            vertices[i].position = sf::Vector2f(all_particles[i].x, all_particles[i].y);
            vertices[i].color = sf::Color(all_particles[i].r, all_particles[i].g, all_particles[i].b, all_particles[i].a);
        }
        window.draw(vertices);
        window.display();
        cudaFree(device_rules);
    }

    cudaFree(device_particles);
    cudaFree(device_group_ids);
    cudaFree(device_fitness);
    return 0;
}
