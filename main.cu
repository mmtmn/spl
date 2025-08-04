// main.cu

// nvcc -o main main.cu -lsfml-graphics -lsfml-window -lsfml-system

#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <random>
#include <array>
#include <iostream>

const int width = 2560;
const int height = 1440;
const int num_particles = 10000;
const int num_green = num_particles / 3;
const int num_red = num_particles / 3;
const int num_yellow = num_particles - num_green - num_red;

// New constants
const float energy_decay = 0.01f;
const float energy_boost = 0.05f;
const float max_energy = 1.0f;
const float gradient_strength = 0.002f;

// Extended particle with energy and type
struct Particle {
    float x, y, vx, vy, energy;
    unsigned char r, g, b, a;
    int type; // 0 = green, 1 = red, 2 = yellow
};

__device__ float get_gradient_fx(float x) {
    return gradient_strength * (width / 2.0f - x);
}

__device__ float get_gradient_fy(float y) {
    return gradient_strength * (height / 2.0f - y);
}

__global__ void update_particles_combined(
    Particle* particles,
    int num_particles,
    const float* interaction_matrix,
    int group_offset,
    int group_size,
    int* group_ids
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= group_size) return;

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

                // Autocatalysis: gain energy if same type and close
                if (p1.type == p2.type && dist < 40.0f) {
                    p1.energy = fminf(max_energy, p1.energy + energy_boost);
                }

                // Template match: strong alignment in cluster if dx ≈ dy
                if (fabsf(dx - dy) < 5.0f) {
                    fx += 0.02f * dx;
                    fy += 0.02f * dy;
                }
            }
        }
    }

    // Add energy gradient toward center
    fx += get_gradient_fx(p1.x);
    fy += get_gradient_fy(p1.y);

    // Velocity update
    p1.vx = (p1.vx + fx) * 0.5f;
    p1.vy = (p1.vy + fy) * 0.5f;

    // Position update
    p1.x += p1.vx;
    p1.y += p1.vy;

    // Energy decay
    p1.energy = fmaxf(0.0f, p1.energy - energy_decay);

    // Bounce
    if (p1.x <= 0 || p1.x >= width) p1.vx *= -1;
    if (p1.y <= 0 || p1.y >= height) p1.vy *= -1;

    // Clamp
    p1.x = fminf(fmaxf(p1.x, 0), width);
    p1.y = fminf(fmaxf(p1.y, 0), height);
}

void create_particles(std::vector<Particle>& particles, int number, sf::Color color, int type) {
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
        p.energy = 0.5f;
        p.r = color.r;
        p.g = color.g;
        p.b = color.b;
        p.a = color.a;
        p.type = type;
        particles.push_back(p);
    }
}

int main() {
    sf::RenderWindow window(sf::VideoMode(width, height), "CUDA Biogenesis Sim");

    std::vector<Particle> all_particles;
    create_particles(all_particles, num_green, sf::Color::Green, 0);
    create_particles(all_particles, num_red, sf::Color::Red, 1);
    create_particles(all_particles, num_yellow, sf::Color::Yellow, 2);

    Particle* device_particles;
    cudaMalloc(&device_particles, all_particles.size() * sizeof(Particle));
    cudaMemcpy(device_particles, all_particles.data(), all_particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);

    float interaction_matrix[9];
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < 9; ++i) interaction_matrix[i] = dis(gen);

    float* device_matrix;
    cudaMalloc(&device_matrix, sizeof(interaction_matrix));
    cudaMemcpy(device_matrix, interaction_matrix, sizeof(interaction_matrix), cudaMemcpyHostToDevice);

    int group_ids[7] = {
        0, num_green, num_green + num_red,
        num_green, num_red, num_yellow,
        0
    };
    int* device_group_ids;
    cudaMalloc(&device_group_ids, sizeof(group_ids));

    int threads_per_block = 256;

    while (window.isOpen()) {
        sf::Event e;
        while (window.pollEvent(e)) {
            if (e.type == sf::Event::Closed)
                window.close();
        }

        for (int group = 0; group < 3; ++group) {
            group_ids[6] = group;
            cudaMemcpy(device_group_ids, group_ids, sizeof(group_ids), cudaMemcpyHostToDevice);
            int offset = group_ids[group];
            int size = group_ids[group + 3];
            int blocks = (size + threads_per_block - 1) / threads_per_block;
            update_particles_combined<<<blocks, threads_per_block>>>(
                device_particles, all_particles.size(), device_matrix, offset, size, device_group_ids
            );
        }

        cudaDeviceSynchronize();
        cudaMemcpy(all_particles.data(), device_particles, all_particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);

        window.clear();
        sf::VertexArray vertices(sf::Points, all_particles.size());

        for (size_t i = 0; i < all_particles.size(); ++i) {
            int brightness = static_cast<int>(64 + 191 * all_particles[i].energy);

            vertices[i].position = sf::Vector2f(all_particles[i].x, all_particles[i].y);
            vertices[i].color = sf::Color(
                std::min(255, all_particles[i].r + brightness / 3),
                std::min(255, all_particles[i].g + brightness / 3),
                std::min(255, all_particles[i].b + brightness / 3),
                255
            );
        }

        window.draw(vertices);
        window.display();
    }

    cudaFree(device_particles);
    cudaFree(device_matrix);
    cudaFree(device_group_ids);
    return 0;
}
