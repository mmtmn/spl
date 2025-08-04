// 3d.cu
// compile: nvcc -o 3d 3d.cu -lGL -lglfw -lGLU -ldl -lX11 -lpthread -lXrandr -lXi -lXxf86vm -lXinerama -lXcursor

#include <GLFW/glfw3.h>
#include <GL/glu.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>

const int width = 512;
const int height = 512;
const int depth = 512;
const int total_groups = 5;
const int total_particles = 100000;
const int particles_per_group = total_particles / total_groups;
const int cell_size = 32;
const int grid_x = width / cell_size;
const int grid_y = height / cell_size;
const int grid_z = depth / cell_size;

struct Particle {
    float x, y, z, vx, vy, vz;
    unsigned char r, g, b, a;
};

__device__ int clamp(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}

__global__ void update_particles_spatial_3d(
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

    Particle& p1 = particles[idx];
    float fx = 0, fy = 0, fz = 0;

    int group_idx = group_ids[idx];

    int cell_x = clamp((int)(p1.x / cell_size), 0, grid_x - 1);
    int cell_y = clamp((int)(p1.y / cell_size), 0, grid_y - 1);
    int cell_z = clamp((int)(p1.z / cell_size), 0, grid_z - 1);

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cell_x + dx;
                int ny = cell_y + dy;
                int nz = cell_z + dz;
                if (nx < 0 || ny < 0 || nz < 0 || nx >= grid_x || ny >= grid_y || nz >= grid_z) continue;
                int cell_id = (nz * grid_y * grid_x) + (ny * grid_x) + nx;

                int start = bin_starts[cell_id];
                int count = bin_counts[cell_id];

                for (int i = 0; i < count; ++i) {
                    int j = bin_particles[start + i];
                    if (j == idx) continue;

                    Particle p2 = particles[j];
                    float dx = p1.x - p2.x;
                    float dy = p1.y - p2.y;
                    float dz = p1.z - p2.z;
                    float dist2 = dx * dx + dy * dy + dz * dz;

                    if (dist2 > 0 && dist2 < 6400) {
                        float dist = sqrtf(dist2);
                        int g2 = group_ids[j];
                        float interaction = interaction_matrix[group_idx * total_groups + g2];
                        float force = interaction / dist;
                        fx += force * dx;
                        fy += force * dy;
                        fz += force * dz;
                    }
                }
            }
        }
    }

    p1.vx = (p1.vx + fx) * 0.5f;
    p1.vy = (p1.vy + fy) * 0.5f;
    p1.vz = (p1.vz + fz) * 0.5f;

    p1.x += p1.vx;
    p1.y += p1.vy;
    p1.z += p1.vz;

    if (p1.x <= 0 || p1.x >= width) p1.vx *= -1;
    if (p1.y <= 0 || p1.y >= height) p1.vy *= -1;
    if (p1.z <= 0 || p1.z >= depth) p1.vz *= -1;

    p1.x = fminf(fmaxf(p1.x, 0), width);
    p1.y = fminf(fmaxf(p1.y, 0), height);
    p1.z = fminf(fmaxf(p1.z, 0), depth);

    shared_score[threadIdx.x] = expf(-sqrtf(p1.vx * p1.vx + p1.vy * p1.vy + p1.vz * p1.vz) * 0.1f);
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < blockDim.x; ++i) total += shared_score[i];
        atomicAdd(out_fitness, total);
    }
}

void create_particles(std::vector<Particle>& particles, int number, int group_id) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis_x(50, width - 50);
    std::uniform_real_distribution<float> dis_y(50, height - 50);
    std::uniform_real_distribution<float> dis_z(50, depth - 50);
    std::uniform_real_distribution<float> dis_v(-1.0f, 1.0f);
    std::uniform_int_distribution<int> dis_c(64, 255);

    for (int i = 0; i < number; ++i) {
        Particle p;
        p.x = dis_x(gen);
        p.y = dis_y(gen);
        p.z = dis_z(gen);
        p.vx = dis_v(gen);
        p.vy = dis_v(gen);
        p.vz = dis_v(gen);
        p.r = dis_c(gen);
        p.g = dis_c(gen);
        p.b = dis_c(gen);
        p.a = 255;
        particles.push_back(p);
    }
}

int get_cell_id(float x, float y, float z) {
    int cx = std::min((int)(x / cell_size), grid_x - 1);
    int cy = std::min((int)(y / cell_size), grid_y - 1);
    int cz = std::min((int)(z / cell_size), grid_z - 1);
    return cz * grid_y * grid_x + cy * grid_x + cx;
}

float yaw = -90.0f;
float pitch = 0.0f;
float frontX = 0.0f, frontY = 0.0f, frontZ = -1.0f;


void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    static bool firstMouse = true;
    static float lastX = 512.0f, lastY = 384.0f;
    static float sensitivity = 0.1f;

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    if (firstMouse) {
        firstMouse = false;
        return;
    }

    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
}


int main() {
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(2560, 1440, "genesis", NULL, NULL);
    if (!window) return -1;
    glfwMakeContextCurrent(window);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouse_callback);

    glEnable(GL_POINT_SMOOTH);
    glPointSize(2.0f);
    glEnable(GL_DEPTH_TEST);

    float camX = 256, camY = 256, camZ = 1024;
    float speed = 4.0f;


    std::vector<Particle> all_particles;
    std::vector<int> group_for_particle;
    for (int i = 0; i < total_groups; ++i) {
        create_particles(all_particles, particles_per_group, i);
        for (int j = 0; j < particles_per_group; ++j) group_for_particle.push_back(i);
    }

    std::vector<float> interaction_matrix(total_groups * total_groups, 0.0f);
    for (int g = 0; g < total_groups; ++g)
        interaction_matrix[g * total_groups + g] = 0.5f;

    Particle* device_particles;
    cudaMalloc(&device_particles, all_particles.size() * sizeof(Particle));
    cudaMemcpy(device_particles, all_particles.data(), all_particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);

    float* device_matrix;
    cudaMalloc(&device_matrix, interaction_matrix.size() * sizeof(float));
    cudaMemcpy(device_matrix, interaction_matrix.data(), interaction_matrix.size() * sizeof(float), cudaMemcpyHostToDevice);

    float* device_fitness;
    cudaMalloc(&device_fitness, sizeof(float));

    int grid_total = grid_x * grid_y * grid_z;
    std::vector<int> bin_starts(grid_total), bin_counts(grid_total), bin_particles(all_particles.size());

    int* d_bin_starts;
    int* d_bin_counts;
    int* d_bin_particles;
    cudaMalloc(&d_bin_starts, bin_starts.size() * sizeof(int));
    cudaMalloc(&d_bin_counts, bin_counts.size() * sizeof(int));
    cudaMalloc(&d_bin_particles, bin_particles.size() * sizeof(int));

    std::vector<int> group_ids(group_for_particle);
    int* d_group_ids;
    cudaMalloc(&d_group_ids, group_ids.size() * sizeof(int));
    cudaMemcpy(d_group_ids, group_ids.data(), group_ids.size() * sizeof(int), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (total_particles + threads_per_block - 1) / threads_per_block;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        cudaMemcpy(all_particles.data(), device_particles, all_particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);
        frontX = cosf(yaw * M_PI / 180.0f) * cosf(pitch * M_PI / 180.0f);
        frontY = sinf(pitch * M_PI / 180.0f);
        frontZ = sinf(yaw * M_PI / 180.0f) * cosf(pitch * M_PI / 180.0f);

        float len = sqrtf(frontX * frontX + frontY * frontY + frontZ * frontZ);
        float normX = frontX / len;
        float normY = frontY / len;
        float normZ = frontZ / len;

        float upX = 0.0f, upY = 1.0f, upZ = 0.0f;
        float rightX = upY * frontZ - upZ * frontY;
        float rightY = upZ * frontX - upX * frontZ;
        float rightZ = upX * frontY - upY * frontX;

        float rightLen = sqrtf(rightX * rightX + rightY * rightY + rightZ * rightZ);
        rightX /= rightLen;
        rightY /= rightLen;
        rightZ /= rightLen;




        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            camX += normX * speed;
            camY += normY * speed;
            camZ += normZ * speed;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            camX -= normX * speed;
            camY -= normY * speed;
            camZ -= normZ * speed;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            camX -= rightX * speed;
            camY -= rightY * speed;
            camZ -= rightZ * speed;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            camX += rightX * speed;
            camY += rightY * speed;
            camZ += rightZ * speed;
        }


        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) camY += speed;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) camY -= speed;

        std::fill(bin_counts.begin(), bin_counts.end(), 0);
        for (int i = 0; i < total_particles; ++i) {
            int cell_id = get_cell_id(all_particles[i].x, all_particles[i].y, all_particles[i].z);
            bin_counts[cell_id]++;
        }

        bin_starts[0] = 0;
        for (size_t i = 1; i < bin_starts.size(); ++i)
            bin_starts[i] = bin_starts[i - 1] + bin_counts[i - 1];

        std::fill(bin_counts.begin(), bin_counts.end(), 0);
        for (int i = 0; i < total_particles; ++i) {
            int cell_id = get_cell_id(all_particles[i].x, all_particles[i].y, all_particles[i].z);
            int index = bin_starts[cell_id] + bin_counts[cell_id]++;
            bin_particles[index] = i;
        }

        cudaMemcpy(d_bin_particles, bin_particles.data(), total_particles * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bin_counts, bin_counts.data(), bin_counts.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bin_starts, bin_starts.data(), bin_starts.size() * sizeof(int), cudaMemcpyHostToDevice);

        update_particles_spatial_3d<<<blocks, threads_per_block>>>(
            device_particles, d_bin_starts, d_bin_counts, d_bin_particles,
            total_particles, device_matrix, d_group_ids, device_fitness, total_groups
        );
        cudaDeviceSynchronize();
        cudaMemcpy(all_particles.data(), device_particles, all_particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(60.0, 1024.0 / 768.0, 0.1, 5000.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        float frontX = cos(yaw * M_PI / 180.0f) * cos(pitch * M_PI / 180.0f);
        float frontY = sin(pitch * M_PI / 180.0f);
        float frontZ = sin(yaw * M_PI / 180.0f) * cos(pitch * M_PI / 180.0f);
        gluLookAt(camX, camY, camZ, camX + frontX, camY + frontY, camZ + frontZ, 0, 1, 0);



        glBegin(GL_POINTS);
        for (const auto& p : all_particles) {
            glColor3ub(p.r, p.g, p.b);
            glVertex3f(p.x - width / 2.0f, p.y - height / 2.0f, p.z - depth / 2.0f);
        }
        glEnd();

        glfwSwapBuffers(window);
    }

    cudaFree(device_particles);
    cudaFree(device_matrix);
    cudaFree(device_fitness);
    cudaFree(d_bin_particles);
    cudaFree(d_bin_counts);
    cudaFree(d_bin_starts);
    cudaFree(d_group_ids);
    glfwTerminate();
    return 0;
}
