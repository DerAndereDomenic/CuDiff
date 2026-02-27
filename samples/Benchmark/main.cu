#include <iostream>
#include <CuDiff/CuDiff.h>
#include <CuDiff/ext/glm.h>

#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <string>

template<typename T>
__device__ __forceinline__ void do_not_optimize_away(const T& value)
{
    asm volatile("" ::"l"(&value) : "memory");
}

class Timer
{
public:
    Timer() { reset(); }

    void reset() { _start = std::chrono::high_resolution_clock::now(); }

    float elapsedSeconds() const
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - _start)
                   .count() *
               0.001f * 0.001f * 0.001f;
    }

    float elapsedMillis() const { return elapsedSeconds() * 1000.0f; }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;
};

__global__ void trace_kernel_derivative(int width, int height, float* derivative_buffer)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(auto tid = id; tid < width * height; tid += num_threads)
    {
        if(tid >= width * height) return;

        int pixel_x = tid % (int)width;
        int pixel_y = tid / (int)width;

        float u_ = (((float)pixel_x + 0.5f) / ((float)width) - 0.5f) * 2.0f;
        float v_ = (((float)pixel_y + 0.5f) / ((float)height) - 0.5f) * 2.0f;

        glm::vec3 direction_ = glm::normalize(u_ * glm::vec3(1, 0, 0) + v_ * glm::vec3(0, 1, 0) + glm::vec3(0, 0, 1));
        float theta_         = glm::acos(glm::clamp(direction_.y, -1.0f, 1.0f));
        float phi_           = std::atan2(direction_.z, direction_.x);

        auto [uv, angles] = CuDiff::make_variables<4>(glm::vec2(u_, v_), glm::vec2(theta_, phi_));

        auto [theta, phi] = CuDiff::unwrap(angles);

        auto sinTheta = CuDiff::sin(theta);

        auto x = sinTheta * CuDiff::cos(phi);
        auto y = CuDiff::cos(theta);
        auto z = sinTheta * CuDiff::sin(phi);

        auto dir    = CuDiff::wrap(x, y, z);
        auto origin = 0.01f * dir;

        glm::vec3 normal = glm::vec3(0, 0, -1);
        glm::vec3 point  = glm::vec3(0, 0, 5);

        auto t = (CuDiff::dot(point - origin, normal)) / (CuDiff::dot(dir, normal));

        auto xo = origin + t * dir;

        auto [u_out, v_out, _] = CuDiff::unwrap(xo);

        auto wo = CuDiff::reflect(dir, normal);

        auto [dx, dy, dz] = CuDiff::unwrap(wo);

        auto theta_out = CuDiff::acos(CuDiff::clamp(dy, -1.0f, 1.0f));
        auto phi_out   = CuDiff::atan2(dz, dx);

        float du_outdu     = u_out.derivative(0);
        float du_outdv     = u_out.derivative(1);
        float du_outdtheta = u_out.derivative(2);
        float du_outdphi   = u_out.derivative(3);

        float dv_outdu     = v_out.derivative(0);
        float dv_outdv     = v_out.derivative(1);
        float dv_outdtheta = v_out.derivative(2);
        float dv_outdphi   = v_out.derivative(3);

        float dtheta_out_outdu     = theta_out.derivative(0);
        float dtheta_out_outdv     = theta_out.derivative(1);
        float dtheta_out_outdtheta = theta_out.derivative(2);
        float dtheta_out_outdphi   = theta_out.derivative(3);

        float dphi_out_outdu     = phi_out.derivative(0);
        float dphi_out_outdv     = phi_out.derivative(1);
        float dphi_out_outdtheta = phi_out.derivative(2);
        float dphi_out_outdphi   = phi_out.derivative(3);

        derivative_buffer[pixel_x + width * pixel_y + width * height * 0] = du_outdu;
        derivative_buffer[pixel_x + width * pixel_y + width * height * 1] = du_outdv;
        derivative_buffer[pixel_x + width * pixel_y + width * height * 2] = du_outdtheta;
        derivative_buffer[pixel_x + width * pixel_y + width * height * 3] = du_outdphi;

        derivative_buffer[pixel_x + width * pixel_y + width * height * 4] = dv_outdu;
        derivative_buffer[pixel_x + width * pixel_y + width * height * 5] = dv_outdv;
        derivative_buffer[pixel_x + width * pixel_y + width * height * 6] = dv_outdtheta;
        derivative_buffer[pixel_x + width * pixel_y + width * height * 7] = dv_outdphi;

        derivative_buffer[pixel_x + width * pixel_y + width * height * 8]  = dtheta_out_outdu;
        derivative_buffer[pixel_x + width * pixel_y + width * height * 9]  = dtheta_out_outdv;
        derivative_buffer[pixel_x + width * pixel_y + width * height * 10] = dtheta_out_outdtheta;
        derivative_buffer[pixel_x + width * pixel_y + width * height * 11] = dtheta_out_outdphi;

        derivative_buffer[pixel_x + width * pixel_y + width * height * 12] = dphi_out_outdu;
        derivative_buffer[pixel_x + width * pixel_y + width * height * 13] = dphi_out_outdv;
        derivative_buffer[pixel_x + width * pixel_y + width * height * 14] = dphi_out_outdtheta;
        derivative_buffer[pixel_x + width * pixel_y + width * height * 15] = dphi_out_outdphi;


        do_not_optimize_away(du_outdu);
        do_not_optimize_away(du_outdv);
        do_not_optimize_away(du_outdtheta);
        do_not_optimize_away(du_outdphi);

        do_not_optimize_away(dv_outdu);
        do_not_optimize_away(dv_outdv);
        do_not_optimize_away(dv_outdtheta);
        do_not_optimize_away(dv_outdphi);

        do_not_optimize_away(dtheta_out_outdu);
        do_not_optimize_away(dtheta_out_outdv);
        do_not_optimize_away(dtheta_out_outdtheta);
        do_not_optimize_away(dtheta_out_outdphi);

        do_not_optimize_away(dphi_out_outdu);
        do_not_optimize_away(dphi_out_outdv);
        do_not_optimize_away(dphi_out_outdtheta);
        do_not_optimize_away(dphi_out_outdphi);
    }
}

__global__ void trace_kernel_noderivative(int width, int height)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(auto tid = id; tid < width * height; tid += num_threads)
    {
        if(tid >= width * height) return;

        int pixel_x = tid % (int)width;
        int pixel_y = tid / (int)width;

        float u = (((float)pixel_x + 0.5f) / ((float)width) - 0.5f) * 2.0f;
        float v = (((float)pixel_y + 0.5f) / ((float)height) - 0.5f) * 2.0f;

        glm::vec3 direction_ = glm::normalize(u * glm::vec3(1, 0, 0) + v * glm::vec3(0, 1, 0) + glm::vec3(0, 0, 1));
        float theta          = glm::acos(glm::clamp(direction_.y, -1.0f, 1.0f));
        float phi            = std::atan2(direction_.z, direction_.x);

        auto sinTheta = glm::sin(theta);

        auto x = sinTheta * glm::cos(phi);
        auto y = glm::cos(theta);
        auto z = sinTheta * glm::sin(phi);

        auto dir    = glm::vec3(x, y, z);
        auto origin = 0.01f * dir;

        glm::vec3 normal = glm::vec3(0, 0, -1);
        glm::vec3 point  = glm::vec3(0, 0, 5);

        auto t = (glm::dot(point - origin, normal)) / (glm::dot(dir, normal));

        auto xo = origin + t * dir;

        auto u_out = xo.x;
        auto v_out = xo.y;

        auto wo = glm::reflect(dir, normal);

        auto dx = wo.x;
        auto dy = wo.y;
        auto dz = wo.z;

        auto theta_out = glm::acos(glm::clamp(dy, -1.0f, 1.0f));
        auto phi_out   = std::atan2(dz, dx);

        do_not_optimize_away(u_out);
        do_not_optimize_away(v_out);
        do_not_optimize_away(theta_out);
        do_not_optimize_away(phi_out);
    }
}

struct Stats
{
    float min;
    float max;
    float average;
    float stddev;
    float median;
};

Stats compute_stats(const std::vector<float>& data)
{
    Stats s {};

    if(data.empty()) return s;

    // --- Min & Max ---
    auto [minIt, maxIt] = std::minmax_element(data.begin(), data.end());
    s.min               = *minIt;
    s.max               = *maxIt;

    // --- Average ---
    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    s.average = sum / data.size();

    // --- Standard Deviation ---
    float sq_sum = std::accumulate(data.begin(),
                                   data.end(),
                                   0.0f,
                                   [&](float acc, float v)
                                   {
                                       float diff = v - s.average;
                                       return acc + diff * diff;
                                   });
    s.stddev     = std::sqrt(sq_sum / data.size());    // population stddev

    // --- Median ---
    std::vector<float> temp = data;    // copy to avoid modifying original
    size_t n                = temp.size();
    auto mid                = temp.begin() + n / 2;

    std::nth_element(temp.begin(), mid, temp.end());
    if(n % 2 == 1)
    {
        s.median = *mid;
    }
    else
    {
        // Need the other middle value
        auto mid2 = temp.begin() + (n / 2 - 1);
        std::nth_element(temp.begin(), mid2, temp.end());
        s.median = (*mid + *mid2) / 2.0f;
    }

    return s;
}

void pring_stats_header()
{
    using std::setw;
    // Column widths
    const int wName = 25;
    const int wVal  = 12;

    std::cout << std::left << setw(wName) << "Name" << std::right << setw(wVal) << "Min" << std::right << setw(wVal)
              << "Max" << std::right << setw(wVal) << "Avg(Std)" << std::right << setw(wVal) << "Median"
              << "\n";
}

void print_stats_row(std::string_view name, const Stats& s)
{
    using std::setw;

    // Column widths
    const int wName = 25;
    const int wVal  = 12;

    std::cout << std::left << setw(wName) << name << std::right << setw(wVal) << std::fixed << std::setprecision(2)
              << s.min << std::right << setw(wVal) << s.max << std::right << setw(wVal) << s.average << "(" << s.stddev
              << ")" << std::right << setw(wVal) << s.median << "\n";
}

template<typename Function>
float measure_execution_time(Function f)
{
    Timer timer;
    f();
    return timer.elapsedMillis();
}

template<typename Function>
void benchmark(std::string_view name, Function f, int num_executions = 1000)
{
    // warm up
    for(int i = 0; i < 10; ++i)
        f();

    std::vector<float> timings(num_executions);
    for(int i = 0; i < num_executions; ++i)
    {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        timings[i] = measure_execution_time(f);
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }

    Stats stats = compute_stats(timings);
    print_stats_row(name, stats);
}

int main()
{
    int width  = 2000;
    int height = 2000;
    float* derivative_buffer;
    cudaMalloc((void**)&derivative_buffer, sizeof(float) * 16 * width * height);
    auto device_derivative = [&]()
    {
        int num_threads           = width * height;
        int num_threads_per_block = 128;
        int num_blocks            = num_threads / num_threads_per_block + 1;

        Timer timer;
        trace_kernel_derivative<<<num_blocks, num_threads_per_block>>>(width, height, derivative_buffer);
        cudaDeviceSynchronize();
    };

    auto device_noderivative = [&]()
    {
        int num_threads           = width * height;
        int num_threads_per_block = 128;
        int num_blocks            = num_threads / num_threads_per_block + 1;

        Timer timer;
        trace_kernel_noderivative<<<num_blocks, num_threads_per_block>>>(width, height);
        cudaDeviceSynchronize();
    };

    pring_stats_header();
    benchmark("Device w/o derivative", device_noderivative);
    benchmark("Device w/ derivative", device_derivative);

    cudaFree((void*)derivative_buffer);
}