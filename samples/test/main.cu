#include <iostream>
#include <CuDiff/CuDiff.h>

__global__ void test()
{
    auto [x, y, z] = CuDiff::make_variables<3>(1.0f, -2.0f, 5.0f);

    auto f = (3.0f * x + (-2.0 - y) * y) / z - 3.0f;
    f      = -f * f;

    f = CuDiff::sin(f);
    f = CuDiff::exp(f);
    f = CuDiff::cos(f + z);
    f = CuDiff::sqrt(f);
    f = CuDiff::log(f + 42.0f);
    f = CuDiff::pow(f, x * x - y + CuDiff::pow(z, -3.1f));
    f = CuDiff::clamp(f, 50.0f, 60.0f);

    printf("f = %f\n", f.val());
    printf("df/dx = %f\n", f.derivative(0));
    printf("df/dy = %f\n", f.derivative(1));
    printf("df/dz = %f\n", f.derivative(2));
}

int main()
{
    test<<<1, 1>>>();
    cudaDeviceSynchronize();

    auto [x, y, z] = CuDiff::make_variables<3>(1.0f, -2.0f, 5.0f);

    auto f = (3.0f * x + (-2.0 - y) * y) / z - 3.0f;
    f      = -f * f;

    f = CuDiff::sin(f);
    f = CuDiff::exp(f);
    f = CuDiff::cos(f + z);
    f = CuDiff::sqrt(f);
    f = CuDiff::log(f + 42.0f);

    printf("f = %f\n", f.val());
    printf("df/dx = %f\n", f.derivative(0));
    printf("df/dy = %f\n", f.derivative(1));
    printf("df/dz = %f\n", f.derivative(2));
}