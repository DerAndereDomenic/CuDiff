#include <iostream>
#include <CuDiff/CuDiff.h>
#include <CuDiff/ext/glm/Traits.h>
#include <CuDiff/ext/glm/Function.h>

__global__ void test()
{
    {
        auto [x, y, z] = CuDiff::make_variables<3>(1.0f, -2.0f, 5.0f);

        auto f = (3.0f * x + (-2.0f - y) * y) / z - 3.0f;
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

    {
        auto [v, w, a] = CuDiff::make_variables<5>(glm::vec2(1.0f, 2.0f), glm::vec2(3.0f, 4.0f), -4.0f);

        auto f = a * v + w * w + (3.0f + a);
        f      = CuDiff::dot(f, w) * f;

        auto f_ = CuDiff::length(f);

        printf("df/dvx = %f\n", f_.derivative(0));

        printf("df/dvy = %f\n", f_.derivative(1));

        printf("df/dwx = %f\n", f_.derivative(2));

        printf("df/dwy = %f\n", f_.derivative(3));

        printf("df/da = %f\n", f_.derivative(4));
    }

    {
        auto [x] = CuDiff::make_variables<2>(glm::vec2(-1.0f, 2.0f));

        auto y = CuDiff::normalize(x);

        printf("%f %f\n%f %f\n", y.derivative(0).x, y.derivative(1).x, y.derivative(0).y, y.derivative(1).y);
    }

    {
        auto [v] = CuDiff::make_variables<2>(glm::vec2(-1.0f, 2.0f));
        auto w   = glm::vec2(3.0f, 4.0f);

        auto y = CuDiff::dot(v, w);

        printf("%f %f\n", y.derivative(0), y.derivative(1));
    }

    {
        auto [v] = CuDiff::make_variables<2>(normalize(glm::vec2(-1.0f, 2.0f)));
        auto w   = normalize(glm::vec2(3.0f, 4.0f));

        auto y = CuDiff::reflect(v, w);

        printf("%f %f\n", y.derivative(0).x, y.derivative(1).x);
        printf("%f %f\n", y.derivative(0).y, y.derivative(1).y);
    }

    {
        auto [v, w] = CuDiff::make_variables<4>(normalize(glm::vec2(-1.0f, 2.0f)), normalize(glm::vec2(3.0f, 4.0f)));

        auto y = CuDiff::reflect(v, w);

        printf("%f %f %f %f\n", y.derivative(0).x, y.derivative(1).x, y.derivative(2).x, y.derivative(3).x);
        printf("%f %f %f %f\n", y.derivative(0).y, y.derivative(1).y, y.derivative(2).y, y.derivative(3).y);
    }

    {
        auto [v, w] = CuDiff::make_variables<4>(normalize(glm::vec2(1.0f, -2.0f)), normalize(glm::vec2(0.0f, 1.0f)));

        auto y = CuDiff::refract(v, w, 1.5f);

        printf("%f %f %f %f\n", y.derivative(0).x, y.derivative(1).x, y.derivative(2).x, y.derivative(3).x);
        printf("%f %f %f %f\n", y.derivative(0).y, y.derivative(1).y, y.derivative(2).y, y.derivative(3).y);
    }

    {
        auto [v] = CuDiff::make_variables<2>(normalize(glm::vec2(1.0f, -2.0f)));
        auto w   = normalize(glm::vec2(0.0f, 1.0f));
        auto y   = CuDiff::refract(v, w, 1.5f);

        printf("%f %f\n", y.derivative(0).x, y.derivative(1).x);
        printf("%f %f\n", y.derivative(0).y, y.derivative(1).y);
    }

    {
        auto [v, eta] = CuDiff::make_variables<3>(normalize(glm::vec2(1.0f, -2.0f)), 1.5f);
        auto w        = normalize(glm::vec2(0.0f, 1.0f));
        auto y        = CuDiff::refract(v, w, eta);

        printf("%f %f %f\n", y.derivative(0).x, y.derivative(1).x, y.derivative(2).x);
        printf("%f %f %f\n", y.derivative(0).y, y.derivative(1).y, y.derivative(2).y);
    }
}

int main()
{
    printf("Device:\n");
    test<<<1, 1>>>();
    cudaDeviceSynchronize();

    printf("Host:\n");
    auto [x, y, z] = CuDiff::make_variables<3>(1.0f, -2.0f, 5.0f);

    auto f = (3.0f * x + (-2.0f - y) * y) / z - 3.0f;
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

    // auto [v] = CuDiff::make_variables<3>(glm::vec3(1.0f, 2.0f, 3.0f));

    // printf("dx/dx = %f\n", v.derivative(0).x);
    // printf("dy/dx = %f\n", v.derivative(0).y);
    // printf("dz/dx = %f\n", v.derivative(0).z);

    // printf("dx/dy = %f\n", v.derivative(1).x);
    // printf("dy/dy = %f\n", v.derivative(1).y);
    // printf("dz/dy = %f\n", v.derivative(1).z);

    // printf("dx/dz = %f\n", v.derivative(2).x);
    // printf("dy/dz = %f\n", v.derivative(2).y);
    // printf("dz/dz = %f\n", v.derivative(2).z);
}