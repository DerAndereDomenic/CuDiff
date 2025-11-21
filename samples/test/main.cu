#include <iostream>
#include <random>
#include <CuDiff/CuDiff.h>
#include <CuDiff/ext/glm.h>

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
        f_      = CuDiff::max(f_, 0.0f);

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
        auto [v] = CuDiff::make_variables<2>(normalize(glm::vec2(1.0f, -2.0f)));
        auto y   = CuDiff::clamp(v, glm::vec2(0), glm::vec2(1));

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

    {
        auto [x, y] = CuDiff::make_variables<2>(-1.0f, 2.0f);

        CuDiff::Dual<2, glm::vec2> v = CuDiff::wrap(x, y);

        v = CuDiff::normalize(v);

        auto [v1, v2] = CuDiff::unwrap(v);

        printf("%f %f\n", v1.derivative(0), v1.derivative(1));
        printf("%f %f\n", v2.derivative(0), v2.derivative(1));
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

    // Interpolation test
    {
        constexpr int width  = 2;
        float texture[width] = {1.0f, 0.2f};
        auto [u]             = CuDiff::make_variables<3>(0.6f);

        auto fx = u * (width - 1);
        int x0  = glm::floor(fx);
        int x1  = glm::min(x0 + 1, (int)width - 1);
        printf("%i %i\n", x0, x1);

        auto tx = fx - x0;

        CuDiff::Dual<3> c0 = texture[x0];
        c0.setDerivative(1, 1.0f);
        CuDiff::Dual<3> c1 = texture[x1];
        c1.setDerivative(2, 1.0f);

        auto val = (1.0f - tx) * c0 + tx * c1;

        printf("%f\n", val.val());
        printf("%f\n", val.derivative(0));
        printf("%f\n", val.derivative(1));
        printf("%f\n", val.derivative(2));
    }

    {
        glm::vec3 v1 = glm::vec3(1, 2, 3);
        glm::vec3 v2 = glm::vec3(4, 5, 6);
        glm::vec3 v3 = glm::vec3(7, 8, 9);

        glm::mat3 M = glm::mat3(v1, v2, v3);

        glm::vec3 v = glm::vec3(1, 1, 1);

        auto w = v * M;
        std::cout << w.x << " " << w.y << " " << w.z << "\n";
    }

    {
        glm::vec3 ray_dir = glm::normalize(glm::vec3(1, -2, -1.5));

        auto f = [](const glm::vec3& ray_dir)
        {
            glm::vec3 ray_origin = glm::vec3(0.3, 0.4, -0.1);

            glm::vec3 p0 = glm::vec3(1, -3, -1);
            glm::vec3 p1 = glm::vec3(2, -1, -2);
            glm::vec3 p2 = glm::vec3(0, 0, -4);

            glm::vec3 v0     = p1 - p0;
            glm::vec3 v1     = p2 - p0;
            glm::vec3 normal = glm::normalize(glm::cross(v0, v1));

            auto [xi, wi] = CuDiff::make_variables<6>(ray_origin, ray_dir);

            auto tmax = (CuDiff::dot((p0 - xi), normal)) / (CuDiff::dot(wi, normal));
            auto xo   = xi + tmax * wi;
            return xo;
        };

        auto xo = f(ray_dir);

        printf("%f %f %f\n", xo.val().x, xo.val().y, xo.val().z);
        printf("--------------\n");
        printf("%f %f %f\n", xo.derivative(3).x, xo.derivative(4).x, xo.derivative(5).x);
        printf("%f %f %f\n", xo.derivative(3).y, xo.derivative(4).y, xo.derivative(5).y);
        printf("%f %f %f\n", xo.derivative(3).z, xo.derivative(4).z, xo.derivative(5).z);

        float h = 0.0001f;
        auto f1 = f(glm::normalize(ray_dir + glm::vec3(h, 0, 0)));
        auto f2 = f(glm::normalize(ray_dir - glm::vec3(h, 0, 0)));

        glm::vec3 diff = (f1.val() - f2.val()) / (2.0f * h);
        printf("%f %f %f\n", diff.x, diff.y, diff.z);
    }

    {
        auto [x] = CuDiff::make_variables<3>(glm::vec3(1, 1, 1));

        glm::mat3 M = glm::mat3(glm::vec3(1, 2, 3), glm::vec3(4, 5, 6), glm::vec3(7, 8, 9));

        auto y = M * x;

        printf("%f %f %f\n", y.derivative(0).x, y.derivative(1).x, y.derivative(2).x);
        printf("%f %f %f\n", y.derivative(0).y, y.derivative(1).y, y.derivative(2).y);
        printf("%f %f %f\n", y.derivative(0).z, y.derivative(1).z, y.derivative(2).z);

        printf("%f %f %f\n", M[0].x, M[1].x, M[2].x);
        printf("%f %f %f\n", M[0].y, M[1].y, M[2].y);
        printf("%f %f %f\n", M[0].z, M[1].z, M[2].z);
    }
}