#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <CuDiff/CuDiff.h>
#include <CuDiff/ext/glm.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace
{
constexpr float kTol = 1e-4f;

#define CUDA_CHECK(expr)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cuda_error = (expr);                                                                               \
        ASSERT_EQ(cuda_error, cudaSuccess) << cudaGetErrorString(cuda_error);                                          \
    } while(false)

template<typename T>
void expect_near_scalar(T actual, T expected, float tol = kTol)
{
    EXPECT_NEAR(static_cast<double>(actual), static_cast<double>(expected), tol);
}

template<typename T, glm::length_t N, glm::qualifier P>
void expect_vec_near(const glm::vec<N, T, P>& actual, const glm::vec<N, T, P>& expected, float tol = kTol)
{
    for(glm::length_t i = 0; i < N; ++i)
    {
        EXPECT_NEAR(static_cast<double>(actual[i]), static_cast<double>(expected[i]), tol);
    }
}

template<int N, typename T>
void expect_dual_near(const CuDiff::Dual<N, T>& actual,
                      const T& expected_value,
                      const std::array<T, N>& expected_derivatives,
                      float tol = kTol)
{
    expect_near_scalar(actual.val(), expected_value, tol);
    for(int i = 0; i < N; ++i)
    {
        expect_near_scalar(actual.derivative(i), expected_derivatives[static_cast<std::size_t>(i)], tol);
    }
}

template<int N, glm::length_t L, typename Q, glm::qualifier P>
void expect_dual_near(const CuDiff::Dual<N, glm::vec<L, Q, P>>& actual,
                      const glm::vec<L, Q, P>& expected_value,
                      const std::array<glm::vec<L, Q, P>, N>& expected_derivatives,
                      float tol = kTol)
{
    expect_vec_near(actual.val(), expected_value, tol);
    for(int i = 0; i < N; ++i)
    {
        expect_vec_near(actual.derivative(i), expected_derivatives[static_cast<std::size_t>(i)], tol);
    }
}

template<typename T, typename Fn>
T finite_difference_scalar(T x, Fn&& fn, T h = T(1e-4))
{
    return (fn(x + h) - fn(x - h)) / (T(2) * h);
}

template<typename T>
glm::vec<2, T, glm::defaultp> normalize_jacobian_column(const glm::vec<2, T, glm::defaultp>& v, int column)
{
    const T len      = glm::length(v);
    const T inv_len  = T(1) / len;
    const T inv_len3 = inv_len / (len * len);
    glm::mat<2, 2, T, glm::defaultp> jac(T(0));
    for(int row = 0; row < 2; ++row)
    {
        for(int col = 0; col < 2; ++col)
        {
            jac[col][row] = (row == col ? inv_len : T(0)) - v[row] * v[col] * inv_len3;
        }
    }
    return glm::vec<2, T, glm::defaultp>(jac[column][0], jac[column][1]);
}

template<typename T>
glm::vec<2, T, glm::defaultp> reflect_jacobian_column(const glm::vec<2, T, glm::defaultp>& n, int column)
{
    glm::mat<2, 2, T, glm::defaultp> jac(T(0));
    for(int row = 0; row < 2; ++row)
    {
        for(int col = 0; col < 2; ++col)
        {
            jac[col][row] = (row == col ? T(1) : T(0)) - T(2) * n[row] * n[col];
        }
    }
    return glm::vec<2, T, glm::defaultp>(jac[column][0], jac[column][1]);
}

template<typename T>
CUDIFF_HOSTDEVICE auto scalar_composite(const T& x, const T& y, const T& z)
{
    auto f = (T(3) * x + (T(-2) - y) * y) / z - T(3);
    f      = -f * f;
    f      = CuDiff::sin(f);
    f      = CuDiff::exp(f);
    f      = CuDiff::cos(f + z);
    f      = CuDiff::sqrt(f + T(2));
    f      = CuDiff::log(f + T(42));
    f      = CuDiff::pow(f, x * x - y + CuDiff::pow(z, T(-3.1)));
    return f;
}

struct ScalarParityResult
{
    float value;
    float derivatives[3];
};

__global__ void scalar_parity_kernel(ScalarParityResult* result)
{
    auto [x, y, z] = CuDiff::make_variables<3>(1.0f, -2.0f, 5.0f);
    auto f         = scalar_composite(x, y, z);
    result->value  = f.val();
    for(int i = 0; i < 3; ++i)
    {
        result->derivatives[i] = f.derivative(i);
    }
}

struct VecParityResult
{
    float value[2];
    float derivatives[2][2];
};

__global__ void vector_parity_kernel(VecParityResult* result)
{
    auto [v] = CuDiff::make_variables<2>(glm::vec2(3.0f, 4.0f));
    auto n   = CuDiff::normalize(v);
    for(int i = 0; i < 2; ++i)
    {
        result->value[i] = n.val()[i];
        for(int j = 0; j < 2; ++j)
        {
            result->derivatives[i][j] = n.derivative(j)[i];
        }
    }
}

template<typename T>
T numeric_scalar_function(T x)
{
    return CuDiff::sin(x) + CuDiff::cos(x) + CuDiff::exp(x) + CuDiff::log(x + T(2)) + CuDiff::sqrt(x + T(3)) +
           CuDiff::tan(x);
}

}    // namespace

TEST(CuDiffBasics, SingleVariableLayout)
{
    auto [x] = CuDiff::make_variables<1>(2.5f);

    EXPECT_FLOAT_EQ(x.val(), 2.5f);
    EXPECT_FLOAT_EQ(x.derivative(0), 1.0f);
}

TEST(CuDiffBasics, MultipleVariableLayout)
{
    auto [x, y, z] = CuDiff::make_variables<3>(1.0f, -2.0f, 5.0f);

    EXPECT_FLOAT_EQ(x.derivative(0), 1.0f);
    EXPECT_FLOAT_EQ(x.derivative(1), 0.0f);
    EXPECT_FLOAT_EQ(y.derivative(1), 1.0f);
    EXPECT_FLOAT_EQ(z.derivative(2), 1.0f);
}

TEST(CuDiffScalar, ElementaryFunctionsMatchAnalyticalDerivatives)
{
    {
        auto [x] = CuDiff::make_variables<1>(0.3f);
        auto y   = CuDiff::sin(x);
        expect_dual_near(y, std::sin(0.3f), std::array<float, 1> {std::cos(0.3f)});
    }

    {
        auto [x] = CuDiff::make_variables<1>(0.3f);
        auto y   = CuDiff::cos(x);
        expect_dual_near(y, std::cos(0.3f), std::array<float, 1> {-std::sin(0.3f)});
    }

    {
        auto [x] = CuDiff::make_variables<1>(0.3f);
        auto y   = CuDiff::tan(x);
        expect_dual_near(y, std::tan(0.3f), std::array<float, 1> {1.0f / (std::cos(0.3f) * std::cos(0.3f))});
    }

    {
        auto [x] = CuDiff::make_variables<1>(0.25f);
        auto y   = CuDiff::asin(x);
        expect_dual_near(y, std::asin(0.25f), std::array<float, 1> {1.0f / std::sqrt(1.0f - 0.25f * 0.25f)});
    }

    {
        auto [x] = CuDiff::make_variables<1>(0.25f);
        auto y   = CuDiff::acos(x);
        expect_dual_near(y, std::acos(0.25f), std::array<float, 1> {-1.0f / std::sqrt(1.0f - 0.25f * 0.25f)});
    }

    {
        auto [x] = CuDiff::make_variables<1>(0.7f);
        auto y   = CuDiff::atan(x);
        expect_dual_near(y, std::atan(0.7f), std::array<float, 1> {1.0f / (1.0f + 0.7f * 0.7f)});
    }

    {
        auto [y, x]       = CuDiff::make_variables<2>(0.5f, 1.2f);
        auto r            = CuDiff::atan2(y, x);
        const float denom = 1.2f * 1.2f + 0.5f * 0.5f;
        expect_dual_near(r, std::atan2(0.5f, 1.2f), std::array<float, 2> {1.2f / denom, -0.5f / denom});
    }

    {
        auto [x] = CuDiff::make_variables<1>(0.4f);
        auto y   = CuDiff::exp(x);
        expect_dual_near(y, std::exp(0.4f), std::array<float, 1> {std::exp(0.4f)});
    }

    {
        auto [x] = CuDiff::make_variables<1>(1.4f);
        auto y   = CuDiff::log(x);
        expect_dual_near(y, std::log(1.4f), std::array<float, 1> {1.0f / 1.4f});
    }

    {
        auto [x] = CuDiff::make_variables<1>(4.0f);
        auto y   = CuDiff::sqrt(x);
        expect_dual_near(y, 2.0f, std::array<float, 1> {0.25f});
    }

    {
        auto [x] = CuDiff::make_variables<1>(1.3f);
        auto y   = CuDiff::pow(x, 2.5f);
        expect_dual_near(y, std::pow(1.3f, 2.5f), std::array<float, 1> {2.5f * std::pow(1.3f, 1.5f)});
    }

    {
        auto [x, y]       = CuDiff::make_variables<2>(1.3f, 2.1f);
        auto z            = CuDiff::pow(x, y);
        const float value = std::pow(1.3f, 2.1f);
        expect_dual_near(z, value, std::array<float, 2> {2.1f * std::pow(1.3f, 1.1f), value * std::log(1.3f)});
    }
}

TEST(CuDiffScalar, PiecewiseFunctionsAndDivisionBehaveAsExpected)
{
    {
        auto [x] = CuDiff::make_variables<1>(-3.0f);
        auto y   = CuDiff::abs(x);
        expect_dual_near(y, 3.0f, std::array<float, 1> {-1.0f});
    }

    {
        auto [x] = CuDiff::make_variables<1>(2.0f);
        auto y   = CuDiff::sign(x);
        expect_dual_near(y, 1.0f, std::array<float, 1> {0.0f});
    }

    {
        auto [x] = CuDiff::make_variables<1>(2.7f);
        auto y   = CuDiff::floor(x);
        expect_dual_near(y, 2.0f, std::array<float, 1> {0.0f});
    }

    {
        auto [x] = CuDiff::make_variables<1>(2.7f);
        auto y   = CuDiff::ceil(x);
        expect_dual_near(y, 3.0f, std::array<float, 1> {0.0f});
    }

    {
        auto [x, y] = CuDiff::make_variables<2>(1.5f, 3.0f);
        auto z      = CuDiff::min(x, y);
        expect_dual_near(z, 1.5f, std::array<float, 2> {1.0f, 0.0f});
    }

    {
        auto [x, y] = CuDiff::make_variables<2>(4.0f, 3.0f);
        auto z      = CuDiff::max(x, y);
        expect_dual_near(z, 4.0f, std::array<float, 2> {1.0f, 0.0f});
    }

    {
        auto [x] = CuDiff::make_variables<1>(0.25f);
        auto z   = CuDiff::clamp(x, 0.0f, 1.0f);
        expect_dual_near(z, 0.25f, std::array<float, 1> {1.0f});
    }

    {
        auto [x] = CuDiff::make_variables<1>(-0.25f);
        auto z   = CuDiff::clamp(x, 0.0f, 1.0f);
        expect_dual_near(z, 0.0f, std::array<float, 1> {0.0f});
    }

    {
        auto [x, y] = CuDiff::make_variables<2>(1.0f, 1e-7f);
        auto z      = x / y;
        EXPECT_TRUE(std::isfinite(z.val()));
        EXPECT_TRUE(std::isfinite(z.derivative(0)));
        EXPECT_TRUE(std::isfinite(z.derivative(1)));
        EXPECT_NEAR(z.derivative(0), 0.01f, 1e-6f);
    }
}

TEST(CuDiffScalar, CompositeExpressionMatchesFiniteDifferences)
{
    auto analytic = [](const auto& x, const auto& y, const auto& z)
    {
        return CuDiff::sin(x * y) + CuDiff::exp(z) + CuDiff::log(x + 2.0f) + CuDiff::sqrt(y + 5.0f) +
               CuDiff::atan2(x, z);
    };
    auto numeric = [](float x, float y, float z)
    {
        return std::sin(x * y) + std::exp(z) + std::log(x + 2.0f) + std::sqrt(y + 5.0f) + std::atan2(x, z);
    };

    auto [x, y, z] = CuDiff::make_variables<3>(0.4f, -0.2f, 0.7f);
    auto f         = analytic(x, y, z);

    const float h  = 1e-4f;
    const float dx = (numeric(0.4f + h, -0.2f, 0.7f) - numeric(0.4f - h, -0.2f, 0.7f)) / (2.0f * h);
    const float dy = (numeric(0.4f, -0.2f + h, 0.7f) - numeric(0.4f, -0.2f - h, 0.7f)) / (2.0f * h);
    const float dz = (numeric(0.4f, -0.2f, 0.7f + h) - numeric(0.4f, -0.2f, 0.7f - h)) / (2.0f * h);

    expect_near_scalar(f.val(), numeric(0.4f, -0.2f, 0.7f), 1e-4f);
    expect_near_scalar(f.derivative(0), dx, 5e-3f);
    expect_near_scalar(f.derivative(1), dy, 5e-3f);
    expect_near_scalar(f.derivative(2), dz, 5e-3f);
}

TEST(CuDiffVector, WrapUnwrapPreserveValuesAndDerivatives)
{
    auto [x, y] = CuDiff::make_variables<2>(1.5f, -2.5f);
    auto v      = CuDiff::wrap(x, y);
    auto [u, w] = CuDiff::unwrap(v);

    expect_dual_near(u, 1.5f, std::array<float, 2> {1.0f, 0.0f});
    expect_dual_near(w, -2.5f, std::array<float, 2> {0.0f, 1.0f});
}

TEST(CuDiffVector, DotLengthNormalizeReflectMatchAnalyticalDerivatives)
{
    {
        auto [v] = CuDiff::make_variables<2>(glm::vec2(3.0f, 4.0f));
        auto y   = CuDiff::dot(v, glm::vec2(1.0f, 2.0f));
        expect_dual_near(y, 11.0f, std::array<float, 2> {1.0f, 2.0f});
    }

    {
        auto [v] = CuDiff::make_variables<2>(glm::vec2(3.0f, 4.0f));
        auto y   = CuDiff::length(v);
        expect_dual_near(y, 5.0f, std::array<float, 2> {3.0f / 5.0f, 4.0f / 5.0f});
    }

    {
        auto [v] = CuDiff::make_variables<2>(glm::vec2(3.0f, 4.0f));
        auto y   = CuDiff::normalize(v);
        const glm::vec2 expected_value(0.6f, 0.8f);
        const glm::vec2 d0 = normalize_jacobian_column(glm::vec2(3.0f, 4.0f), 0);
        const glm::vec2 d1 = normalize_jacobian_column(glm::vec2(3.0f, 4.0f), 1);
        expect_dual_near(y, expected_value, std::array<glm::vec2, 2> {d0, d1});
    }

    {
        auto [v] = CuDiff::make_variables<2>(glm::vec2(1.0f, -2.0f));
        const glm::vec2 n(0.0f, 1.0f);
        auto y                         = CuDiff::reflect(v, n);
        const glm::vec2 expected_value = glm::reflect(glm::vec2(1.0f, -2.0f), n);
        const glm::vec2 d0             = reflect_jacobian_column(n, 0);
        const glm::vec2 d1             = reflect_jacobian_column(n, 1);
        expect_dual_near(y, expected_value, std::array<glm::vec2, 2> {d0, d1});
    }
}

TEST(CuDiffVector, VectorClampAndRefractUseComponentAndFiniteDifferenceBehavior)
{
    {
        auto [v] = CuDiff::make_variables<2>(glm::vec2(-1.0f, 2.0f));
        auto y   = CuDiff::clamp(v, glm::vec2(0.0f), glm::vec2(1.0f));
        expect_vec_near(y.val(), glm::vec2(0.0f, 1.0f));
        expect_vec_near(y.derivative(0), glm::vec2(0.0f, 0.0f));
        expect_vec_near(y.derivative(1), glm::vec2(0.0f, 0.0f));
    }

    {
        auto numeric = [](const glm::vec2& input)
        {
            const glm::vec2 n = glm::normalize(glm::vec2(0.0f, 1.0f));
            return CuDiff::refract(input, n, 1.5f);
        };

        const glm::vec2 input(1.0f, -2.0f);
        auto [v]          = CuDiff::make_variables<2>(input);
        const glm::vec2 n = glm::normalize(glm::vec2(0.0f, 1.0f));
        auto y            = CuDiff::refract(v, n, 1.5f);

        glm::vec2 expected_value = numeric(input);
        expect_vec_near(y.val(), expected_value, 1e-4f);

        const float h = 1e-4f;
        glm::vec2 fd0 = (numeric(glm::vec2(1.0f + h, -2.0f)) - numeric(glm::vec2(1.0f - h, -2.0f))) / (2.0f * h);
        glm::vec2 fd1 = (numeric(glm::vec2(1.0f, -2.0f + h)) - numeric(glm::vec2(1.0f, -2.0f - h))) / (2.0f * h);
        expect_vec_near(y.derivative(0), fd0, 3e-3f);
        expect_vec_near(y.derivative(1), fd1, 3e-3f);
    }
}

TEST(CuDiffGPU, ScalarCompositeMatchesHost)
{
    ScalarParityResult* device_result = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_result), sizeof(ScalarParityResult)));

    scalar_parity_kernel<<<1, 1>>>(device_result);
    CUDA_CHECK(cudaGetLastError());

    ScalarParityResult host_result {};
    CUDA_CHECK(cudaMemcpy(&host_result, device_result, sizeof(ScalarParityResult), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_result));

    auto [x, y, z] = CuDiff::make_variables<3>(1.0f, -2.0f, 5.0f);
    auto f         = scalar_composite(x, y, z);

    expect_near_scalar(host_result.value, f.val());
    expect_near_scalar(host_result.derivatives[0], f.derivative(0));
    expect_near_scalar(host_result.derivatives[1], f.derivative(1));
    expect_near_scalar(host_result.derivatives[2], f.derivative(2));
}

TEST(CuDiffGPU, VectorNormalizeMatchesHost)
{
    VecParityResult* device_result = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_result), sizeof(VecParityResult)));

    vector_parity_kernel<<<1, 1>>>(device_result);
    CUDA_CHECK(cudaGetLastError());

    VecParityResult host_result {};
    CUDA_CHECK(cudaMemcpy(&host_result, device_result, sizeof(VecParityResult), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_result));

    auto [v] = CuDiff::make_variables<2>(glm::vec2(3.0f, 4.0f));
    auto n   = CuDiff::normalize(v);

    expect_near_scalar(host_result.value[0], n.val().x);
    expect_near_scalar(host_result.value[1], n.val().y);
    expect_near_scalar(host_result.derivatives[0][0], n.derivative(0).x);
    expect_near_scalar(host_result.derivatives[0][1], n.derivative(1).x);
    expect_near_scalar(host_result.derivatives[1][0], n.derivative(0).y);
    expect_near_scalar(host_result.derivatives[1][1], n.derivative(1).y);
}

TEST(CuDiffRegression, SmallDenominatorDivisionRemainsFinite)
{
    auto [x, y] = CuDiff::make_variables<2>(1.0f, 1e-7f);
    auto z      = x / y;
    EXPECT_TRUE(std::isfinite(z.val()));
    EXPECT_TRUE(std::isfinite(z.derivative(0)));
    EXPECT_TRUE(std::isfinite(z.derivative(1)));
}

TEST(CuDiffRegression, NumericFiniteDifferenceSanityForCompositeScalar)
{
    auto [x] = CuDiff::make_variables<1>(0.35f);
    auto y   = numeric_scalar_function(x);

    const float expected_value = numeric_scalar_function(0.35f);
    const float expected_derivative =
        finite_difference_scalar(0.35f, [](float v) { return numeric_scalar_function(v); });

    expect_near_scalar(y.val(), expected_value);
    expect_near_scalar(y.derivative(0), expected_derivative, 2e-3f);
}
