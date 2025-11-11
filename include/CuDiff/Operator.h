#pragma once

#include <CuDiff/Dual.h>
#include <CuDiff/Platform.h>

namespace CuDiff
{
template<int N, typename T, typename U>
struct OperatorAdd
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& a, const Dual<N, U>& b)
    {
        using R = decltype(a.val() + b.val());
        Dual<N, R> r(a.val() + b.val());
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i) + b.derivative(i));
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& a, const U& b)
    {
        using R = decltype(a.val() + b);
        Dual<N, R> r(a.val() + b);
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i));
        }
        return r;
    }
};

template<int N, typename T, typename U>
struct OperatorSub
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& a, const Dual<N, U>& b)
    {
        using R = decltype(a.val() - b.val());
        Dual<N, R> r(a.val() - b.val());
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i) - b.derivative(i));
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& a, const U& b)
    {
        using R = decltype(a.val() - b);
        Dual<N, R> r(a.val() - b);
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i));
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static auto call(const U& a, const Dual<N, T>& b)
    {
        using R = decltype(a - b.val());
        Dual<N, R> r(a - b.val());
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, -b.derivative(i));
        }
        return r;
    }
};

template<int N, typename T, typename U>
struct OperatorMul
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& a, const Dual<N, U>& b)
    {
        using R = decltype(a.val() * b.val());
        Dual<N, R> r(a.val() * b.val());
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i) * b.val() + a.val() * b.derivative(i));
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& a, const U& b)
    {
        using R = decltype(a.val() * b);
        Dual<N, R> r(a.val() * b);
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i) * b);
        }
        return r;
    }
};

template<int N, typename T, typename U>
struct OperatorDiv
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& a, const Dual<N, U>& b)
    {
        using R = decltype(a.val() / b.val());
        Dual<N, R> r(a.val() / b.val());
        auto denom = b.val() * b.val();
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, (a.derivative(i) * b.val() - a.val() * b.derivative(i)) / denom);
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& a, const U& b)
    {
        using R = decltype(a.val() / b);
        Dual<N, R> r(a.val() / b);
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i) / b);
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static auto call(const U& a, const Dual<N, T>& b)
    {
        using R = decltype(a / b.val());
        Dual<N, R> r(a / b.val());
        auto denom = b.val() * b.val();
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, (-a * b.derivative(i)) / denom);
        }
        return r;
    }
};

template<int N, typename T>
struct OperatorNeg
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& a) { return Dual<N, T>(T(0)) - a; }
};

// Binary arithmetic operators
template<int N, typename T, typename U>
CUDIFF_HOSTDEVICE inline auto operator+(const Dual<N, T>& a, const Dual<N, U>& b)
{
    return OperatorAdd<N, T, U>::call(a, b);
}

template<int N, typename T, typename U>
CUDIFF_HOSTDEVICE inline auto operator-(const Dual<N, T>& a, const Dual<N, U>& b)
{
    return OperatorSub<N, T, U>::call(a, b);
}

template<int N, typename T, typename U>
CUDIFF_HOSTDEVICE inline auto operator*(const Dual<N, T>& a, const Dual<N, U>& b)
{
    return OperatorMul<N, T, U>::call(a, b);
}

template<int N, typename T, typename U>
CUDIFF_HOSTDEVICE inline auto operator/(const Dual<N, T>& a, const Dual<N, U>& b)
{
    return OperatorDiv<N, T, U>::call(a, b);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator+(const Dual<N, T>& a, const U& b)
{
    return OperatorAdd<N, T, U>::call(a, b);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator+(const U& a, const Dual<N, T>& b)
{
    return OperatorAdd<N, T, U>::call(b, a);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator-(const Dual<N, T>& a, const U& b)
{
    return OperatorSub<N, T, U>::call(a, b);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator-(const U& a, const Dual<N, T>& b)
{
    return OperatorSub<N, T, U>::call(a, b);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator*(const Dual<N, T>& a, const U& b)
{
    return OperatorMul<N, T, U>::call(a, b);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator*(const U& a, const Dual<N, T>& b)
{
    return OperatorMul<N, T, U>::call(b, a);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator/(const Dual<N, T>& a, const U& b)
{
    return OperatorDiv<N, T, U>::call(a, b);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator/(const U& a, const Dual<N, T>& b)
{
    return OperatorDiv<N, T, U>::call(a, b);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE inline Dual<N, T> operator-(const Dual<N, T>& a)
{
    return OperatorNeg<N, T>::call(a);
}
}    // namespace CuDiff