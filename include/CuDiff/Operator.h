#pragma once

#include <CuDiff/Dual.h>
#include <CuDiff/Platform.h>

namespace CuDiff
{
template<int N>
struct OperatorAdd
{
    template<typename Atype, typename Btype>
    CUDIFF_HOSTDEVICE static auto call(const Atype& a, const Btype& b)
    {
        using R = decltype(value_of(a) + value_of(b));
        Dual<N, R> r(value_of(a) + value_of(b));
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, derivative_of(a, i) + derivative_of(b, i));
        }
        return r;
    }
};

template<int N>
struct OperatorSub
{
    template<typename Atype, typename Btype>
    CUDIFF_HOSTDEVICE static auto call(const Atype& a, const Btype& b)
    {
        using R = decltype(value_of(a) - value_of(b));
        Dual<N, R> r(value_of(a) - value_of(b));
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, derivative_of(a, i) - derivative_of(b, i));
        }
        return r;
    }
};

template<int N>
struct OperatorMul
{
    template<typename Atype, typename Btype>
    CUDIFF_HOSTDEVICE static auto call(const Atype& a, const Btype& b)
    {
        using R = decltype(value_of(a) * value_of(b));
        Dual<N, R> r(value_of(a) * value_of(b));
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, derivative_of(a, i) * value_of(b) + value_of(a) * derivative_of(b, i));
        }
        return r;
    }
};

template<int N>
struct OperatorDiv
{
    template<typename Atype, typename Btype>
    CUDIFF_HOSTDEVICE static auto call(const Atype& a, const Btype& b)
    {
        using R = decltype(value_of(a) / value_of(b));
        Dual<N, R> r(value_of(a) / value_of(b));
        auto denom = value_of(b) * value_of(b);
        if constexpr(std::is_floating_point_v<dual_value_type_t<Btype>>)
            denom = denom > dual_value_type_t<Btype>(0) ? ::max(1e-5f, denom) : ::min(-1e-5f, denom);
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, (derivative_of(a, i) * value_of(b) - value_of(a) * derivative_of(b, i)) / denom);
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
    return OperatorAdd<N>::call(a, b);
}

template<int N, typename T, typename U>
CUDIFF_HOSTDEVICE inline auto operator-(const Dual<N, T>& a, const Dual<N, U>& b)
{
    return OperatorSub<N>::call(a, b);
}

template<int N, typename T, typename U>
CUDIFF_HOSTDEVICE inline auto operator*(const Dual<N, T>& a, const Dual<N, U>& b)
{
    return OperatorMul<N>::call(a, b);
}

template<int N, typename T, typename U>
CUDIFF_HOSTDEVICE inline auto operator/(const Dual<N, T>& a, const Dual<N, U>& b)
{
    return OperatorDiv<N>::call(a, b);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator+(const Dual<N, T>& a, const U& b)
{
    return OperatorAdd<N>::call(a, b);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator+(const U& a, const Dual<N, T>& b)
{
    return OperatorAdd<N>::call(a, b);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator-(const Dual<N, T>& a, const U& b)
{
    return OperatorSub<N>::call(a, b);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator-(const U& a, const Dual<N, T>& b)
{
    return OperatorSub<N>::call(a, b);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator*(const Dual<N, T>& a, const U& b)
{
    return OperatorMul<N>::call(a, b);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator*(const U& a, const Dual<N, T>& b)
{
    return OperatorMul<N>::call(a, b);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator/(const Dual<N, T>& a, const U& b)
{
    return OperatorDiv<N>::call(a, b);
}

template<int N, typename T, typename U, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator/(const U& a, const Dual<N, T>& b)
{
    return OperatorDiv<N>::call(a, b);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE inline Dual<N, T> operator-(const Dual<N, T>& a)
{
    return OperatorNeg<N, T>::call(a);
}
}    // namespace CuDiff