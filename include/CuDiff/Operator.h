#pragma once

#include <CuDiff/Dual.h>
#include <CuDiff/Platform.h>

namespace CuDiff
{
template<int N, typename T, typename U, bool _stochasticT, bool _stochasticU>
struct OperatorAdd
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T, _stochasticT>& a, const Dual<N, U, _stochasticU>& b)
    {
        constexpr bool stochasticR = (_stochasticT || _stochasticU);
        using R                    = decltype(a.val() + b.val());
        Dual<N, R, stochasticR> r(a.val() + b.val());
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i) + b.derivative(i));
            if constexpr(stochasticR)
            {
                if constexpr(_stochasticT)
                {
                    r.mut_score(i) += a.score(i);
                }
                if constexpr(_stochasticU)
                {
                    r.mut_score(i) += b.score(i);
                }
            }
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T, _stochasticT>& a, const U& b)
    {
        using R = decltype(a.val() + b);
        Dual<N, R, _stochasticT> r(a.val() + b);
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i));
            if constexpr(_stochasticT)
            {
                r.setScore(i, a.score(i));
            }
        }
        return r;
    }
};

template<int N, typename T, typename U, bool _stochasticT, bool _stochasticU>
struct OperatorSub
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T, _stochasticT>& a, const Dual<N, U, _stochasticU>& b)
    {
        constexpr bool stochasticR = (_stochasticT || _stochasticU);
        using R                    = decltype(a.val() - b.val());
        Dual<N, R, stochasticR> r(a.val() - b.val());
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i) - b.derivative(i));
            if constexpr(stochasticR)
            {
                if constexpr(_stochasticT)
                {
                    r.mut_score(i) += a.score(i);
                }
                if constexpr(_stochasticU)
                {
                    r.mut_score(i) += b.score(i);
                }
            }
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T, _stochasticT>& a, const U& b)
    {
        using R = decltype(a.val() - b);
        Dual<N, R, _stochasticT> r(a.val() - b);
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i));
            if constexpr(_stochasticT)
            {
                r.setScore(i, a.score(i));
            }
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static auto call(const U& a, const Dual<N, T, _stochasticT>& b)
    {
        using R = decltype(a - b.val());
        Dual<N, R, _stochasticT> r(a - b.val());
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, -b.derivative(i));
            if constexpr(_stochasticT)
            {
                r.setScore(i, b.score(i));
            }
        }
        return r;
    }
};

template<int N, typename T, typename U, bool _stochasticT, bool _stochasticU>
struct OperatorMul
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T, _stochasticT>& a, const Dual<N, U, _stochasticU>& b)
    {
        constexpr bool stochasticR = (_stochasticT || _stochasticU);
        using R                    = decltype(a.val() * b.val());
        Dual<N, R, stochasticR> r(a.val() * b.val());
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i) * b.val() + a.val() * b.derivative(i));
            if constexpr(stochasticR)
            {
                if constexpr(_stochasticT)
                {
                    r.mut_score(i) += a.score(i);
                }
                if constexpr(_stochasticU)
                {
                    r.mut_score(i) += b.score(i);
                }
            }
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T, _stochasticT>& a, const U& b)
    {
        using R = decltype(a.val() * b);
        Dual<N, R, _stochasticT> r(a.val() * b);
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i) * b);
            if constexpr(_stochasticT)
            {
                r.setScore(i, a.score(i));
            }
        }
        return r;
    }
};

template<int N, typename T, typename U, bool _stochasticT, bool _stochasticU>
struct OperatorDiv
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T, _stochasticT>& a, const Dual<N, U, _stochasticU>& b)
    {
        constexpr bool stochasticR = (_stochasticT || _stochasticU);
        using R                    = decltype(a.val() / b.val());
        Dual<N, R, stochasticR> r(a.val() / b.val());
        auto denom = b.val() * b.val();
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, (a.derivative(i) * b.val() - a.val() * b.derivative(i)) / denom);
            if constexpr(stochasticR)
            {
                if constexpr(_stochasticT)
                {
                    r.mut_score(i) += a.score(i);
                }
                if constexpr(_stochasticU)
                {
                    r.mut_score(i) += b.score(i);
                }
            }
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T, _stochasticT>& a, const U& b)
    {
        using R = decltype(a.val() / b);
        Dual<N, R, _stochasticT> r(a.val() / b);
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i) / b);
            if constexpr(_stochasticT)
            {
                r.setScore(i, a.score(i));
            }
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static auto call(const U& a, const Dual<N, T, _stochasticT>& b)
    {
        using R = decltype(a / b.val());
        Dual<N, R, _stochasticT> r(a / b.val());
        auto denom = b.val() * b.val();
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, (-a * b.derivative(i)) / denom);
            if constexpr(_stochasticT)
            {
                r.setScore(i, b.score(i));
            }
        }
        return r;
    }
};

template<int N, typename T, bool _stochasticT>
struct OperatorNeg
{
    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& a)
    {
        return Dual<N, T>(T(0)) - a;
    }
};

// Binary arithmetic operators
template<int N, typename T, typename U, bool _stochasticT, bool _stochasticU>
CUDIFF_HOSTDEVICE inline auto operator+(const Dual<N, T, _stochasticT>& a, const Dual<N, U, _stochasticU>& b)
{
    return OperatorAdd<N, T, U, _stochasticT, _stochasticU>::call(a, b);
}

template<int N, typename T, typename U, bool _stochasticT, bool _stochasticU>
CUDIFF_HOSTDEVICE inline auto operator-(const Dual<N, T, _stochasticT>& a, const Dual<N, U, _stochasticU>& b)
{
    return OperatorSub<N, T, U, _stochasticT, _stochasticU>::call(a, b);
}

template<int N, typename T, typename U, bool _stochasticT, bool _stochasticU>
CUDIFF_HOSTDEVICE inline auto operator*(const Dual<N, T, _stochasticT>& a, const Dual<N, U, _stochasticU>& b)
{
    return OperatorMul<N, T, U, _stochasticT, _stochasticU>::call(a, b);
}

template<int N, typename T, typename U, bool _stochasticT, bool _stochasticU>
CUDIFF_HOSTDEVICE inline auto operator/(const Dual<N, T, _stochasticT>& a, const Dual<N, U, _stochasticU>& b)
{
    return OperatorDiv<N, T, U, _stochasticT, _stochasticU>::call(a, b);
}

template<int N, typename T, typename U, bool _stochasticT, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator+(const Dual<N, T, _stochasticT>& a, const U& b)
{
    return OperatorAdd<N, T, U, _stochasticT, false>::call(a, b);
}

template<int N, typename T, typename U, bool _stochasticT, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator+(const U& a, const Dual<N, T, _stochasticT>& b)
{
    return OperatorAdd<N, T, U, _stochasticT, false>::call(b, a);
}

template<int N, typename T, typename U, bool _stochasticT, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator-(const Dual<N, T, _stochasticT>& a, const U& b)
{
    return OperatorSub<N, T, U, _stochasticT, false>::call(a, b);
}

template<int N, typename T, typename U, bool _stochasticT, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator-(const U& a, const Dual<N, T, _stochasticT>& b)
{
    return OperatorSub<N, T, U, _stochasticT, false>::call(a, b);
}

template<int N, typename T, typename U, bool _stochasticT, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator*(const Dual<N, T, _stochasticT>& a, const U& b)
{
    return OperatorMul<N, T, U, _stochasticT, false>::call(a, b);
}

template<int N, typename T, typename U, bool _stochasticT, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator*(const U& a, const Dual<N, T, _stochasticT>& b)
{
    return OperatorMul<N, T, U, _stochasticT, false>::call(b, a);
}

template<int N, typename T, typename U, bool _stochasticT, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator/(const Dual<N, T, _stochasticT>& a, const U& b)
{
    return OperatorDiv<N, T, U, _stochasticT, false>::call(a, b);
}

template<int N, typename T, typename U, bool _stochasticT, typename = std::enable_if_t<!is_dual_v<U>>>
CUDIFF_HOSTDEVICE inline auto operator/(const U& a, const Dual<N, T, _stochasticT>& b)
{
    return OperatorDiv<N, T, U, _stochasticT, false>::call(a, b);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE inline Dual<N, T, _stochasticT> operator-(const Dual<N, T, _stochasticT>& a)
{
    return OperatorNeg<N, T, _stochasticT>::call(a);
}
}    // namespace CuDiff