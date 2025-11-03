#pragma once

#include <CuDiff/Dual.h>
#include <CuDiff/Platform.h>

namespace CuDiff
{
template<int N, typename T, typename U = float>
struct OperatorAdd
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& a, const Dual<N, T>& b)
    {
        Dual<N, T> r(a.val() + b.val());
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i) + b.derivative(i));
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& a, const U& b)
    {
        return OperatorAdd<N, T, U>::call(a, Dual<N, T>(static_cast<T>(b)));
    }
};

template<int N, typename T, typename U = float>
struct OperatorSub
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& a, const Dual<N, T>& b)
    {
        Dual<N, T> r(a.val() - b.val());
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i) - b.derivative(i));
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& a, const U& b)
    {
        return OperatorSub<N, T, U>::call(a, Dual<N, T>(static_cast<T>(b)));
    }

    CUDIFF_HOSTDEVICE static Dual<N, T> call(const U& a, const Dual<N, T>& b)
    {
        return OperatorSub<N, T, U>::call(Dual<N, T>(static_cast<T>(a)), b);
    }
};

template<int N, typename T, typename U = float>
struct OperatorMul
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& a, const Dual<N, T>& b)
    {
        Dual<N, T> r(a.val() * b.val());
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, a.derivative(i) * b.val() + a.val() * b.derivative(i));
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& a, const U& b)
    {
        return OperatorMul<N, T, U>::call(a, Dual<N, T>(static_cast<T>(b)));
    }
};

template<int N, typename T, typename U = float>
struct OperatorDiv
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& a, const Dual<N, T>& b)
    {
        Dual<N, T> r(a.val() / b.val());
        T denom = b.val() * b.val();
        for(size_t i = 0; i < N; ++i)
        {
            r.setDerivative(i, (a.derivative(i) * b.val() - a.val() * b.derivative(i)) / denom);
        }
        return r;
    }

    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& a, const U& b)
    {
        return OperatorDiv<N, T, U>::call(a, Dual<N, T>(static_cast<T>(b)));
    }

    CUDIFF_HOSTDEVICE static Dual<N, T> call(const U& a, const Dual<N, T>& b)
    {
        return OperatorDiv<N, T, U>::call(Dual<N, T>(static_cast<T>(a)), b);
    }
};

template<int N, typename T, typename U = float>
struct OperatorNeg
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& a) { return Dual<N, T>(0) - a; }
};

// Binary arithmetic operators
template<int N, typename T>
CUDIFF_HOSTDEVICE inline Dual<N, T> operator+(const Dual<N, T>& a, const Dual<N, T>& b)
{
    return OperatorAdd<N, T>::call(a, b);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE inline Dual<N, T> operator-(const Dual<N, T>& a, const Dual<N, T>& b)
{
    return OperatorSub<N, T>::call(a, b);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE inline Dual<N, T> operator*(const Dual<N, T>& a, const Dual<N, T>& b)
{
    return OperatorMul<N, T>::call(a, b);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE inline Dual<N, T> operator/(const Dual<N, T>& a, const Dual<N, T>& b)
{
    return OperatorDiv<N, T>::call(a, b);
}

template<int N, typename T, typename U>
CUDIFF_HOSTDEVICE inline Dual<N, T> operator+(const Dual<N, T>& a, const U& b)
{
    return OperatorAdd<N, T, U>::call(a, b);
}

template<int N, typename T, typename U>
CUDIFF_HOSTDEVICE inline Dual<N, T> operator+(const U& a, const Dual<N, T>& b)
{
    return OperatorAdd<N, T, U>::call(b, a);
}

template<int N, typename T, typename U>
CUDIFF_HOSTDEVICE inline Dual<N, T> operator-(const Dual<N, T>& a, const U& b)
{
    return OperatorSub<N, T, U>::call(a, b);
}

template<int N, typename T, typename U>
CUDIFF_HOSTDEVICE inline Dual<N, T> operator-(const U& a, const Dual<N, T>& b)
{
    return OperatorSub<N, T, U>::call(a, b);
}

template<int N, typename T, typename U>
CUDIFF_HOSTDEVICE inline Dual<N, T> operator*(const Dual<N, T>& a, const U& b)
{
    return OperatorMul<N, T, U>::call(a, b);
}

template<int N, typename T, typename U>
CUDIFF_HOSTDEVICE inline Dual<N, T> operator*(const U& a, const Dual<N, T>& b)
{
    return OperatorMul<N, T, U>::call(b, a);
}

template<int N, typename T, typename U>
CUDIFF_HOSTDEVICE inline Dual<N, T> operator/(const Dual<N, T>& a, const U& b)
{
    return OperatorDiv<N, T, U>::call(a, b);
}

template<int N, typename T, typename U>
CUDIFF_HOSTDEVICE inline Dual<N, T> operator/(const U& a, const Dual<N, T>& b)
{
    return OperatorDiv<N, T, U>::call(b, a);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE inline Dual<N, T> operator-(const Dual<N, T>& a)
{
    return OperatorNeg<N, T>::call(a);
}
}    // namespace CuDiff