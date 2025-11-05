#pragma once

#include <CuDiff/Dual.h>

namespace CuDiff
{

template<int N, typename T>
struct OperatorSqrt
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x)
    {
        auto sq  = std::sqrt(x.val());
        auto res = Dual<N, T>(sq);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, x.derivative(i) / (T(2) * sq));
        }

        return res;
    }
};

template<int N, typename T>
struct OperatorSin
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x)
    {
        auto sin = std::sin(x.val());
        auto cos = std::cos(x.val());
        auto res = Dual<N, T>(sin);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, x.derivative(i) * cos);
        }

        return res;
    }
};

template<int N, typename T>
struct OperatorCos
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x)
    {
        auto cos = std::cos(x.val());
        auto sin = std::sin(x.val());
        auto res = Dual<N, T>(cos);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, -x.derivative(i) * sin);
        }

        return res;
    }
};

template<int N, typename T>
struct OperatorTan
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x)
    {
        return OperatorSin<N, T>::call(x) / OperatorCos<N, T>::call(x);
    }
};

template<int N, typename T>
struct OperatorAsin
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x)
    {
        auto val   = std::asin(x.val());
        auto denom = std::sqrt(T(1) - x.val() * x.val());
        auto res   = Dual<N, T>(val);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, x.derivative(i) / denom);
        }

        return res;
    }
};

template<int N, typename T>
struct OperatorAcos
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x)
    {
        auto val   = std::acos(x.val());
        auto denom = std::sqrt(T(1) - x.val() * x.val());
        auto res   = Dual<N, T>(val);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, -x.derivative(i) / denom);
        }

        return res;
    }
};

template<int N, typename T>
struct OperatorAtan
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x)
    {
        auto val   = std::atan(x.val());
        auto denom = T(1) + x.val() * x.val();
        auto res   = Dual<N, T>(val);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, x.derivative(i) / denom);
        }

        return res;
    }
};

template<int N, typename T>
struct OperatorAtan2
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& y, const Dual<N, T>& x)
    {
        auto val   = std::atan2(y.val(), x.val());
        auto denom = x.val() * x.val() + y.val() * y.val();
        auto res   = Dual<N, T>(val);

        for(int i = 0; i < N; ++i)
        {
            T dy = y.derivative(i);
            T dx = x.derivative(i);
            res.setDerivative(i, (x.val() * dy - y.val() * dx) / denom);
        }

        return res;
    }
};

template<int N, typename T>
struct OperatorExp
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x)
    {
        auto val = std::exp(x.val());
        auto res = Dual<N, T>(val);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, val * x.derivative(i));
        }

        return res;
    }
};

template<int N, typename T>
struct OperatorLog
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x)
    {
        auto val = std::log(x.val());
        auto res = Dual<N, T>(val);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, x.derivative(i) / x.val());
        }

        return res;
    }
};

template<int N, typename T>
struct OperatorAbs
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x)
    {
        auto val = std::abs(x.val());
        auto res = Dual<N, T>(val);

        T sign = (x.val() > T(0)) ? T(1) : (x.val() < T(0)) ? T(-1) : T(0);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, sign * x.derivative(i));
        }

        return res;
    }
};

template<int N, typename T>
struct OperatorPow
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x, const T& n)
    {
        auto val = std::pow(x.val(), n);
        auto res = Dual<N, T>(val);
        auto pow = std::pow(x.val(), n - T(1));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, n * pow * x.derivative(i));
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& f, const Dual<N, T>& g)
    {
        auto val = std::pow(f.val(), g.val());
        auto res = Dual<N, T>(val);
        auto log = std::log(f.val());

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, val * (f.derivative(i) / f.val() * g.val() + log * g.derivative(i)));
        }

        return res;
    }
};

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> sqrt(const Dual<N, T>& a)
{
    return OperatorSqrt<N, T>::call(a);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> sin(const Dual<N, T>& a)
{
    return OperatorSin<N, T>::call(a);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> cos(const Dual<N, T>& a)
{
    return OperatorCos<N, T>::call(a);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> tan(const Dual<N, T>& a)
{
    return OperatorTan<N, T>::call(a);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> asin(const Dual<N, T>& a)
{
    return OperatorAsin<N, T>::call(a);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> acos(const Dual<N, T>& a)
{
    return OperatorAcos<N, T>::call(a);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> atan(const Dual<N, T>& a)
{
    return OperatorAtan<N, T>::call(a);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> atan2(const Dual<N, T>& y, const Dual<N, T>& x)
{
    return OperatorAtan2<N, T>::call(y, x);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> exp(const Dual<N, T>& a)
{
    return OperatorExp<N, T>::call(a);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> log(const Dual<N, T>& a)
{
    return OperatorLog<N, T>::call(a);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> abs(const Dual<N, T>& a)
{
    return OperatorAbs<N, T>::call(a);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> pow(const Dual<N, T>& a, const T& n)
{
    return OperatorPow<N, T>::call(a, n);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> pow(const Dual<N, T>& f, const Dual<N, T>& g)
{
    return OperatorPow<N, T>::call(f, g);
}
}    // namespace CuDiff