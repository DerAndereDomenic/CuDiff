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
    template<typename Atype, typename Btype>
    CUDIFF_HOSTDEVICE static auto call(const Atype& y, const Btype& x)
    {
        auto val   = std::atan2(value_of(y), value_of(x));
        auto denom = value_of(x) * value_of(x) + value_of(y) * value_of(y);
        auto res   = Dual<N, T>(val);

        for(int i = 0; i < N; ++i)
        {
            T dy = derivative_of(y, i);
            T dx = derivative_of(x, i);
            res.setDerivative(i, (value_of(x) * dy - value_of(y) * dx) / denom);
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
    template<typename Atype, typename Btype>
    CUDIFF_HOSTDEVICE static auto call(const Atype& f, const Btype& g)
    {
        auto val = std::pow(value_of(f), value_of(g));
        auto res = Dual<N, T>(val);
        auto log = std::log(value_of(f));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, val * (derivative_of(f, i) / value_of(f) * value_of(g) + log * derivative_of(g, i)));
        }

        return res;
    }
};

template<int N, typename T>
struct OperatorSign
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x)
    {
        auto v = x.val();
        auto s = (v > T(0)) ? T(1) : (v < T(0)) ? T(-1) : T(0);

        Dual<N, T> res(s);
        // Derivative of sign(x) is zero almost everywhere
        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, T(0));
        }
        return res;
    }
};

template<int N, typename T>
struct OperatorFloor
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x)
    {
        Dual<N, T> res(std::floor(x.val()));
        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, T(0));
        }
        return res;
    }
};

template<int N, typename T>
struct OperatorCeil
{
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x)
    {
        Dual<N, T> res(std::ceil(x.val()));
        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, T(0));
        }
        return res;
    }
};

template<int N, typename T>
struct OperatorMin
{
    template<typename Atype, typename Btype>
    CUDIFF_HOSTDEVICE static auto call(const Atype& a, const Btype& b)
    {
        bool takeA = value_of(a) < value_of(b);
        Dual<N, T> res(takeA ? value_of(a) : value_of(b));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, takeA ? derivative_of(a, i) : derivative_of(b, i));
        }

        return res;
    }
};

template<int N, typename T>
struct OperatorMax
{
    template<typename Atype, typename Btype>
    CUDIFF_HOSTDEVICE static auto call(const Atype& a, const Btype& b)
    {
        bool takeA = value_of(a) > value_of(b);
        Dual<N, T> res(takeA ? value_of(a) : value_of(b));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, takeA ? derivative_of(a, i) : derivative_of(b, i));
        }

        return res;
    }
};

template<int N, typename T>
struct OperatorClamp
{
    template<typename Atype, typename Btype, typename Ctype>
    CUDIFF_HOSTDEVICE static auto call(const Atype& x, const Btype& lo, const Ctype& hi)
    {
        if(value_of(x) < value_of(lo))
        {
            Dual<N, T> res(value_of(lo));
            for(int i = 0; i < N; ++i)
                res.setDerivative(i, derivative_of(lo, i));
            return res;
        }
        else if(value_of(x) > value_of(hi))
        {
            Dual<N, T> res(value_of(hi));
            for(int i = 0; i < N; ++i)
                res.setDerivative(i, derivative_of(hi, i));
            return res;
        }
        else
        {
            Dual<N, T> res(value_of(x));
            for(int i = 0; i < N; ++i)
                res.setDerivative(i, derivative_of(x, i));
            return res;
        }
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

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> pow(const T& f, const Dual<N, T>& g)
{
    return OperatorPow<N, T>::call(f, g);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> sign(const Dual<N, T>& x)
{
    return OperatorSign<N, T>::call(x);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> floor(const Dual<N, T>& x)
{
    return OperatorFloor<N, T>::call(x);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> ceil(const Dual<N, T>& x)
{
    return OperatorCeil<N, T>::call(x);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> min(const Dual<N, T>& a, const Dual<N, T>& b)
{
    return OperatorMin<N, T>::call(a, b);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> min(const Dual<N, T>& a, const T& b)
{
    return OperatorMin<N, T>::call(a, b);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> min(const T& a, const Dual<N, T>& b)
{
    return OperatorMin<N, T>::call(b, a);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> max(const Dual<N, T>& a, const Dual<N, T>& b)
{
    return OperatorMax<N, T>::call(a, b);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> max(const Dual<N, T>& a, const T& b)
{
    return OperatorMax<N, T>::call(a, b);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> max(const T& a, const Dual<N, T>& b)
{
    return OperatorMax<N, T>::call(b, a);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> clamp(const Dual<N, T>& x, const Dual<N, T>& lo, const Dual<N, T>& hi)
{
    return OperatorClamp<N, T>::call(x, lo, hi);
}

template<int N, typename T>
CUDIFF_HOSTDEVICE Dual<N, T> clamp(const Dual<N, T>& x, const T& lo, const T& hi)
{
    return OperatorClamp<N, T>::call(x, lo, hi);
}
}    // namespace CuDiff