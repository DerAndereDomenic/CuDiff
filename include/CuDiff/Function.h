#pragma once

#include <CuDiff/Dual.h>
#include <algorithm>

namespace CuDiff
{

template<typename T>
struct OperatorSqrt
{
    CUDIFF_HOSTDEVICE static auto call(const T& x) { return std::sqrt(x); }
};

template<int N, typename T>
struct OperatorSqrt<Dual<N, T>>
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& x)
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

template<typename T>
struct OperatorSin
{
    CUDIFF_HOSTDEVICE static auto call(const T& x) { return std::sin(x); }
};

template<int N, typename T>
struct OperatorSin<Dual<N, T>>
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& x)
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

template<typename T>
struct OperatorCos
{
    CUDIFF_HOSTDEVICE static auto call(const T& x) { return std::cos(x); }
};

template<int N, typename T>
struct OperatorCos<Dual<N, T>>
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& x)
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

template<typename T>
struct OperatorTan
{
    CUDIFF_HOSTDEVICE static auto call(const T& x) { return OperatorSin<T>::call(x) / OperatorCos<T>::call(x); }
};

template<typename T>
struct OperatorAsin
{
    CUDIFF_HOSTDEVICE static auto call(const T& x) { return std::asin(x); }
};

template<int N, typename T>
struct OperatorAsin<Dual<N, T>>
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& x)
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

template<typename T>
struct OperatorAcos
{
    CUDIFF_HOSTDEVICE static auto call(const T& x) { return std::acos(x); }
};

template<int N, typename T>
struct OperatorAcos<Dual<N, T>>
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& x)
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

template<typename T>
struct OperatorAtan
{
    CUDIFF_HOSTDEVICE static auto call(const T& x) { return std::atan(x); }
};

template<int N, typename T>
struct OperatorAtan<Dual<N, T>>
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& x)
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

template<typename T, typename U, bool = !is_dual_v<T> && !is_dual_v<U>>
struct OperatorAtan2
{
    CUDIFF_HOSTDEVICE static auto call(const T& y, const U& x) { return std::atan2(y, x); }
};

template<typename T, typename U>
struct OperatorAtan2<T, U, false>
{
    CUDIFF_HOSTDEVICE static auto call(const T& y, const U& x)
    {
        constexpr int N =
            is_dual_v<T> ? dual_component_count<T>::num_variables : dual_component_count<U>::num_variables;
        using ValueType = dual_value_type_t<T>;

        auto val   = std::atan2(value_of(y), value_of(x));
        auto denom = value_of(x) * value_of(x) + value_of(y) * value_of(y);
        auto res   = Dual<N, ValueType>(val);
        for(int i = 0; i < N; ++i)
        {
            ValueType dy = derivative_of(y, i);
            ValueType dx = derivative_of(x, i);
            res.setDerivative(i, (value_of(x) * dy - value_of(y) * dx) / denom);
        }

        return res;
    }
};

template<typename T>
struct OperatorExp
{
    CUDIFF_HOSTDEVICE static auto call(const T& x) { return std::exp(x); }
};

template<int N, typename T>
struct OperatorExp<Dual<N, T>>
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& x)
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

template<typename T>
struct OperatorLog
{
    CUDIFF_HOSTDEVICE static auto call(const T& x) { return std::log(x); }
};

template<int N, typename T>
struct OperatorLog<Dual<N, T>>
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& x)
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

template<typename T>
struct OperatorAbs
{
    CUDIFF_HOSTDEVICE static auto call(const T& x) { return std::abs(x); }
};

template<int N, typename T>
struct OperatorAbs<Dual<N, T>>
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& x)
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

template<typename T, typename U, bool = !is_dual_v<T> && !is_dual_v<U>>
struct OperatorPow
{
    CUDIFF_HOSTDEVICE static auto call(const T& f, const U& g) { return std::pow(f, g); }
};

template<typename T, typename U>
struct OperatorPow<T, U, false>
{
    CUDIFF_HOSTDEVICE static auto call(const T& f, const U& g)
    {
        constexpr int N =
            is_dual_v<T> ? dual_component_count<T>::num_variables : dual_component_count<U>::num_variables;
        using ValueType = dual_value_type_t<T>;
        auto val        = std::pow(value_of(f), value_of(g));
        auto res        = Dual<N, ValueType>(val);
        auto log        = std::log(value_of(f));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, val * (derivative_of(f, i) / value_of(f) * value_of(g) + log * derivative_of(g, i)));
        }

        return res;
    }
};

template<typename T>
struct OperatorSign
{
    CUDIFF_HOSTDEVICE static auto call(const T& x) { return (x > T(0)) ? T(1) : (x < T(0)) ? T(-1) : T(0); }
};

template<int N, typename T>
struct OperatorSign<Dual<N, T>>
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& x)
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

template<typename T>
struct OperatorFloor
{
    CUDIFF_HOSTDEVICE static auto call(const T& x) { return std::floor(x); }
};

template<int N, typename T>
struct OperatorFloor<Dual<N, T>>
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& x)
    {
        Dual<N, T> res(std::floor(x.val()));
        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, T(0));
        }
        return res;
    }
};

template<typename T>
struct OperatorCeil
{
    CUDIFF_HOSTDEVICE static auto call(const T& x) { return std::ceil(x); }
};

template<int N, typename T>
struct OperatorCeil<Dual<N, T>>
{
    CUDIFF_HOSTDEVICE static auto call(const Dual<N, T>& x)
    {
        Dual<N, T> res(std::ceil(x.val()));
        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, T(0));
        }
        return res;
    }
};

template<typename Atype, typename Btype, bool = !is_dual_v<Atype> && !is_dual_v<Btype>>
struct OperatorMin
{
    CUDIFF_HOSTDEVICE static auto call(const Atype& a, const Btype& b) { return std::min(a, b); }
};

template<typename Atype, typename Btype>
struct OperatorMin<Atype, Btype, false>
{
    CUDIFF_HOSTDEVICE static auto call(const Atype& a, const Btype& b)
    {
        constexpr int N =
            is_dual_v<Atype> ? dual_component_count<Atype>::num_variables : dual_component_count<Btype>::num_variables;
        using ValueType = dual_value_type_t<Atype>;
        bool takeA      = value_of(a) < value_of(b);
        Dual<N, ValueType> res(takeA ? value_of(a) : value_of(b));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, takeA ? derivative_of(a, i) : derivative_of(b, i));
        }

        return res;
    }
};

template<typename Atype, typename Btype, bool = !is_dual_v<Atype> && !is_dual_v<Btype>>
struct OperatorMax
{
    CUDIFF_HOSTDEVICE static auto call(const Atype& a, const Btype& b) { return std::max(a, b); }
};

template<typename Atype, typename Btype>
struct OperatorMax<Atype, Btype, false>
{
    CUDIFF_HOSTDEVICE static auto call(const Atype& a, const Btype& b)
    {
        constexpr int N =
            is_dual_v<Atype> ? dual_component_count<Atype>::num_variables : dual_component_count<Btype>::num_variables;
        using ValueType = decltype(value_of(a));
        bool takeA      = value_of(a) > value_of(b);
        Dual<N, ValueType> res(takeA ? value_of(a) : value_of(b));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, takeA ? derivative_of(a, i) : derivative_of(b, i));
        }

        return res;
    }
};

template<typename Atype,
         typename Btype,
         typename Ctype,
         bool = !is_dual_v<Atype> && !is_dual_v<Btype> && !is_dual_v<Ctype>>
struct OperatorClamp
{
    CUDIFF_HOSTDEVICE static auto call(const Atype& x, const Btype& lo, const Ctype& hi)
    {
        return x < lo ? lo : (x > hi ? hi : x);
    }
};

template<typename Atype, typename Btype, typename Ctype>
struct OperatorClamp<Atype, Btype, Ctype, false>
{
    CUDIFF_HOSTDEVICE static auto call(const Atype& x, const Btype& lo, const Ctype& hi)
    {
        constexpr int N = std::max({dual_component_count<Atype>::num_variables,
                                    dual_component_count<Btype>::num_variables,
                                    dual_component_count<Ctype>::num_variables});
        using ValueType =
            std::common_type_t<dual_value_type_t<Atype>, dual_value_type_t<Btype>, dual_value_type_t<Ctype>>;

        if(value_of(x) < value_of(lo))
        {
            Dual<N, ValueType> res(value_of(lo));
            for(int i = 0; i < N; ++i)
                res.setDerivative(i, derivative_of(lo, i));
            return res;
        }
        else if(value_of(x) > value_of(hi))
        {
            Dual<N, ValueType> res(value_of(hi));
            for(int i = 0; i < N; ++i)
                res.setDerivative(i, derivative_of(hi, i));
            return res;
        }
        else
        {
            Dual<N, ValueType> res(value_of(x));
            for(int i = 0; i < N; ++i)
                res.setDerivative(i, derivative_of(x, i));
            return res;
        }
    }
};

template<typename T>
CUDIFF_HOSTDEVICE auto sqrt(const T& a)
{
    return OperatorSqrt<T>::call(a);
}

template<typename T>
CUDIFF_HOSTDEVICE auto sin(const T& a)
{
    return OperatorSin<T>::call(a);
}

template<typename T>
CUDIFF_HOSTDEVICE auto cos(const T& a)
{
    return OperatorCos<T>::call(a);
}

template<typename T>
CUDIFF_HOSTDEVICE auto tan(const T& a)
{
    return OperatorTan<T>::call(a);
}

template<typename T>
CUDIFF_HOSTDEVICE auto asin(const T& a)
{
    return OperatorAsin<T>::call(a);
}

template<typename T>
CUDIFF_HOSTDEVICE auto acos(const T& a)
{
    return OperatorAcos<T>::call(a);
}

template<typename T>
CUDIFF_HOSTDEVICE auto atan(const T& a)
{
    return OperatorAtan<T>::call(a);
}

template<typename T, typename U>
CUDIFF_HOSTDEVICE auto atan2(const T& y, const U& x)
{
    return OperatorAtan2<T, U>::call(y, x);
}

template<typename T>
CUDIFF_HOSTDEVICE auto exp(const T& a)
{
    return OperatorExp<T>::call(a);
}

template<typename T>
CUDIFF_HOSTDEVICE auto log(const T& a)
{
    return OperatorLog<T>::call(a);
}

template<typename T>
CUDIFF_HOSTDEVICE auto abs(const T& a)
{
    return OperatorAbs<T>::call(a);
}

template<typename T, typename U>
CUDIFF_HOSTDEVICE auto pow(const T& f, const U& g)
{
    return OperatorPow<T, U>::call(f, g);
}

template<typename T>
CUDIFF_HOSTDEVICE auto sign(const T& x)
{
    return OperatorSign<T>::call(x);
}

template<typename T>
CUDIFF_HOSTDEVICE auto floor(const T& x)
{
    return OperatorFloor<T>::call(x);
}

template<typename T>
CUDIFF_HOSTDEVICE auto ceil(const T& x)
{
    return OperatorCeil<T>::call(x);
}

template<typename T, typename U>
CUDIFF_HOSTDEVICE auto min(const T& a, const U& b)
{
    return OperatorMin<T, U>::call(a, b);
}

template<typename T, typename U>
CUDIFF_HOSTDEVICE auto max(const T& a, const U& b)
{
    return OperatorMax<T, U>::call(a, b);
}

template<typename T, typename U, typename V>
CUDIFF_HOSTDEVICE auto clamp(const T& x, const U& lo, const V& hi)
{
    return OperatorClamp<T, U, V>::call(x, lo, hi);
}
}    // namespace CuDiff