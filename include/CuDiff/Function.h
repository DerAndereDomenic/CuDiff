#pragma once

#include <CuDiff/Dual.h>

namespace CuDiff
{

template<int N, typename T, bool _stochasticT>
struct OperatorSqrt
{
    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x)
    {
        auto sq  = std::sqrt(x.val());
        auto res = Dual<N, T, _stochasticT>(sq);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, x.derivative(i) / (T(2) * sq));
            if constexpr(_stochasticT)
            {
                res.setScore(i, x.score(i));
            }
        }

        return res;
    }
};

template<int N, typename T, bool _stochasticT>
struct OperatorSin
{
    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x)
    {
        auto sin = std::sin(x.val());
        auto cos = std::cos(x.val());
        auto res = Dual<N, T, _stochasticT>(sin);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, x.derivative(i) * cos);
            if constexpr(_stochasticT)
            {
                res.setScore(i, x.score(i));
            }
        }

        return res;
    }
};

template<int N, typename T, bool _stochasticT>
struct OperatorCos
{
    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x)
    {
        auto cos = std::cos(x.val());
        auto sin = std::sin(x.val());
        auto res = Dual<N, T, _stochasticT>(cos);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, -x.derivative(i) * sin);
            if constexpr(_stochasticT)
            {
                res.setScore(i, x.score(i));
            }
        }

        return res;
    }
};

template<int N, typename T, bool _stochasticT>
struct OperatorTan
{
    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x)
    {
        return OperatorSin<N, T, _stochasticT>::call(x) / OperatorCos<N, T, _stochasticT>::call(x);
    }
};

template<int N, typename T, bool _stochasticT>
struct OperatorAsin
{
    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x)
    {
        auto val   = std::asin(x.val());
        auto denom = std::sqrt(T(1) - x.val() * x.val());
        auto res   = Dual<N, T, _stochasticT>(val);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, x.derivative(i) / denom);
            if constexpr(_stochasticT)
            {
                res.setScore(i, x.score(i));
            }
        }

        return res;
    }
};

template<int N, typename T, bool _stochasticT>
struct OperatorAcos
{
    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x)
    {
        auto val   = std::acos(x.val());
        auto denom = std::sqrt(T(1) - x.val() * x.val());
        auto res   = Dual<N, T, _stochasticT>(val);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, -x.derivative(i) / denom);
            if constexpr(_stochasticT)
            {
                res.setScore(i, x.score(i));
            }
        }

        return res;
    }
};

template<int N, typename T, bool _stochasticT>
struct OperatorAtan
{
    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x)
    {
        auto val   = std::atan(x.val());
        auto denom = T(1) + x.val() * x.val();
        auto res   = Dual<N, T, _stochasticT>(val);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, x.derivative(i) / denom);
            if constexpr(_stochasticT)
            {
                res.setScore(i, x.score(i));
            }
        }

        return res;
    }
};

template<int N, typename T, bool _stochasticT, bool _stochasticU>
struct OperatorAtan2
{
    static constexpr bool stochasticR = (_stochasticT || _stochasticU);
    CUDIFF_HOSTDEVICE static Dual<N, T, stochasticR> call(const Dual<N, T, _stochasticT>& y,
                                                          const Dual<N, T, _stochasticU>& x)
    {
        auto val   = std::atan2(y.val(), x.val());
        auto denom = x.val() * x.val() + y.val() * y.val();
        auto res   = Dual<N, T, stochasticR>(val);

        for(int i = 0; i < N; ++i)
        {
            T dy = y.derivative(i);
            T dx = x.derivative(i);
            res.setDerivative(i, (x.val() * dy - y.val() * dx) / denom);
            if constexpr(stochasticR)
            {
                if constexpr(_stochasticT)
                {
                    res.mut_score(i) += y.score(i);
                }
                if constexpr(_stochasticU)
                {
                    res.mut_score(i) += x.score(i);
                }
            }
        }

        return res;
    }
};

template<int N, typename T, bool _stochasticT>
struct OperatorExp
{
    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x)
    {
        auto val = std::exp(x.val());
        auto res = Dual<N, T, _stochasticT>(val);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, val * x.derivative(i));
            if constexpr(_stochasticT)
            {
                res.setScore(i, x.score(i));
            }
        }

        return res;
    }
};

template<int N, typename T, bool _stochasticT>
struct OperatorLog
{
    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x)
    {
        auto val = std::log(x.val());
        auto res = Dual<N, T, _stochasticT>(val);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, x.derivative(i) / x.val());
            if constexpr(_stochasticT)
            {
                res.setScore(i, x.score(i));
            }
        }

        return res;
    }
};

template<int N, typename T, bool _stochasticT>
struct OperatorAbs
{
    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x)
    {
        auto val = std::abs(x.val());
        auto res = Dual<N, T, _stochasticT>(val);

        T sign = (x.val() > T(0)) ? T(1) : (x.val() < T(0)) ? T(-1) : T(0);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, sign * x.derivative(i));
            if constexpr(_stochasticT)
            {
                res.setScore(i, x.score(i));
            }
        }

        return res;
    }
};

template<int N, typename T, bool _stochasticT, bool _stochasticU>
struct OperatorPow
{
    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x, const T& n)
    {
        auto val = std::pow(x.val(), n);
        auto res = Dual<N, T, _stochasticT>(val);
        auto pow = std::pow(x.val(), n - T(1));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, n * pow * x.derivative(i));
            if constexpr(_stochasticT)
            {
                res.setScore(i, x.score(i));
            }
        }

        return res;
    }

    static constexpr bool stochasticR = (_stochasticT || _stochasticU);
    CUDIFF_HOSTDEVICE static Dual<N, T, stochasticR> call(const Dual<N, T, _stochasticT>& f,
                                                          const Dual<N, T, _stochasticU>& g)
    {
        auto val = std::pow(f.val(), g.val());
        auto res = Dual<N, T, stochasticR>(val);
        auto log = std::log(f.val());

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, val * (f.derivative(i) / f.val() * g.val() + log * g.derivative(i)));
            if constexpr(stochasticR)
            {
                if constexpr(_stochasticT)
                {
                    res.mut_score(i) += f.score(i);
                }
                if constexpr(_stochasticU)
                {
                    res.mut_score(i) += g.score(i);
                }
            }
        }

        return res;
    }
};

template<int N, typename T, bool _stochasticT>
struct OperatorSign
{
    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x)
    {
        auto v = x.val();
        auto s = (v > T(0)) ? T(1) : (v < T(0)) ? T(-1) : T(0);

        Dual<N, T, _stochasticT> res(s);
        // Derivative of sign(x) is zero almost everywhere
        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, T(0));
            if constexpr(_stochasticT)
            {
                res.setScore(i, x.score(i));
            }
        }
        return res;
    }
};

template<int N, typename T, bool _stochasticT>
struct OperatorFloor
{
    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x)
    {
        Dual<N, T, _stochasticT> res(std::floor(x.val()));
        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, T(0));
            if constexpr(_stochasticT)
            {
                res.setScore(i, x.score(i));
            }
        }
        return res;
    }
};

template<int N, typename T, bool _stochasticT>
struct OperatorCeil
{
    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x)
    {
        Dual<N, T, _stochasticT> res(std::ceil(x.val()));
        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, T(0));
            if constexpr(_stochasticT)
            {
                res.setScore(i, x.score(i));
            }
        }
        return res;
    }
};

template<int N, typename T, bool _stochasticT, bool _stochasticU>
struct OperatorMin
{
    static constexpr bool stochasticR = (_stochasticT || _stochasticU);
    CUDIFF_HOSTDEVICE static Dual<N, T, stochasticR> call(const Dual<N, T, _stochasticT>& a,
                                                          const Dual<N, T, _stochasticU>& b)
    {
        bool takeA = a.val() < b.val();
        Dual<N, T, stochasticR> res(takeA ? a.val() : b.val());

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, takeA ? a.derivative(i) : b.derivative(i));
            if constexpr(stochasticR)
            {
                if constexpr(_stochasticT)
                {
                    res.mut_score(i) += a.score(i);
                }
                if constexpr(_stochasticU)
                {
                    res.mut_score(i) += b.score(i);
                }
            }
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& a, const T& b)
    {
        bool takeA = a.val() < b;
        Dual<N, T, _stochasticT> res(takeA ? a.val() : b);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, takeA ? a.derivative(i) : T(0));
            if constexpr(_stochasticT)
            {
                res.setScore(i, a.score(i));
            }
        }

        return res;
    }
};

template<int N, typename T, bool _stochasticT, bool _stochasticU>
struct OperatorMax
{
    static constexpr bool stochasticR = (_stochasticT || _stochasticU);
    CUDIFF_HOSTDEVICE static Dual<N, T, stochasticR> call(const Dual<N, T, _stochasticT>& a,
                                                          const Dual<N, T, _stochasticU>& b)
    {
        bool takeA = a.val() > b.val();
        Dual<N, T, stochasticR> res(takeA ? a.val() : b.val());

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, takeA ? a.derivative(i) : b.derivative(i));
            if constexpr(stochasticR)
            {
                if constexpr(_stochasticT)
                {
                    res.mut_score(i) += a.score(i);
                }
                if constexpr(_stochasticU)
                {
                    res.mut_score(i) += b.score(i);
                }
            }
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& a, const T& b)
    {
        bool takeA = a.val() > b;
        Dual<N, T, _stochasticT> res(takeA ? a.val() : b);

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, takeA ? a.derivative(i) : T(0));
            if constexpr(_stochasticT)
            {
                res.setScore(i, a.score(i));
            }
        }

        return res;
    }
};

template<int N, typename T, bool _stochasticT, bool _stochasticU, bool _stochasticK>
struct OperatorClamp
{
    static constexpr bool stochasticR = (_stochasticT || _stochasticU || _stochasticK);
    CUDIFF_HOSTDEVICE static Dual<N, T, stochasticR>
    call(const Dual<N, T, _stochasticT>& x, const Dual<N, T, _stochasticU>& lo, const Dual<N, T, _stochasticK>& hi)
    {
        if(x.val() < lo.val())
        {
            Dual<N, T, stochasticR> res(lo.val());
            for(int i = 0; i < N; ++i)
            {
                res.setDerivative(i, lo.derivative(i));
                if constexpr(stochasticR)
                {
                    if constexpr(_stochasticT)
                    {
                        res.mut_score(i) += x.score(i);
                    }
                    if constexpr(_stochasticU)
                    {
                        res.mut_score(i) += lo.score(i);
                    }
                    if constexpr(_stochasticR)
                    {
                        res.mut_score(i) += hi.score(i);
                    }
                }
            }
            return res;
        }
        else if(x.val() > hi.val())
        {
            Dual<N, T, stochasticR> res(hi.val());
            for(int i = 0; i < N; ++i)
            {
                res.setDerivative(i, hi.derivative(i));
                if constexpr(stochasticR)
                {
                    if constexpr(_stochasticT)
                    {
                        res.mut_score(i) += x.score(i);
                    }
                    if constexpr(_stochasticU)
                    {
                        res.mut_score(i) += lo.score(i);
                    }
                    if constexpr(_stochasticR)
                    {
                        res.mut_score(i) += hi.score(i);
                    }
                }
            }
            return res;
        }
        else
        {
            Dual<N, T, stochasticR> res(x.val());
            for(int i = 0; i < N; ++i)
            {
                res.setDerivative(i, x.derivative(i));
                if constexpr(stochasticR)
                {
                    if constexpr(_stochasticT)
                    {
                        res.mut_score(i) += x.score(i);
                    }
                    if constexpr(_stochasticU)
                    {
                        res.mut_score(i) += lo.score(i);
                    }
                    if constexpr(_stochasticR)
                    {
                        res.mut_score(i) += hi.score(i);
                    }
                }
            }
            return res;
        }
    }

    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x, const T& lo, const T& hi)
    {
        if(x.val() < lo)
        {
            Dual<N, T, _stochasticT> res(lo);
            for(int i = 0; i < N; ++i)
            {
                res.setDerivative(i, T(0));
                if constexpr(_stochasticT)
                {
                    res.setScore(i, x.score(i));
                }
            }
            return res;
        }
        else if(x.val() > hi)
        {
            Dual<N, T, _stochasticT> res(hi);
            for(int i = 0; i < N; ++i)
            {
                res.setDerivative(i, T(0));
                if constexpr(_stochasticT)
                {
                    res.setScore(i, x.score(i));
                }
            }
            return res;
        }
        else
        {
            Dual<N, T, _stochasticT> res(x.val());
            for(int i = 0; i < N; ++i)
            {
                res.setDerivative(i, x.derivative(i));
                if constexpr(_stochasticT)
                {
                    res.setScore(i, x.score(i));
                }
            }
            return res;
        }
    }
};

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> sqrt(const Dual<N, T, _stochasticT>& a)
{
    return OperatorSqrt<N, T, _stochasticT>::call(a);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> sin(const Dual<N, T, _stochasticT>& a)
{
    return OperatorSin<N, T, _stochasticT>::call(a);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> cos(const Dual<N, T, _stochasticT>& a)
{
    return OperatorCos<N, T, _stochasticT>::call(a);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> tan(const Dual<N, T, _stochasticT>& a)
{
    return OperatorTan<N, T, _stochasticT>::call(a);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> asin(const Dual<N, T, _stochasticT>& a)
{
    return OperatorAsin<N, T, _stochasticT>::call(a);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> acos(const Dual<N, T, _stochasticT>& a)
{
    return OperatorAcos<N, T, _stochasticT>::call(a);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> atan(const Dual<N, T, _stochasticT>& a)
{
    return OperatorAtan<N, T, _stochasticT>::call(a);
}

template<int N, typename T, bool _stochasticT, bool _stochasticU>
CUDIFF_HOSTDEVICE Dual<N, T, (_stochasticT || _stochasticU)> atan2(const Dual<N, T, _stochasticT>& y,
                                                                   const Dual<N, T, _stochasticU>& x)
{
    return OperatorAtan2<N, T, _stochasticT, _stochasticU>::call(y, x);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> exp(const Dual<N, T, _stochasticT>& a)
{
    return OperatorExp<N, T, _stochasticT>::call(a);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> log(const Dual<N, T, _stochasticT>& a)
{
    return OperatorLog<N, T, _stochasticT>::call(a);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> abs(const Dual<N, T, _stochasticT>& a)
{
    return OperatorAbs<N, T, _stochasticT>::call(a);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> pow(const Dual<N, T, _stochasticT>& a, const T& n)
{
    return OperatorPow<N, T, _stochasticT, false>::call(a, n);
}

template<int N, typename T, bool _stochasticT, bool _stochasticU>
CUDIFF_HOSTDEVICE Dual<N, T, (_stochasticT || _stochasticU)> pow(const Dual<N, T, _stochasticT>& f,
                                                                 const Dual<N, T, _stochasticU>& g)
{
    return OperatorPow<N, T, _stochasticT, _stochasticU>::call(f, g);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> sign(const Dual<N, T, _stochasticT>& x)
{
    return OperatorSign<N, T, _stochasticT>::call(x);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> floor(const Dual<N, T, _stochasticT>& x)
{
    return OperatorFloor<N, T, _stochasticT>::call(x);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> ceil(const Dual<N, T, _stochasticT>& x)
{
    return OperatorCeil<N, T, _stochasticT>::call(x);
}

template<int N, typename T, bool _stochasticT, bool _stochasticU>
CUDIFF_HOSTDEVICE Dual<N, T, (_stochasticT || _stochasticU)> min(const Dual<N, T, _stochasticT>& a,
                                                                 const Dual<N, T, _stochasticU>& b)
{
    return OperatorMin<N, T, _stochasticT, _stochasticU>::call(a, b);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> min(const Dual<N, T, _stochasticT>& a, const T& b)
{
    return OperatorMin<N, T, _stochasticT, false>::call(a, b);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> min(const T& a, const Dual<N, T, _stochasticT>& b)
{
    return OperatorMin<N, T, _stochasticT, false>::call(b, a);
}

template<int N, typename T, bool _stochasticT, bool _stochasticU>
CUDIFF_HOSTDEVICE Dual<N, T, (_stochasticT || _stochasticU)> max(const Dual<N, T, _stochasticT>& a,
                                                                 const Dual<N, T, _stochasticU>& b)
{
    return OperatorMax<N, T, _stochasticT, _stochasticU>::call(a, b);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> max(const Dual<N, T, _stochasticT>& a, const T& b)
{
    return OperatorMax<N, T, _stochasticT, false>::call(a, b);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> max(const T& a, const Dual<N, T, _stochasticT>& b)
{
    return OperatorMax<N, T, _stochasticT, false>::call(b, a);
}

template<int N, typename T, bool _stochasticT, bool _stochasticU, bool _stochasticK>
CUDIFF_HOSTDEVICE Dual<N, T, (_stochasticT || _stochasticU || _stochasticK)>
clamp(const Dual<N, T, _stochasticT>& x, const Dual<N, T, _stochasticU>& lo, const Dual<N, T, _stochasticK>& hi)
{
    return OperatorClamp<N, T, _stochasticT, _stochasticU, _stochasticK>::call(x, lo, hi);
}

template<int N, typename T, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, T, _stochasticT> clamp(const Dual<N, T, _stochasticT>& x, const T& lo, const T& hi)
{
    return OperatorClamp<N, T, _stochasticT, false, false>::call(x, lo, hi);
}
}    // namespace CuDiff