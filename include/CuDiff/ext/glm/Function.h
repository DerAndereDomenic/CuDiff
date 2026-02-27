#pragma once

#include <CuDiff/Function.h>

#include <glm/glm.hpp>

namespace CuDiff
{
template<int N, int M, typename Q, glm::qualifier P>
struct OperatorDot;

template<int N, typename Q, glm::qualifier P>
struct OperatorDot<N, 1, Q, P>
{
    template<typename Atype, typename Btype>
    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const Atype& v, const Btype& w)
    {
        auto val_v = value_of(v);
        auto val_w = value_of(w);
        Dual<N, Q> res(glm::dot(val_v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, (derivative_of(v, i).x * val_w.x + val_v.x * derivative_of(w, i).x));
        }

        return res;
    }

    template<typename Atype, typename = std::enable_if_t<!is_dual_v<Atype>>>
    CUDIFF_HOSTDEVICE static Q call(const Atype& v, const Atype& w)
    {
        return glm::dot(v, w);
    }
};

template<int N, typename Q, glm::qualifier P>
struct OperatorDot<N, 2, Q, P>
{
    template<typename Atype, typename Btype>
    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const Atype& v, const Btype& w)
    {
        auto val_v = value_of(v);
        auto val_w = value_of(w);
        Dual<N, Q> res(glm::dot(val_v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i,
                              (derivative_of(v, i).x * val_w.x + val_v.x * derivative_of(w, i).x) +
                                  (derivative_of(v, i).y * val_w.y + val_v.y * derivative_of(w, i).y));
        }

        return res;
    }

    template<typename Atype, typename = std::enable_if_t<!is_dual_v<Atype>>>
    CUDIFF_HOSTDEVICE static Q call(const Atype& v, const Atype& w)
    {
        return glm::dot(v, w);
    }
};

template<int N, typename Q, glm::qualifier P>
struct OperatorDot<N, 3, Q, P>
{
    template<typename Atype, typename Btype>
    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const Atype& v, const Btype& w)
    {
        auto val_v = value_of(v);
        auto val_w = value_of(w);
        Dual<N, Q> res(glm::dot(val_v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i,
                              (derivative_of(v, i).x * val_w.x + val_v.x * derivative_of(w, i).x) +
                                  (derivative_of(v, i).y * val_w.y + val_v.y * derivative_of(w, i).y) +
                                  (derivative_of(v, i).z * val_w.z + val_v.z * derivative_of(w, i).z));
        }

        return res;
    }

    template<typename Atype, typename = std::enable_if_t<!is_dual_v<Atype>>>
    CUDIFF_HOSTDEVICE static Q call(const Atype& v, const Atype& w)
    {
        return glm::dot(v, w);
    }
};

template<int N, typename Q, glm::qualifier P>
struct OperatorDot<N, 4, Q, P>
{
    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const Dual<N, glm::vec<4, Q, P>>& v, const Dual<N, glm::vec<4, Q, P>>& w)
    {
        auto val_v = value_of(v);
        auto val_w = value_of(w);
        Dual<N, Q> res(glm::dot(val_v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i,
                              (derivative_of(v, i).x * val_w.x + val_v.x * derivative_of(w, i).x) +
                                  (derivative_of(v, i).y * val_w.y + val_v.y * derivative_of(w, i).y) +
                                  (derivative_of(v, i).z * val_w.z + val_v.z * derivative_of(w, i).z) +
                                  (derivative_of(v, i).w * val_w.w + val_v.w * derivative_of(w, i).w));
        }

        return res;
    }

    template<typename Atype, typename = std::enable_if_t<!is_dual_v<Atype>>>
    CUDIFF_HOSTDEVICE static Q call(const Atype& v, const Atype& w)
    {
        return glm::dot(v, w);
    }
};

template<int N, int M, typename Q, glm::qualifier P>
struct OperatorLength
{
    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const Dual<N, glm::vec<M, Q, P>>& v) { return sqrt(dot(v, v)); }

    CUDIFF_HOSTDEVICE static Q call(const glm::vec<M, Q, P>& v) { return glm::length(v); }
};

template<int N, int M, typename Q, glm::qualifier P>
struct OperatorNormalize
{
    CUDIFF_HOSTDEVICE static Dual<N, glm::vec<M, Q, P>> call(const Dual<N, glm::vec<M, Q, P>>& v)
    {
        return v / length(v);
    }

    CUDIFF_HOSTDEVICE static glm::vec<M, Q, P> call(const glm::vec<M, Q, P>& v) { return glm::normalize(v); }
};

template<int N, int M, typename Q, glm::qualifier P>
struct OperatorClamp<N, glm::vec<M, Q, P>>
{
    using T = glm::vec<M, Q, P>;
    template<typename Atype, typename Btype, typename Ctype>
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Atype& x, const Btype& lo, const Ctype& hi)
    {
        const auto x_val  = value_of(x);
        const auto lo_val = value_of(lo);
        const auto hi_val = value_of(hi);
        Dual<N, T> res;
        for(int j = 0; j < M; ++j)
        {
            if(x_val[j] < lo_val[j])
            {
                res.mut_val()[j] = lo_val[j];
                for(int i = 0; i < N; ++i)
                    res.mut_derivative(i)[j] = derivative_of(lo, i)[j];
            }
            else if(x_val[j] > hi_val[j])
            {
                res.mut_val()[j] = hi_val[j];
                for(int i = 0; i < N; ++i)
                    res.mut_derivative(i)[j] = derivative_of(hi, i)[j];
            }
            else
            {
                res.mut_val()[j] = x_val[j];
                for(int i = 0; i < N; ++i)
                    res.mut_derivative(i)[j] = derivative_of(x, i)[j];
            }
        }
        return res;
    }

    template<typename Atype, typename = std::enable_if_t<!is_dual_v<Atype>>>
    CUDIFF_HOSTDEVICE static T call(const Atype& x, const Atype& lo, const Atype& hi)
    {
        T res;
        for(int j = 0; j < M; ++j)
        {
            if(x[j] < lo[j])
                res[j] = lo[j];
            else if(x[j] > hi[j])
                res[j] = hi[j];
            else
                res[j] = x[j];
        }
        return res;
    }
};

template<typename vType, typename nType>
struct OperatorReflect
{
    CUDIFF_HOSTDEVICE static auto call(const vType& v, const nType& n)
    {
        return v - n * dot(n, v) * dual_value_type_t<vType>(2);
    }

    template<typename vType, typename = std::enable_if_t<!is_dual_v<vType>>>
    CUDIFF_HOSTDEVICE static auto call(const vType& v, const vType& n)
    {
        return v - n * dot(n, v) * decltype(v)(2);
    }
};

template<typename IType, typename NType, typename etaType>
struct OperatorRefract
{
    CUDIFF_HOSTDEVICE static auto call(const IType& I, const NType& N, const etaType& eta)
    {
        using Q             = dual_value_type_t<etaType>;
        const auto dotValue = dot(N, I);
        const auto k        = (Q(1) - eta * eta * (Q(1) - dotValue * dotValue));
        auto result         = (k.val() >= Q(0)) ? (eta * I - (eta * dotValue + sqrt(k)) * N)
                                                : decltype((eta * I - (eta * dotValue + sqrt(k)) * N))();
        return result;
    }

    template<typename ITyp, typename = std::enable_if_t<!is_dual_v<ITyp>>>
    CUDIFF_HOSTDEVICE static auto call(const ITyp& I, const ITyp& N, const ITyp& eta)
    {
        return glm::refract(I, N, eta);
    }
};

template<int N, int M, typename Q, glm::qualifier P>
CUDIFF_HOSTDEVICE Dual<N, Q> dot(const Dual<N, glm::vec<M, Q, P>>& v, const Dual<N, glm::vec<M, Q, P>>& w)
{
    return OperatorDot<N, M, Q, P>::call(v, w);
}

template<int N, int M, typename Q, glm::qualifier P>
CUDIFF_HOSTDEVICE Dual<N, Q> dot(const Dual<N, glm::vec<M, Q, P>>& v, const glm::vec<M, Q, P>& w)
{
    return OperatorDot<N, M, Q, P>::call(v, w);
}

template<int N, int M, typename Q, glm::qualifier P>
CUDIFF_HOSTDEVICE Dual<N, Q> dot(const glm::vec<M, Q, P>& v, const Dual<N, glm::vec<M, Q, P>>& w)
{
    return OperatorDot<N, M, Q, P>::call(v, w);
}

template<int M, typename Q, glm::qualifier P>
CUDIFF_HOSTDEVICE auto dot(const glm::vec<M, Q, P>& v, const glm::vec<M, Q, P>& w)
{
    return OperatorDot<0, M, Q, P>::call(v, w);
}

template<int N, int M, typename Q, glm::qualifier P>
CUDIFF_HOSTDEVICE Dual<N, Q> length(const Dual<N, glm::vec<M, Q, P>>& v)
{
    return OperatorLength<N, M, Q, P>::call(v);
}

template<int M, typename Q, glm::qualifier P>
CUDIFF_HOSTDEVICE auto length(const glm::vec<M, Q, P>& v)
{
    return OperatorLength<0, M, Q, P>::call(v);
}

template<int N, int M, typename Q, glm::qualifier P>
CUDIFF_HOSTDEVICE Dual<N, glm::vec<M, Q, P>> normalize(const Dual<N, glm::vec<M, Q, P>>& v)
{
    return OperatorNormalize<N, M, Q, P>::call(v);
}

template<int M, typename Q, glm::qualifier P>
CUDIFF_HOSTDEVICE auto normalize(const glm::vec<M, Q, P>& v)
{
    return OperatorNormalize<0, M, Q, P>::call(v);
}

template<typename vType, typename nType>
CUDIFF_HOSTDEVICE auto reflect(const vType& v, const nType& n)
{
    return OperatorReflect<vType, nType>::call(v, n);
}

template<typename vType, typename nType, typename etaType>
CUDIFF_HOSTDEVICE auto refract(const vType& v, const nType& n, const etaType& eta)
{
    return OperatorRefract<vType, nType, etaType>::call(v, n, eta);
}
}    // namespace CuDiff