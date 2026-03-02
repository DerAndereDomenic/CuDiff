#pragma once

#include <CuDiff/Function.h>

#include <glm/glm.hpp>

namespace CuDiff
{

template<typename A, typename B, bool = !is_dual_v<A> && !is_dual_v<B>>
struct OperatorDot
{
    CUDIFF_HOSTDEVICE static auto call(const A& v, const B& w) { return glm::dot(v, w); }
};

template<typename A, typename B>
struct OperatorDot<A, B, false>
{
    CUDIFF_HOSTDEVICE
    static auto call(const A& v, const B& w)
    {
        auto val_v = value_of(v);
        auto val_w = value_of(w);

        auto val = glm::dot(val_v, val_w);

        constexpr int N = dual_component_count<std::conditional_t<is_dual_v<A>, A, B>>::num_variables;
        using ValueType = decltype(val);

        Dual<N, ValueType> res(val);

        for(int i = 0; i < N; ++i)
        {
            auto dv = derivative_of(v, i);
            auto dw = derivative_of(w, i);

            res.setDerivative(i, glm::dot(dv, val_w) + glm::dot(val_v, dw));
        }

        return res;
    }
};

template<typename T>
struct OperatorLength
{
    CUDIFF_HOSTDEVICE static auto call(const T& v)
    {
        return OperatorSqrt<decltype(OperatorDot<T, T>::call(v, v))>::call(OperatorDot<T, T>::call(v, v));
    }
};

template<typename T>
struct OperatorNormalize
{
    CUDIFF_HOSTDEVICE static auto call(const T& v) { return v / OperatorLength<T>::call(v); }
};

template<int N, int M, typename Q, glm::qualifier P>
struct OperatorClamp<Dual<N, glm::vec<M, Q, P>>, Dual<N, glm::vec<M, Q, P>>, Dual<N, glm::vec<M, Q, P>>, false>
{
    using T = glm::vec<M, Q, P>;
    template<typename Atype, typename Btype, typename Ctype>
    CUDIFF_HOSTDEVICE static auto call(const Atype& x, const Btype& lo, const Ctype& hi)
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
};

template<int N, int M, typename Q, glm::qualifier P>
struct OperatorClamp<Dual<N, glm::vec<M, Q, P>>, glm::vec<M, Q, P>, glm::vec<M, Q, P>, false>
{
    using T = glm::vec<M, Q, P>;
    template<typename Atype, typename Btype, typename Ctype>
    CUDIFF_HOSTDEVICE static auto call(const Atype& x, const Btype& lo, const Ctype& hi)
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
};

template<typename vType, typename nType>
struct OperatorReflect
{
    CUDIFF_HOSTDEVICE static auto call(const vType& v, const nType& n)
    {
        return v - n * OperatorDot<nType, vType>::call(n, v) * dual_value_type_t<vType>(2);
    }
};

template<typename IType, typename NType, typename etaType>
struct OperatorRefract
{
    CUDIFF_HOSTDEVICE static auto call(const IType& I, const NType& N, const etaType& eta)
    {
        using Q             = dual_value_type_t<etaType>;
        const auto dotValue = OperatorDot<NType, IType>::call(N, I);
        auto k              = (Q(1) - eta * eta * (Q(1) - dotValue * dotValue));
        auto result         = (value_of(k) >= Q(0))
                                  ? (eta * I - (eta * dotValue + OperatorSqrt<decltype(k)>::call(k)) * N)
                                  : decltype((eta * I - (eta * dotValue + OperatorSqrt<decltype(k)>::call(k)) * N))();

        return result;
    }
};

template<typename Atype, typename Btype>
CUDIFF_HOSTDEVICE auto dot(const Atype& v, const Btype& w)
{
    return OperatorDot<Atype, Btype>::call(v, w);
}

template<typename T>
CUDIFF_HOSTDEVICE auto length(const T& v)
{
    return OperatorLength<T>::call(v);
}

template<typename T>
CUDIFF_HOSTDEVICE auto normalize(const T& v)
{
    return OperatorNormalize<T>::call(v);
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