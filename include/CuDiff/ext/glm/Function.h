#pragma once

#include <CuDiff/Function.h>

#include <glm/glm.hpp>

namespace CuDiff
{
template<int N, int M, typename Q, glm::qualifier P, bool _stochasticT, bool _stochasticU>
struct OperatorDot;

template<int N, typename Q, glm::qualifier P, bool _stochasticT, bool _stochasticU>
struct OperatorDot<N, 1, Q, P, _stochasticT, _stochasticU>
{
    static constexpr bool stochasticR = (_stochasticT || _stochasticU);
    CUDIFF_HOSTDEVICE static Dual<N, Q, stochasticR> call(const Dual<N, glm::vec<1, Q, P>, _stochasticT>& v,
                                                          const Dual<N, glm::vec<1, Q, P>, _stochasticU>& w)
    {
        auto& val_v = v.val();
        auto& val_w = w.val();
        Dual<N, Q, stochasticR> res(glm::dot(val_v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, (v.derivative(i).x * val_w.x + val_v.x * w.derivative(i).x));
            if constexpr(stochasticR)
            {
                if constexpr(_stochasticT)
                {
                    res.mut_score(i) += v.score(i);
                }
                if constexpr(_stochasticU)
                {
                    res.mut_score(i) += w.score(i);
                }
            }
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q, _stochasticT> call(const Dual<N, glm::vec<1, Q, P>, _stochasticT>& v,
                                                           const glm::vec<1, Q, P>& w)
    {
        auto& val_v = v.val();
        Dual<N, Q, _stochasticT> res(glm::dot(val_v, w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, (v.derivative(i).x * w.x));
            if constexpr(_stochasticT)
            {
                res.mut_score(i) += v.score(i);
            }
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q, _stochasticU> call(const glm::vec<1, Q, P>& v,
                                                           const Dual<N, glm::vec<1, Q, P>, _stochasticU>& w)
    {
        auto& val_w = w.val();
        Dual<N, Q, _stochasticU> res(glm::dot(v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, (v.x * w.derivative(i).x));
            if constexpr(_stochasticU)
            {
                res.mut_score(i) += w.score(i);
            }
        }

        return res;
    }
};

template<int N, typename Q, glm::qualifier P, bool _stochasticT, bool _stochasticU>
struct OperatorDot<N, 2, Q, P, _stochasticT, _stochasticU>
{
    static constexpr bool stochasticR = (_stochasticT || _stochasticU);
    CUDIFF_HOSTDEVICE static Dual<N, Q, stochasticR> call(const Dual<N, glm::vec<2, Q, P>, _stochasticT>& v,
                                                          const Dual<N, glm::vec<2, Q, P>, _stochasticU>& w)
    {
        auto& val_v = v.val();
        auto& val_w = w.val();
        Dual<N, Q, stochasticR> res(glm::dot(val_v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i,
                              (v.derivative(i).x * val_w.x + val_v.x * w.derivative(i).x) +
                                  (v.derivative(i).y * val_w.y + val_v.y * w.derivative(i).y));
            if constexpr(stochasticR)
            {
                if constexpr(_stochasticT)
                {
                    res.mut_score(i) += v.score(i);
                }
                if constexpr(_stochasticU)
                {
                    res.mut_score(i) += w.score(i);
                }
            }
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q, _stochasticT> call(const Dual<N, glm::vec<2, Q, P>, _stochasticT>& v,
                                                           const glm::vec<2, Q, P>& w)
    {
        auto& val_v = v.val();
        Dual<N, Q, _stochasticT> res(glm::dot(val_v, w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, (v.derivative(i).x * w.x) + (v.derivative(i).y * w.y));
            if constexpr(_stochasticT)
            {
                res.mut_score(i) += v.score(i);
            }
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q, _stochasticU> call(const glm::vec<2, Q, P>& v,
                                                           const Dual<N, glm::vec<2, Q, P>, _stochasticU>& w)
    {
        auto& val_w = w.val();
        Dual<N, Q, _stochasticU> res(glm::dot(v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, (v.x * w.derivative(i).x) + (v.y * w.derivative(i).y));
            if constexpr(_stochasticU)
            {
                res.mut_score(i) += w.score(i);
            }
        }

        return res;
    }
};

template<int N, typename Q, glm::qualifier P, bool _stochasticT, bool _stochasticU>
struct OperatorDot<N, 3, Q, P, _stochasticT, _stochasticU>
{
    static constexpr bool stochasticR = (_stochasticT || _stochasticU);
    CUDIFF_HOSTDEVICE static Dual<N, Q, stochasticR> call(const Dual<N, glm::vec<3, Q, P>, _stochasticT>& v,
                                                          const Dual<N, glm::vec<3, Q, P>, _stochasticU>& w)
    {
        auto& val_v = v.val();
        auto& val_w = w.val();
        Dual<N, Q, stochasticR> res(glm::dot(val_v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i,
                              (v.derivative(i).x * val_w.x + val_v.x * w.derivative(i).x) +
                                  (v.derivative(i).y * val_w.y + val_v.y * w.derivative(i).y) +
                                  (v.derivative(i).z * val_w.z + val_v.z * w.derivative(i).z));

            if constexpr(stochasticR)
            {
                if constexpr(_stochasticT)
                {
                    res.mut_score(i) += v.score(i);
                }
                if constexpr(_stochasticU)
                {
                    res.mut_score(i) += w.score(i);
                }
            }
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q, _stochasticT> call(const Dual<N, glm::vec<3, Q, P>, _stochasticT>& v,
                                                           const glm::vec<3, Q, P>& w)
    {
        auto& val_v = v.val();
        Dual<N, Q, _stochasticT> res(glm::dot(val_v, w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, (v.derivative(i).x * w.x) + (v.derivative(i).y * w.y) + (v.derivative(i).z * w.z));
            if constexpr(_stochasticT)
            {
                res.mut_score(i) += v.score(i);
            }
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q, _stochasticU> call(const glm::vec<3, Q, P>& v,
                                                           const Dual<N, glm::vec<3, Q, P>, _stochasticU>& w)
    {
        auto& val_w = w.val();
        Dual<N, Q, _stochasticU> res(glm::dot(v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, (v.x * w.derivative(i).x) + (v.y * w.derivative(i).y) + (v.z * w.derivative(i).z));
            if constexpr(_stochasticU)
            {
                res.mut_score(i) += w.score(i);
            }
        }

        return res;
    }
};

template<int N, typename Q, glm::qualifier P, bool _stochasticT, bool _stochasticU>
struct OperatorDot<N, 4, Q, P, _stochasticT, _stochasticU>
{
    static constexpr bool stochasticR = (_stochasticT || _stochasticU);
    CUDIFF_HOSTDEVICE static Dual<N, Q, stochasticR> call(const Dual<N, glm::vec<4, Q, P>, _stochasticT>& v,
                                                          const Dual<N, glm::vec<4, Q, P>, _stochasticU>& w)
    {
        auto& val_v = v.val();
        auto& val_w = w.val();
        Dual<N, Q, stochasticR> res(glm::dot(val_v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i,
                              (v.derivative(i).x * val_w.x + val_v.x * w.derivative(i).x) +
                                  (v.derivative(i).y * val_w.y + val_v.y * w.derivative(i).y) +
                                  (v.derivative(i).z * val_w.z + val_v.z * w.derivative(i).z) +
                                  (v.derivative(i).w * val_w.w + val_v.w * w.derivative(i).w));
            if constexpr(stochasticR)
            {
                if constexpr(_stochasticT)
                {
                    res.mut_score(i) += v.score(i);
                }
                if constexpr(_stochasticU)
                {
                    res.mut_score(i) += w.score(i);
                }
            }
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q, _stochasticT> call(const Dual<N, glm::vec<4, Q, P>, _stochasticT>& v,
                                                           const glm::vec<4, Q, P>& w)
    {
        auto& val_v = v.val();
        Dual<N, Q, _stochasticT> res(glm::dot(val_v, w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i,
                              (v.derivative(i).x * w.x) + (v.derivative(i).y * w.y) + (v.derivative(i).z * w.z) +
                                  (v.derivative(i).w * w.w));
            if constexpr(_stochasticT)
            {
                res.mut_score(i) += v.score(i);
            }
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q, _stochasticU> call(const glm::vec<4, Q, P>& v,
                                                           const Dual<N, glm::vec<4, Q, P>, _stochasticU>& w)
    {
        auto& val_w = w.val();
        Dual<N, Q, _stochasticU> res(glm::dot(v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i,
                              (v.x * w.derivative(i).x) + (v.y * w.derivative(i).y) + (v.z * w.derivative(i).z) +
                                  (v.w * w.derivative(i).w));
            if constexpr(_stochasticU)
            {
                res.mut_score(i) += w.score(i);
            }
        }

        return res;
    }
};

template<int N, int M, typename Q, glm::qualifier P, bool _stochasticT, bool _stochasticU>
CUDIFF_HOSTDEVICE Dual<N, Q, (_stochasticT || _stochasticU)> dot(const Dual<N, glm::vec<M, Q, P>, _stochasticT>& v,
                                                                 const Dual<N, glm::vec<M, Q, P>, _stochasticU>& w)
{
    return OperatorDot<N, M, Q, P, _stochasticT, _stochasticU>::call(v, w);
}

template<int N, int M, typename Q, glm::qualifier P, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, Q, _stochasticT> dot(const Dual<N, glm::vec<M, Q, P>, _stochasticT>& v,
                                               const glm::vec<M, Q, P>& w)
{
    return OperatorDot<N, M, Q, P, _stochasticT, false>::call(v, w);
}

template<int N, int M, typename Q, glm::qualifier P, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, Q, _stochasticT> dot(const glm::vec<M, Q, P>& v,
                                               const Dual<N, glm::vec<M, Q, P>, _stochasticT>& w)
{
    return OperatorDot<N, M, Q, P, _stochasticT, false>::call(v, w);
}

template<int N, int M, typename Q, glm::qualifier P, bool _stochasticT>
struct OperatorLength
{
    CUDIFF_HOSTDEVICE static Dual<N, Q, _stochasticT> call(const Dual<N, glm::vec<M, Q, P>, _stochasticT>& v)
    {
        return sqrt(CuDiff::dot(v, v));
    }
};

template<int N, int M, typename Q, glm::qualifier P, bool _stochasticT>
struct OperatorNormalize
{
    CUDIFF_HOSTDEVICE static Dual<N, glm::vec<M, Q, P>, _stochasticT>
    call(const Dual<N, glm::vec<M, Q, P>, _stochasticT>& v)
    {
        return v / length(v);
    }
};

template<int N, int M, typename Q, glm::qualifier P, bool _stochasticT, bool _stochasticU, bool _stochasticK>
struct OperatorClamp<N, glm::vec<M, Q, P>, _stochasticT, _stochasticU, _stochasticK>
{
    using T                           = glm::vec<M, Q, P>;
    static constexpr bool stochasticR = (_stochasticT || _stochasticU || _stochasticK);
    CUDIFF_HOSTDEVICE static Dual<N, T, stochasticR>
    call(const Dual<N, T, _stochasticT>& x, const Dual<N, T, _stochasticU>& lo, const Dual<N, T, _stochasticK>& hi)
    {
        const auto& x_val  = x.val();
        const auto& lo_val = lo.val();
        const auto& hi_val = hi.val();
        Dual<N, T, stochasticR> res;
        for(int j = 0; j < M; ++j)
        {
            if(x_val[j] < lo_val[j])
            {
                res.mut_val()[j] = lo_val[j];
                for(int i = 0; i < N; ++i)
                {
                    res.mut_derivative(i)[j] = lo.derivative(i)[j];
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
            }
            else if(x_val[j] > hi_val[j])
            {
                res.mut_val()[j] = hi_val[j];
                for(int i = 0; i < N; ++i)
                {
                    res.mut_derivative(i)[j] = hi.derivative(i)[j];
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
            }
            else
            {
                res.mut_val()[j] = x_val[j];
                for(int i = 0; i < N; ++i)
                {
                    res.mut_derivative(i)[j] = x.derivative(i)[j];
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
            }
        }
        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, T, _stochasticT> call(const Dual<N, T, _stochasticT>& x, const T& lo, const T& hi)
    {
        const auto& x_val = x.val();
        Dual<N, T, _stochasticT> res;
        for(int j = 0; j < M; ++j)
        {
            if(x_val[j] < lo[j])
            {
                res.mut_val()[j] = lo[j];
                for(int i = 0; i < N; ++i)
                {
                    res.mut_derivative(i)[j] = Q(0);
                    if constexpr(_stochasticT)
                    {
                        res.mut_score(i) += x.score(i);
                    }
                }
            }
            else if(x_val[j] > hi[j])
            {
                res.mut_val()[j] = hi[j];
                for(int i = 0; i < N; ++i)
                {
                    res.mut_derivative(i)[j] = Q(0);
                    if constexpr(_stochasticT)
                    {
                        res.mut_score(i) += x.score(i);
                    }
                }
            }
            else
            {
                res.mut_val()[j] = x_val[j];
                for(int i = 0; i < N; ++i)
                {
                    res.mut_derivative(i)[j] = x.derivative(i)[j];
                    if constexpr(_stochasticT)
                    {
                        res.mut_score(i) += x.score(i);
                    }
                }
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
        return v - n * CuDiff::dot(n, v) * dual_value_t<vType>(2);
    }
};

template<typename IType, typename NType, typename etaType>
struct OperatorRefract
{
    CUDIFF_HOSTDEVICE static auto call(const IType& I, const NType& N, const etaType& eta)
    {
        using Q             = dual_value_t<etaType>;
        const auto dotValue = CuDiff::dot(N, I);
        const auto k        = (Q(1) - eta * eta * (Q(1) - dotValue * dotValue));
        auto result         = (k.val() >= Q(0)) ? (eta * I - (eta * dotValue + sqrt(k)) * N)
                                                : decltype((eta * I - (eta * dotValue + sqrt(k)) * N))();
        return result;
    }
};

template<int N, int M, typename Q, glm::qualifier P, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, Q, _stochasticT> length(const Dual<N, glm::vec<M, Q, P>, _stochasticT>& v)
{
    return OperatorLength<N, M, Q, P, _stochasticT>::call(v);
}

template<int N, int M, typename Q, glm::qualifier P, bool _stochasticT>
CUDIFF_HOSTDEVICE Dual<N, glm::vec<M, Q, P>, _stochasticT> normalize(const Dual<N, glm::vec<M, Q, P>, _stochasticT>& v)
{
    return OperatorNormalize<N, M, Q, P, _stochasticT>::call(v);
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