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
    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const Dual<N, glm::vec<1, Q, P>>& v, const Dual<N, glm::vec<1, Q, P>>& w)
    {
        auto& val_v = v.val();
        auto& val_w = w.val();
        Dual<N, Q> res(glm::dot(val_v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, (v.derivative(i).x * val_w.x + val_v.x * w.derivative(i).x));
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const Dual<N, glm::vec<1, Q, P>>& v, const glm::vec<1, Q, P>& w)
    {
        auto& val_v = v.val();
        Dual<N, Q> res(glm::dot(val_v, w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, (v.derivative(i).x * w.x));
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const glm::vec<1, Q, P>& v, const Dual<N, glm::vec<1, Q, P>>& w)
    {
        auto& val_w = w.val();
        Dual<N, Q> res(glm::dot(v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, (v.x * w.derivative(i).x));
        }

        return res;
    }
};

template<int N, typename Q, glm::qualifier P>
struct OperatorDot<N, 2, Q, P>
{
    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const Dual<N, glm::vec<2, Q, P>>& v, const Dual<N, glm::vec<2, Q, P>>& w)
    {
        auto& val_v = v.val();
        auto& val_w = w.val();
        Dual<N, Q> res(glm::dot(val_v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i,
                              (v.derivative(i).x * val_w.x + val_v.x * w.derivative(i).x) +
                                  (v.derivative(i).y * val_w.y + val_v.y * w.derivative(i).y));
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const Dual<N, glm::vec<2, Q, P>>& v, const glm::vec<2, Q, P>& w)
    {
        auto& val_v = v.val();
        Dual<N, Q> res(glm::dot(val_v, w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, (v.derivative(i).x * w.x) + (v.derivative(i).y * w.y));
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const glm::vec<2, Q, P>& v, const Dual<N, glm::vec<2, Q, P>>& w)
    {
        auto& val_w = w.val();
        Dual<N, Q> res(glm::dot(v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, (v.x * w.derivative(i).x) + (v.y * w.derivative(i).y));
        }

        return res;
    }
};

template<int N, typename Q, glm::qualifier P>
struct OperatorDot<N, 3, Q, P>
{
    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const Dual<N, glm::vec<3, Q, P>>& v, const Dual<N, glm::vec<3, Q, P>>& w)
    {
        auto& val_v = v.val();
        auto& val_w = w.val();
        Dual<N, Q> res(glm::dot(val_v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i,
                              (v.derivative(i).x * val_w.x + val_v.x * w.derivative(i).x) +
                                  (v.derivative(i).y * val_w.y + val_v.y * w.derivative(i).y) +
                                  (v.derivative(i).z * val_w.z + val_v.z * w.derivative(i).z));
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const Dual<N, glm::vec<3, Q, P>>& v, const glm::vec<3, Q, P>& w)
    {
        auto& val_v = v.val();
        Dual<N, Q> res(glm::dot(val_v, w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, (v.derivative(i).x * w.x) + (v.derivative(i).y * w.y) + (v.derivative(i).z * w.z));
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const glm::vec<3, Q, P>& v, const Dual<N, glm::vec<3, Q, P>>& w)
    {
        auto& val_w = w.val();
        Dual<N, Q> res(glm::dot(v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i, (v.x * w.derivative(i).x) + (v.y * w.derivative(i).y) + (v.z * w.derivative(i).z));
        }

        return res;
    }
};

template<int N, typename Q, glm::qualifier P>
struct OperatorDot<N, 4, Q, P>
{
    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const Dual<N, glm::vec<4, Q, P>>& v, const Dual<N, glm::vec<4, Q, P>>& w)
    {
        auto& val_v = v.val();
        auto& val_w = w.val();
        Dual<N, Q> res(glm::dot(val_v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i,
                              (v.derivative(i).x * val_w.x + val_v.x * w.derivative(i).x) +
                                  (v.derivative(i).y * val_w.y + val_v.y * w.derivative(i).y) +
                                  (v.derivative(i).z * val_w.z + val_v.z * w.derivative(i).z) +
                                  (v.derivative(i).w * val_w.w + val_v.w * w.derivative(i).w));
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const Dual<N, glm::vec<4, Q, P>>& v, const glm::vec<4, Q, P>& w)
    {
        auto& val_v = v.val();
        Dual<N, Q> res(glm::dot(val_v, w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i,
                              (v.derivative(i).x * w.x) + (v.derivative(i).y * w.y) + (v.derivative(i).z * w.z) +
                                  (v.derivative(i).w * w.w));
        }

        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const glm::vec<4, Q, P>& v, const Dual<N, glm::vec<4, Q, P>>& w)
    {
        auto& val_w = w.val();
        Dual<N, Q> res(glm::dot(v, val_w));

        for(int i = 0; i < N; ++i)
        {
            res.setDerivative(i,
                              (v.x * w.derivative(i).x) + (v.y * w.derivative(i).y) + (v.z * w.derivative(i).z) +
                                  (v.w * w.derivative(i).w));
        }

        return res;
    }
};

template<int N, int M, typename Q, glm::qualifier P>
struct OperatorLength
{
    CUDIFF_HOSTDEVICE static Dual<N, Q> call(const Dual<N, glm::vec<M, Q, P>>& v) { return sqrt(dot(v, v)); }
};

template<int N, int M, typename Q, glm::qualifier P>
struct OperatorNormalize
{
    CUDIFF_HOSTDEVICE static Dual<N, glm::vec<M, Q, P>> call(const Dual<N, glm::vec<M, Q, P>>& v)
    {
        return v / length(v);
    }
};

template<int N, int M, typename Q, glm::qualifier P>
struct OperatorClamp<N, glm::vec<M, Q, P>>
{
    using T = glm::vec<M, Q, P>;
    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x, const Dual<N, T>& lo, const Dual<N, T>& hi)
    {
        const auto& x_val  = x.val();
        const auto& lo_val = lo.val();
        const auto& hi_val = hi.val();
        Dual<N, T> res;
        for(int j = 0; j < M; ++j)
        {
            if(x_val[j] < lo_val[j])
            {
                res.mut_val()[j] = lo_val[j];
                for(int i = 0; i < N; ++i)
                    res.mut_derivative(i)[j] = lo.derivative(i)[j];
            }
            else if(x_val[j] > hi_val[j])
            {
                res.mut_val()[j] = hi_val[j];
                for(int i = 0; i < N; ++i)
                    res.mut_derivative(i)[j] = hi.derivative(i)[j];
            }
            else
            {
                res.mut_val()[j] = x_val[j];
                for(int i = 0; i < N; ++i)
                    res.mut_derivative(i)[j] = x.derivative(i)[j];
            }
        }
        return res;
    }

    CUDIFF_HOSTDEVICE static Dual<N, T> call(const Dual<N, T>& x, const T& lo, const T& hi)
    {
        const auto& x_val = x.val();
        Dual<N, T> res;
        for(int j = 0; j < M; ++j)
        {
            if(x_val[j] < lo[j])
            {
                res.mut_val()[j] = lo[j];
                for(int i = 0; i < N; ++i)
                    res.mut_derivative(i)[j] = Q(0);
            }
            else if(x_val[j] > hi[j])
            {
                res.mut_val()[j] = hi[j];
                for(int i = 0; i < N; ++i)
                    res.mut_derivative(i)[j] = Q(0);
            }
            else
            {
                res.mut_val()[j] = x_val[j];
                for(int i = 0; i < N; ++i)
                    res.mut_derivative(i)[j] = x.derivative(i)[j];
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
        return v - n * dot(n, v) * dual_value_t<vType>(2);
    }
};

template<typename IType, typename NType, typename etaType>
struct OperatorRefract
{
    CUDIFF_HOSTDEVICE static auto call(const IType& I, const NType& N, const etaType& eta)
    {
        using Q             = dual_value_t<etaType>;
        const auto dotValue = dot(N, I);
        const auto k        = (Q(1) - eta * eta * (Q(1) - dotValue * dotValue));
        auto result         = (k.val() >= Q(0)) ? (eta * I - (eta * dotValue + sqrt(k)) * N)
                                                : decltype((eta * I - (eta * dotValue + sqrt(k)) * N))();
        return result;
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

template<int N, int M, typename Q, glm::qualifier P>
CUDIFF_HOSTDEVICE Dual<N, Q> length(const Dual<N, glm::vec<M, Q, P>>& v)
{
    return OperatorLength<N, M, Q, P>::call(v);
}

template<int N, int M, typename Q, glm::qualifier P>
CUDIFF_HOSTDEVICE Dual<N, glm::vec<M, Q, P>> normalize(const Dual<N, glm::vec<M, Q, P>>& v)
{
    return OperatorNormalize<N, M, Q, P>::call(v);
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