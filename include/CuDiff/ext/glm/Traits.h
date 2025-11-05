#pragma once

#include <CuDiff/Platform.h>

#include <type_traits>
#include <glm/glm.hpp>

namespace CuDiff
{

template<typename Q, glm::qualifier P>
struct DerivativeTraits<glm::vec<1, Q, P>, std::enable_if_t<std::is_arithmetic_v<Q>>>
{
    static CUDIFF_HOSTDEVICE constexpr size_t components() { return 1; }
    static CUDIFF_HOSTDEVICE glm::vec<1, Q, P> unit(size_t) { return glm::vec<1, Q, P>(1); }
};

template<typename Q, glm::qualifier P>
struct DerivativeTraits<glm::vec<2, Q, P>, std::enable_if_t<std::is_arithmetic_v<Q>>>
{
    static CUDIFF_HOSTDEVICE constexpr size_t components() { return 2; }
    static CUDIFF_HOSTDEVICE glm::vec<2, Q, P> unit(size_t i)
    {
        return i == 0 ? glm::vec<2, Q, P>(Q(1), Q(0)) : glm::vec<2, Q, P>(Q(0), Q(1));
    }
};

template<typename Q, glm::qualifier P>
struct DerivativeTraits<glm::vec<3, Q, P>, std::enable_if_t<std::is_arithmetic_v<Q>>>
{
    static CUDIFF_HOSTDEVICE constexpr size_t components() { return 3; }
    static CUDIFF_HOSTDEVICE glm::vec<3, Q, P> unit(size_t i)
    {
        switch(i)
        {
            case 0:
                return glm::vec<3, Q, P>(Q(1), Q(0), Q(0));
            case 1:
                return glm::vec<3, Q, P>(Q(0), Q(1), Q(0));
            case 2:
                return glm::vec<3, Q, P>(Q(0), Q(0), Q(1));
        }
        return glm::vec<3, Q, P>(0);
    }
};

template<typename Q, glm::qualifier P>
struct DerivativeTraits<glm::vec<4, Q, P>, std::enable_if_t<std::is_arithmetic_v<Q>>>
{
    static CUDIFF_HOSTDEVICE constexpr size_t components() { return 4; }
    static CUDIFF_HOSTDEVICE glm::vec<4, Q, P> unit(size_t i)
    {
        switch(i)
        {
            case 0:
                return glm::vec<4, Q, P>(Q(1), Q(0), Q(0), Q(0));
            case 1:
                return glm::vec<4, Q, P>(Q(0), Q(1), Q(0), Q(0));
            case 2:
                return glm::vec<4, Q, P>(Q(0), Q(0), Q(1), Q(0));
            case 3:
                return glm::vec<4, Q, P>(Q(0), Q(0), Q(0), Q(1));
        }
        return glm::vec<4, Q, P>(0);
    }
};
}    // namespace CuDiff