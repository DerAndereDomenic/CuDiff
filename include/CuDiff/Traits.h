#pragma once

#include <CuDiff/Platform.h>

#include <type_traits>

namespace CuDiff
{
template<typename T, typename = void>
struct DerivativeTraits;

template<typename T>
struct DerivativeTraits<T, std::enable_if_t<std::is_arithmetic_v<T>>>
{
    static CUDIFF_HOSTDEVICE constexpr size_t components() { return 1; }
    static CUDIFF_HOSTDEVICE T unit(size_t) { return T(1); }
};
}    // namespace CuDiff