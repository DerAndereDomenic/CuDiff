#pragma once

#include <CuDiff/Platform.h>

#ifdef CUDIFF_CUDA_AVAILABLE
    #include <thrust/tuple.h>

namespace CuDiff
{
template<typename... Types>
using Tuple = thrust::tuple<Types...>;

template<typename... Args>
CUDIFF_HOSTDEVICE constexpr auto make_tuple(Args&&... args)
{
    return thrust::make_tuple(std::forward<Args>(args)...);
}
}    // namespace CuDiff


#else
    #include <tuple>

namespace CuDiff
{

template<typename... Types>
using Tuple = std::tuple<Types...>;


template<typename... Args>
constexpr auto make_tuple(Args&&... args)
{
    return std::make_tuple(std::forward<Args>(args)...);
}
}    // namespace CuDiff


#endif