#pragma once

#include <CuDiff/Dual.h>
#include <CuDiff/Traits.h>
#include <CuDiff/VarLayout.h>
#include <CuDiff/Tuple.h>

namespace CuDiff
{

template<int N, typename... Ts, std::size_t... I>
CUDIFF_HOSTDEVICE auto
make_variables_impl(std::index_sequence<I...>, const size_t starts[sizeof...(Ts)], Ts&&... values)
{
    return CuDiff::make_tuple(Dual<N, std::decay_t<Ts>>(std::forward<Ts>(values), starts[I])...);
}

template<int N, typename... Ts>
CUDIFF_HOSTDEVICE auto make_variables(Ts&&... values)
{
    VarLayout layout;
    size_t starts[sizeof...(Ts)] = {layout.template alloc<std::decay_t<Ts>>()...};
    return make_variables_impl<N>(std::index_sequence_for<Ts...> {}, starts, std::forward<Ts>(values)...);
}
}    // namespace CuDiff