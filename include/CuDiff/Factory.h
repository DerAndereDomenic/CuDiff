#pragma once

#include <CuDiff/Dual.h>
#include <CuDiff/Traits.h>
#include <CuDiff/VarLayout.h>
#include <CuDiff/Tuple.h>

namespace CuDiff
{

namespace impl
{
template<int N>
struct make_variables_impl
{
    template<typename... Ts, std::size_t... Is>
    CUDIFF_HOSTDEVICE static auto apply(std::index_sequence<Is...>, const size_t starts[sizeof...(Ts)], Ts&&... values)
    {
        return CuDiff::make_tuple(Dual<N, std::decay_t<Ts>>(std::forward<Ts>(values), starts[Is])...);
    }
};
}    // namespace impl

template<int N, typename... Ts>
CUDIFF_HOSTDEVICE auto make_variables(Ts&&... values)
{
    VarLayout layout;
    size_t starts[sizeof...(Ts)] = {layout.template alloc<std::decay_t<Ts>>()...};
    return impl::make_variables_impl<N>::apply(std::index_sequence_for<Ts...> {}, starts, std::forward<Ts>(values)...);
}
}    // namespace CuDiff