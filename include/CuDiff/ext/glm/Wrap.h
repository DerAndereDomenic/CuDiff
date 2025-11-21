#pragma once

#include <CuDiff/Dual.h>

#include <glm/glm.hpp>
#include <type_traits>

namespace CuDiff
{

namespace impl
{
template<int N, int M, typename Q, glm::qualifier P>
struct wrap_impl
{
    using InDual  = Dual<N, Q>;
    using OutDual = Dual<N, glm::vec<M, Q, P>>;

    template<std::size_t I, std::size_t J, typename Args>
    CUDIFF_HOSTDEVICE static void copy_one_derivative(OutDual& result, const Args& args)
    {
        result.mut_derivative(I)[J] = args[J].derivative(I);
    }

    template<std::size_t I, typename Args, std::size_t... Js>
    CUDIFF_HOSTDEVICE static void copy_derivatives(OutDual& result, const Args& args, std::index_sequence<Js...>)
    {
        (copy_one_derivative<I, Js>(result, args), ...);
    }

    template<typename Args, std::size_t... Is>
    CUDIFF_HOSTDEVICE static void apply(OutDual& result, const Args& args, std::index_sequence<Is...>)
    {
        (copy_derivatives<Is>(result, args, std::make_index_sequence<M> {}), ...);
    }
};
}    // namespace impl

template<typename First,
         typename... Rest,
         typename         = std::enable_if_t<(is_dual<std::decay_t<Rest>>::value && ...)>,
         typename         = std::enable_if_t<(std::is_same_v<First, Rest> && ...)>,
         glm::qualifier P = glm::defaultp>
CUDIFF_HOSTDEVICE auto wrap(First&& first, Rest&&... rest)
{
    constexpr int count = 1 + sizeof...(rest);
    using Q             = dual_value_t<std::decay_t<First>>;
    using T             = glm::vec<count, Q, P>;
    constexpr int N     = dual_component_count<std::decay_t<First>>::num_variables;
    using R             = Dual<N, T>;

    auto result = R(T(first.val(), rest.val()...));

    std::decay_t<First> args[count] = {std::forward<First>(first), std::forward<Rest>(rest)...};

    impl::wrap_impl<N, count, Q, P>::apply(result, args, std::make_index_sequence<N> {});

    return result;
}

namespace impl
{
template<int N, int M, typename Q, glm::qualifier P>
struct unwrap_impl
{
    using InDual  = Dual<N, glm::vec<M, Q, P>>;
    using OutDual = Dual<N, Q>;

    template<std::size_t J, std::size_t I>
    CUDIFF_HOSTDEVICE static void copy_one_derivative(OutDual& out, const InDual& v)
    {
        out.mut_derivative(I) = v.derivative(I)[J];
    }

    template<std::size_t J, std::size_t... Is>
    CUDIFF_HOSTDEVICE static OutDual make_value(const InDual& v, std::index_sequence<Is...>)
    {
        OutDual out(v.val()[J]);

        (copy_one_derivative<J, Is>(out, v), ...);

        return out;
    }

    template<std::size_t... Js>
    CUDIFF_HOSTDEVICE static auto apply(const InDual& v, std::index_sequence<Js...>)
    {
        return CuDiff::make_tuple(make_value<Js>(v, std::make_index_sequence<N> {})...);
    }
};
}    // namespace impl

template<int N, int M, typename Q, glm::qualifier P>
CUDIFF_HOSTDEVICE auto unwrap(const Dual<N, glm::vec<M, Q, P>>& v)
{
    return impl::unwrap_impl<N, M, Q, P>::apply(v, std::make_index_sequence<M> {});
}
}    // namespace CuDiff