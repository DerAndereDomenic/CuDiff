#pragma once

#include <CuDiff/Platform.h>
#include <CuDiff/Traits.h>

namespace CuDiff
{

template<bool _stochastic, int N, typename T>
struct enable_if_stochastic
{
};

template<int N, typename T>
struct enable_if_stochastic<true, N, T>
{
    /**
     * @brief Set the score with respect to the i-th variable
     *
     * @param i The index of the variable
     * @param score The score value
     */
    CUDIFF_HOSTDEVICE void setScore(size_t i, T score) { _scores[i] = score; }

    /**
     * @brief Get the score of variable i
     *
     * @param i The index of the variable
     * @return The score value
     */
    CUDIFF_HOSTDEVICE const T& score(size_t i = 0) const { return _scores[i]; }

    /**
     * @brief Get the score of variable i as mutable reference
     *
     * @param i The index of the variable
     * @return The score value
     */
    CUDIFF_HOSTDEVICE T& mut_score(size_t i = 0) { return _scores[i]; }

protected:
    T _scores[N] = {
        T(0),
    };
};

/**
 * @brief A class to model a Dual number that tracks derivatives for forward differentiation
 *
 * @tparam N Number of independent variables (number of variables we differentiate towards)
 * @tparam T The datatype
 */
template<int N = 1, typename T = float, bool _stochastic = false>
class Dual : public enable_if_stochastic<_stochastic, N, T>
{
public:
    /**
     * @brief Default constructor
     */
    Dual() = default;

    /**
     * @brief Constructor
     * @param val The value
     */
    CUDIFF_HOSTDEVICE Dual(T val) : _val(val) {}

    /**
     * @brief Create a dual variable that tracks derivatives with respect to variable_index.
     * Note that this assumes that if T is a multi-value type (like vec3) that each component is a independent variable
     * and initializes the corresponding jacobian columns.
     * Example:
     *
     * Dual<4, vec2> v1(vec3(1,2), 0); // uses slots [0,1]
     * v1.derivatives = [(1,0),(0,1),(0,0),(0,0)]
     * Dual<4, vec2> v2(vec3(3,4), 2); // uses slots [2,3]
     * v2.derivatives = [(0,0),(0,0),(1,0),(0,1)]
     *
     * @param val The value
     * @param variable_index The start index of the tracked variable
     */
    CUDIFF_HOSTDEVICE Dual(T val, size_t variable_index) : _val(val)
    {
        using Traits           = DerivativeTraits<T>;
        constexpr size_t comps = Traits::components();
        for(int i = 0; i < comps; ++i)
        {
            _derivatives[variable_index + i] = Traits::unit(i);
        }
    }

    /**
     * @brief Get the stored value
     *
     * @return The value
     */
    CUDIFF_HOSTDEVICE operator T() const { return val(); }

    /**
     * @brief Set the derivative with respect to the i-th variable. I.e. set df/dx_i
     *
     * @param i The index of the variable
     * @param dval The derivative value
     */
    CUDIFF_HOSTDEVICE void setDerivative(size_t i, T dval) { _derivatives[i] = dval; }

    /**
     * @brief Get the derivative of variable i, i.e., df/dx_i
     *
     * @param i The index of the variable
     * @return The derivative value
     */
    CUDIFF_HOSTDEVICE const T& derivative(size_t i = 0) const { return _derivatives[i]; }

    /**
     * @brief Get the derivative of variable i, i.e., df/dx_i as mutable reference
     *
     * @param i The index of the variable
     * @return The derivative value
     */
    CUDIFF_HOSTDEVICE T& mut_derivative(size_t i = 0) { return _derivatives[i]; }

    /**
     * @brief Get the value
     *
     * @return The value
     */
    CUDIFF_HOSTDEVICE const T& val() const { return _val; }

    /**
     * @brief Get the value as mutable reference
     * This may invalidate gradients so this should only be used if gradients are updated manually
     *
     * @return A mutable reference to the value
     */
    CUDIFF_HOSTDEVICE T& mut_val() { return _val; }

    template<bool S = _stochastic, typename = std::enable_if_t<S>>
    CUDIFF_HOSTDEVICE operator Dual<N, T, false>() const
    {
        Dual<N, T, false> out(this->val());

        for(int i = 0; i < N; ++i)
        {
            out.setDerivative(i, this->derivative(i) + this->score(i) * this->val());
        }

        return out;
    }

    template<bool S = !_stochastic, typename = std::enable_if_t<S>>
    CUDIFF_HOSTDEVICE operator Dual<N, T, true>() const
    {
        Dual<N, T, true> out(this->val());

        for(int i = 0; i < N; ++i)
        {
            out.setDerivative(i, this->derivative(i));
            out.setScore(i, T(0));
        }

        return out;
    }

private:
    T _val            = T(0);
    T _derivatives[N] = {
        T(0),
    };
};

template<typename>
struct is_dual : std::false_type
{
};

template<int N, typename T>
struct is_dual<Dual<N, T>> : std::true_type
{
};

template<typename T>
inline constexpr bool is_dual_v = is_dual<T>::value;

template<typename T>
struct dual_value_type
{
    using type = T;
};

template<int N, typename T>
struct dual_value_type<Dual<N, T>>
{
    using type = T;
};

template<typename T>
using dual_value_t = typename dual_value_type<T>::type;
}    // namespace CuDiff