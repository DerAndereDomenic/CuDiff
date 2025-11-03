#pragma once

#include <CuDiff/Platform.h>

namespace CuDiff
{
/**
 * @brief A class to model a Dual number that tracks derivatives for forward differentiation
 *
 * @tparam N Number of independent variables (number of variables we differentiate towards)
 * @tparam T The datatype
 */
template<int N = 1, typename T = float>
class Dual
{
public:
    /**
     * @brief Default constructor
     */
    CUDIFF_HOSTDEVICE Dual() = default;

    /**
     * @brief Constructor
     * @param val The value
     */
    CUDIFF_HOSTDEVICE Dual(T val) : _val(val) {}

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
    CUDIFF_HOSTDEVICE T derivative(size_t i = 0) const { return _derivatives[i]; }

    /**
     * @brief Get the value
     *
     * @return The value
     */
    CUDIFF_HOSTDEVICE T val() const { return _val; }

private:
    T _val            = T(0);
    T _derivatives[N] = {
        T(0),
    };
};
}    // namespace CuDiff