<h1 align="center">CuDiff - CUDA compatible automatic differentiation framework </h1>

CuDiff is a lightweight, header-only, CUDA-compatible automatic differentiation framework using forward mode differentiation.

## General

While there are a lot of powerful automatic differentiation frameworks (like pytorch), it is sometimes possible to compute derivatives for small functions without much overhead or setup, especially in CUDA kernels/OptiX shaders. This library is primarly designed for such use-cases.

## Installation

This is a header-only library, so just add it to your CMake project by setting the proper include directories or
```cmake
add_subdirectory(CuDiff)
target_link_libraries(${target_name} PRIVATE CuDiff)
```

## Usage

The library uses the templated type `Dual<N,T>` where `N` is the number of independent variables and `T` the underlying datatype.

```c++
Dual<2, float> x = 3.0f;
Dual<2> y = -1.0f; // float default type

x.setDerivative(0, 1.0f); // Mark differentiable -> dx/dx = 1
y.setDerivative(1, 1.0f); // dy/dy = 1
// in setDerivative the first parameter denotes the index of the independent variables, i.e., x stores [dx/dx, dx/dy] = [1,0] and y stores [dy/dx, dy/dy] = [0,1]

auto f = x * x + y;

printf("f = %f\n", f.val()); // 8
printf("df/dx = %f\n", f.derivative(0)); // 2*x = 6
printf("df/dy = %f\n", f.derivative(1)); // 1
```
For larger sets of variables, setting the derivatives like this manually can get tedious and error prone. You can also use a helper class to keep track of indices
```c++
VarLayout layout;
Dual<2, float> x(3.0f, layout.alloc<float>()); // alloc returns 0 and sets derivatives[0] = 1
Dual<2> y(-1.0f, layout.alloc<float>()); // alloc returns 1 and sets derivatives[1] = 1
```
or even simpler, you can use the Variable factory, which can also be used to initialize different types. Here for example, x is a float type and y double:
```c++
auto [x, y] = CuDiff::make_variables<2>(3.0f, -1.0);
```

## Custom Types

It is also possible to integrate custom types into the framework as long as they support the arithmetic operations you want to use (i.e., overloading `operator+`,...). Additionally, it is required to implement Type Traits for this type. Here is an example for `glm::vec2`
```c++
template<>
struct DerivativeTraits<glm::vec2, void>
{
    static CUDIFF_HOSTDEVICE constexpr size_t components() { return 2; } // How many independent variables does this type hold?

    // This function has to return the derivative of the type with respect to the internal independent variables. For a vector, this is a column of the Jacobian matrix. 
    //unit(0) = d[x,y]/dx = [1,0]
    //unit(1) = d[x,y]/dy = [0,1]
    static CUDIFF_HOSTDEVICE T unit(size_t i) 
    {
        return i == 0 ? glm::vec2(1,0) : glm::vec2(0,1); 
    }
};
```
Now, you can define dual variables
```c++
auto v = CuDiff::make_variables<2>(glm::vec2(1.0f, 2.0f)); // Note that although v is a single variable, it holds 2 independent variables. Therefore, the template argument is 2

// The derivative is a glm::vec2 as well
v.derivative(0).x; // dx/dx = 1
v.derivative(0).y; // dy/dx = 0
v.derivative(1).x; // dx/dy = 0
v.derivative(1).y; // dy/dy = 1

```

## License

The software is provided under MIT license. See [LICENSE](LICENSE) for more information.