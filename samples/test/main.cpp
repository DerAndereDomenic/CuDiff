#include <iostream>
#include <CuDiff/CuDiff.h>

int main()
{
    CuDiff::VarLayout layout;
    CuDiff::Dual<3> x(1.0f, layout.alloc<float>());
    CuDiff::Dual<3> y(-2.0f, layout.alloc<float>());
    CuDiff::Dual<3> z(5.0f, layout.alloc<float>());

    auto f = (3.0f * x + (-2.0 - y) * y) / z - 3.0f;
    f      = -f * f;

    f = CuDiff::sin(f);
    f = CuDiff::exp(f);
    f = CuDiff::cos(f + z);
    f = CuDiff::sqrt(f);
    f = CuDiff::log(f + 42.0f);

    std::cout << "f = " << f.val() << "\n";
    std::cout << "df/dx = " << f.derivative(0) << "\n";
    std::cout << "df/dy = " << f.derivative(1) << "\n";
    std::cout << "df/dz = " << f.derivative(2) << "\n";
}