#pragma once

#include <CuDiff/Platform.h>
#include <CuDiff/Traits.h>

namespace CuDiff
{
struct VarLayout
{
    size_t next = 0;

    template<typename T>
    CUDIFF_HOSTDEVICE size_t alloc()
    {
        size_t start = next;
        next += DerivativeTraits<T>::components();
        return start;
    }
};
}    // namespace CuDiff