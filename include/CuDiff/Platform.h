#pragma once

#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__clang__) && defined(__CUDA__)

    #define CUDIFF_CUDA_AVAILABLE 1

    #define CUDIFF_HOST       __host__
    #define CUDIFF_DEVICE     __device__
    #define CUDIFF_HOSTDEVICE __host__ __device__

#else
    #define CUDIFF_CUDA_AVAILABLE 0

    #define CUDIFF_HOST
    #define CUDIFF_DEVICE
    #define CUDIFF_HOSTDEVICE
#endif

#ifndef CUDIFF_INLINE
    #define CUDIFF_INLINE inline
#endif