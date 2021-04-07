/* Copyright 2021 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/elem/Traits.hpp>
#include <alpaka/offset/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/meta/IntegerSequence.hpp>
#include <alpaka/meta/Metafunctions.hpp>

#include <sycl/sycl.hpp>

#include <array>
#include <type_traits>
#include <utility>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstddef>

//-----------------------------------------------------------------------------
// SYCL vector types trait specializations.
namespace alpaka
{
    namespace detail
    {
        // Remove std::is_same boilerplate
        template <typename T, typename... Ts>
        struct is_any : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};
    }

    namespace traits
    {
        //##################################################################
        //! In contrast to CUDA SYCL doesn't know 1D vectors. It does
        //! support OpenCL's data types which have additional requirements
        //! on top of those in the C++ standard. Note that SYCL's equivalent
        //! to CUDA's dim3 type is a different class type and thus not used
        //! here.
        template<typename T>
        struct IsSyclBuiltInType :
            alpaka::detail::is_any<T,
                // built-in scalar types - these are the standard C++ built-in types, std::size_t, std::byte and sycl::half
                sycl::half,

                // 2 component vector types
                sycl::char2, sycl::schar2, sycl::uchar2,
                sycl::short2, sycl::ushort2,
                sycl::int2, sycl::uint2,
                sycl::long2, sycl::ulong2,
                sycl::longlong2, sycl::ulonglong2,
                sycl::float2, sycl::double2, sycl::half2,
                sycl::cl_char2, sycl::cl_uchar2,
                sycl::cl_short2, sycl::cl_ushort2,
                sycl::cl_int2, sycl::cl_uint2,
                sycl::cl_long2, sycl::cl_ulong2,
                sycl::cl_float2, sycl::cl_double2, sycl::cl_half2,

                // 3 component vector types
                sycl::char3, sycl::schar3, sycl::uchar3,
                sycl::short3, sycl::ushort3,
                sycl::int3, sycl::uint3,
                sycl::long3, sycl::ulong3,
                sycl::longlong3, sycl::ulonglong3,
                sycl::float3, sycl::double3, sycl::half3,
                sycl::cl_char3, sycl::cl_uchar3,
                sycl::cl_short3, sycl::cl_ushort3,
                sycl::cl_int3, sycl::cl_uint3,
                sycl::cl_long3, sycl::cl_ulong3,
                sycl::cl_float3, sycl::cl_double3, sycl::cl_half3,

                // 4 component vector types
                sycl::char4, sycl::schar4, sycl::uchar4,
                sycl::short4, sycl::ushort4,
                sycl::int4, sycl::uint4,
                sycl::long4, sycl::ulong4,
                sycl::longlong4, sycl::ulonglong4,
                sycl::float4, sycl::double4, sycl::half4,
                sycl::cl_char4, sycl::cl_uchar4,
                sycl::cl_short4, sycl::cl_ushort4,
                sycl::cl_int4, sycl::cl_uint4,
                sycl::cl_long4, sycl::cl_ulong4,
                sycl::cl_float4, sycl::cl_double4, sycl::cl_half4,

                // 8 component vector types
                sycl::char8, sycl::schar8, sycl::uchar8,
                sycl::short8, sycl::ushort8,
                sycl::int8, sycl::uint8,
                sycl::long8, sycl::ulong8,
                sycl::longlong8, sycl::ulonglong8,
                sycl::float8, sycl::double8, sycl::half8,
                sycl::cl_char8, sycl::cl_uchar8,
                sycl::cl_short8, sycl::cl_ushort8,
                sycl::cl_int8, sycl::cl_uint8,
                sycl::cl_long8, sycl::cl_ulong8,
                sycl::cl_float8, sycl::cl_double8, sycl::cl_half8,

                // 16 component vector types
                sycl::char16, sycl::schar16, sycl::uchar16,
                sycl::short16, sycl::ushort16,
                sycl::int16, sycl::uint16,
                sycl::long16, sycl::ulong16,
                sycl::longlong16, sycl::ulonglong16,
                sycl::float16, sycl::double16, sycl::half16,
                sycl::cl_char16, sycl::cl_uchar16,
                sycl::cl_short16, sycl::cl_ushort16,
                sycl::cl_int16, sycl::cl_uint16,
                sycl::cl_long16, sycl::cl_ulong16,
                sycl::cl_float16, sycl::cl_double16, sycl::cl_half16
            >
        {};

        //##################################################################
        //! SYCL's types get trait specialization.
        template<typename T>
        struct DimType<T, std::enable_if_t<IsSyclBuiltInType<T>::value>>
        {
            using type = std::conditional_t<std::is_scalar_v<T>, DimInt<std::size_t{1}>, DimInt<T::size()>>;
        };

        //##################################################################
        //! The SYCL vectors' elem type trait specialization.
        template<typename T>
        struct ElemType<T, std::enable_if_t<IsSyclBuiltInType<T>::value>>
        {
            using type = std::conditional_t<std::is_scalar_v<T>, T, typename T::element_type>;
        };
    }

    namespace extent
    {
        namespace traits
        {
            //##################################################################
            //! The SYCL vectors' extent get trait specialization.
            template<typename TExtent>
            struct GetExtent<DimInt<Dim<TExtent>::value>, TExtent,
                             std::enable_if_t<alpaka::traits::IsSyclBuiltInType<TExtent>::value>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    TExtent const & extent)
                {
                    if constexpr(std::is_scalar_v<TExtent>)
                        return extent;
                    else
                        return extent.template swizzle<DimInt<Dim<TExtent>::value>::value>();
                }
            };

            //#############################################################################
            //! The SYCL vectors' extent set trait specialization.
            template<typename TExtent, typename TExtentVal>
            struct SetExtent<DimInt<Dim<TExtent>::value>, TExtent, TExtentVal,
                             std::enable_if_t<alpaka::traits::IsSyclBuiltInType<TExtent>::value>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setExtent(
                    TExtent const & extent,
                    TExtentVal const & extentVal)
                {
                    if constexpr(std::is_scalar_v<TExtent>)
                        extent = extentVal;
                    else
                        extent.template swizzle<DimInt<Dim<TExtent>::value>::value>() = extentVal;
                }
            };
        }
    }

    namespace traits
    {
        //#############################################################################
        //! The SYCL vectors' offset get trait specialization.
        template<typename TOffsets>
        struct GetOffset<DimInt<Dim<TOffsets>::value>, TOffsets,
                         std::enable_if_t<IsSyclBuiltInType<TOffsets>::value>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getOffset(TOffsets const & offsets)
            {
                if constexpr(std::is_scalar_v<TOffsets>)
                    return offsets;
                else
                    return offsets.template swizzle<DimInt<Dim<TOffsets>::value>::value>();
            }
        };

        //#############################################################################
        //! The SYCL vectors' offset set trait specialization.
        template<typename TOffsets, typename TOffset>
        struct SetOffset<DimInt<Dim<TOffsets>::value>, TOffsets, TOffset,
                         std::enable_if_t<IsSyclBuiltInType<TOffsets>::value>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setOffset(TOffsets const & offsets, TOffset const & offset)
            {
                if constexpr(std::is_scalar_v<TOffsets>)
                    offsets = offset;
                else
                    offsets.template swizzle<DimInt<Dim<TOffsets>::value>::value>() = offset;
            }
        };

        //#############################################################################
        //! The SYCL vectors' idx type trait specialization.
        template<typename TIdx>
        struct IdxType<TIdx, std::enable_if_t<IsSyclBuiltInType<TIdx>::value>>
        {
            using type = std::size_t;
        };
    }
}

#endif