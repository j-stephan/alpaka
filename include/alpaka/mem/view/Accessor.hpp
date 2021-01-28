/* Copyright 2021 Bernhard Manfred Gruber

 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#pragma once

#include <alpaka/dim/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>

namespace alpaka
{
    template<typename Pointer, typename Value, typename Idx, typename Dim>
    struct Accessor;

    template<typename Pointer, typename Value, typename Idx>
    struct Accessor<Pointer, Value, Idx, DimInt<1>>
    {
        ALPAKA_FN_ACC auto& operator[](Vec<DimInt<1>, Idx> i) const
        {
            return (*this)(i[0]);
        }

        ALPAKA_FN_ACC auto& operator[](Idx i) const
        {
            return (*this)(i);
        }

        ALPAKA_FN_ACC auto& operator()(Idx i) const
        {
            return p[i];
        }

        Pointer p;
        Vec<DimInt<1>, Idx> extents;
    };

    template<typename Pointer, typename Value, typename Idx>
    struct Accessor<Pointer, Value, Idx, DimInt<2>>
    {
        ALPAKA_FN_ACC auto& operator[](Vec<DimInt<2>, Idx> i) const
        {
            return (*this)(i[0], i[1]);
        }

        ALPAKA_FN_ACC auto& operator()(Idx y, Idx x) const
        {
            return *(reinterpret_cast<Pointer>(reinterpret_cast<char*>(p) + y * rowPitchInBytes) + x);
        }

        Pointer p;
        Idx rowPitchInBytes;
        Vec<DimInt<2>, Idx> extents;
    };

    template<typename Pointer, typename Value, typename Idx>
    struct Accessor<Pointer, Value, Idx, DimInt<3>>
    {
        ALPAKA_FN_ACC auto& operator[](Vec<DimInt<3>, Idx> i) const
        {
            return (*this)(i[0], i[1], i[2]);
        }

        ALPAKA_FN_ACC auto& operator()(Idx z, Idx y, Idx x) const
        {
            return *(
                reinterpret_cast<Pointer>(reinterpret_cast<char*>(p) + z * slicePitchInBytes + y * rowPitchInBytes)
                + x);
        }

        Pointer p;
        Idx slicePitchInBytes;
        Idx rowPitchInBytes;
        Vec<DimInt<3>, Idx> extents;
    };

#if 0
    using Image = cudaTextureObject_t;

    template<typename Value, typename Idx>
    struct Accessor<Image, Value, Idx, DimInt<1>>
    {
        ALPAKA_FN_ACC auto operator[](Vec<DimInt<1>, Idx> i) const -> Value
        {
            return (*this)(i[0]);
        }

        ALPAKA_FN_ACC auto operator[](Idx i) const -> Value
        {
            return (*this)(i);
        }

        ALPAKA_FN_ACC auto operator()(Idx i) const -> Value
        {
            return tex1Dfetch(texObj, i);
        }

        ALPAKA_FN_ACC auto operator()(float i) const -> Value
        {
            return tex1D(texObj, i);
        }

        Image texObj;
        Vec<DimInt<1>, Idx> extents;
    };

    template<typename Value, typename Idx>
    struct Accessor<Image, Value, Idx, DimInt<2>>
    {
        ALPAKA_FN_ACC auto operator[](Vec<DimInt<2>, Idx> i) const -> Value
        {
            return (*this)(i[0], i[1]);
        }

        ALPAKA_FN_ACC auto operator()(Idx y, Idx x) const -> Value
        {
            return tex1Dfetch(texObj, y * rowPitchInValues + x);
        }

        ALPAKA_FN_ACC auto operator()(float y, float x) const -> Value
        {
            return tex2D(texObj, x, y);
        }

        Image texObj;
        Idx rowPitchInValues;
        Vec<DimInt<2>, Idx> extents;
    };

    template<typename Value, typename Idx>
    struct Accessor<Image, Value, Idx, DimInt<3>>
    {
        ALPAKA_FN_ACC auto operator[](Vec<DimInt<3>, Idx> i) const -> Value
        {
            return (*this)(i[0], i[1], i[2]);
        }

        ALPAKA_FN_ACC auto operator()(Idx z, Idx y, Idx x) const -> Value
        {
            return tex1Dfetch(texObj, z * slicePitchInValues + y * rowPitchInValues + x);
        }

        ALPAKA_FN_ACC auto operator()(float z, float y, float x) const -> Value
        {
            return tex3D(texObj, x, y, z);
        }

        Image texObj;
        Idx rowPitchInValues;
        Idx slicePitchInValues;
        Vec<DimInt<3>, Idx> extents;
    };
#endif

    namespace internal
    {
        template<typename Buf, std::size_t... PitchIs, std::size_t... ExtentIs>
        auto buildAccessor(Buf&& buffer, std::index_sequence<PitchIs...>, std::index_sequence<ExtentIs...>)
        {
            using Idx = Idx<std::decay_t<Buf>>;
            using Dim = Dim<std::decay_t<Buf>>;
            using Elem = Elem<std::decay_t<Buf>>;
            auto p = getPtrNative(buffer);
            return Accessor<decltype(p), Elem, Idx, Dim>{
                p,
                getPitchBytes<PitchIs + 1>(buffer)...,
                {extent::getExtent<ExtentIs>(buffer)...}};
        }
    } // namespace internal

    template<typename Buf>
    auto access(Buf&& buffer)
    {
        using Dim = Dim<std::decay_t<Buf>>;
        return internal::buildAccessor(
            buffer,
            std::make_index_sequence<Dim::value - 1>{},
            std::make_index_sequence<Dim::value>{});
    }
} // namespace alpaka
