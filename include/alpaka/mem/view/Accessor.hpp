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
    namespace internal
    {
        template<typename T>
        auto asBytePtr(T* p)
        {
            return reinterpret_cast<char*>(p);
        }
        template<typename T>
        auto asBytePtr(const T* p)
        {
            return reinterpret_cast<const char*>(p);
        }
    } // namespace internal

    enum class AccessMode
    {
        read_only,
        read_write
    };

    template<typename Pointer, typename Value, typename Idx, std::size_t Dim>
    struct Accessor;

    template<typename Pointer, typename Value, typename Idx>
    struct Accessor<Pointer, Value, Idx, 1>
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
    struct Accessor<Pointer, Value, Idx, 2>
    {
        ALPAKA_FN_ACC auto& operator[](Vec<DimInt<2>, Idx> i) const
        {
            return (*this)(i[0], i[1]);
        }

        ALPAKA_FN_ACC auto& operator()(Idx y, Idx x) const
        {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
            return *(reinterpret_cast<Value*>(internal::asBytePtr(p) + y * rowPitchInBytes) + x);
#pragma GCC diagnostic pop
        }

        Pointer p;
        Idx rowPitchInBytes;
        Vec<DimInt<2>, Idx> extents;
    };

    template<typename Pointer, typename Value, typename Idx>
    struct Accessor<Pointer, Value, Idx, 3>
    {
        ALPAKA_FN_ACC auto& operator[](Vec<DimInt<3>, Idx> i) const
        {
            return (*this)(i[0], i[1], i[2]);
        }

        ALPAKA_FN_ACC auto& operator()(Idx z, Idx y, Idx x) const
        {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
            return *(
                reinterpret_cast<Pointer>(internal::asBytePtr(p) + z * slicePitchInBytes + y * rowPitchInBytes) + x);
        }

        Pointer p;
        Idx slicePitchInBytes;
        Idx rowPitchInBytes;
        Vec<DimInt<3>, Idx> extents;
    };

#if 0
    using Image = cudaTextureObject_t;

    template<typename Value, typename Idx>
    struct Accessor<Image, Value, Idx, 1>
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
    struct Accessor<Image, Value, Idx, 2>
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
    struct Accessor<Image, Value, Idx, 3>
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
        template<AccessMode Mode, typename Buf, std::size_t... PitchIs, std::size_t... ExtentIs>
        auto buildAccessor(Buf&& buffer, std::index_sequence<PitchIs...>, std::index_sequence<ExtentIs...>)
        {
            using DBuf = std::decay_t<Buf>;
            using Idx = Idx<DBuf>;
            constexpr auto dim = Dim<DBuf>::value;
            constexpr auto IsConst = Mode == AccessMode::read_only;
            using Elem = std::conditional_t<IsConst, const Elem<DBuf>, Elem<DBuf>>;
            auto p = getPtrNative(buffer);
            return Accessor<Elem*, Elem, Idx, dim>{
                p,
                getPitchBytes<PitchIs + 1>(buffer)...,
                {extent::getExtent<ExtentIs>(buffer)...}};
        }
    } // namespace internal

    template<AccessMode Mode = AccessMode::read_write, typename Buf>
    auto access(Buf&& buffer)
    {
        using Dim = Dim<std::decay_t<Buf>>;
        return internal::buildAccessor<Mode>(
            std::forward<Buf>(buffer),
            std::make_index_sequence<Dim::value - 1>{},
            std::make_index_sequence<Dim::value>{});
    }

    template<typename Buf>
    auto readAccess(Buf&& buffer)
    {
        return access<AccessMode::read_only>(std::forward<Buf>(buffer));
    }
} // namespace alpaka
