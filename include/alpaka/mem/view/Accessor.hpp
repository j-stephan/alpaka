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

    struct ReadAccess
    {
    };
    struct WriteAccess
    {
    };
    struct ReadWriteAccess
    {
    };

    template<typename MemoryHandle, typename Elem, typename BufferIdx, std::size_t Dim, typename AccessModes>
    struct Accessor;

    namespace internal
    {
        template<typename T>
        struct WriteOnlyProxy
        {
            WriteOnlyProxy(T& location) : loc(location)
            {
            }

            template<typename U>
            auto& operator=(U&& value)
            {
                loc = std::forward<U>(value);
                return *this;
            }

        private:
            T& loc;
        };

        template<typename Pointer, typename Elem, typename AccessModes>
        struct AccessReturnTypeImpl;

        template<typename Pointer, typename Elem>
        struct AccessReturnTypeImpl<Pointer, Elem, ReadAccess>
        {
            using type = Elem;
        };

        template<typename Pointer, typename Elem>
        struct AccessReturnTypeImpl<Pointer, Elem, WriteAccess>
        {
            using type = WriteOnlyProxy<Elem>;
        };

        template<typename Pointer, typename Elem>
        struct AccessReturnTypeImpl<Pointer, Elem, ReadWriteAccess>
        {
            using type = Elem&;
        };

        template<typename Pointer, typename Elem, typename HeadAccessMode, typename... TailAccessModes>
        struct AccessReturnTypeImpl<Pointer, Elem, std::tuple<HeadAccessMode, TailAccessModes...>>
            : AccessReturnTypeImpl<Pointer, Elem, HeadAccessMode>
        {
        };
    } // namespace internal

    template<typename Pointer, typename Elem, typename AccessModes>
    using AccessReturnType = typename internal::AccessReturnTypeImpl<Pointer, Elem, AccessModes>::type;

    template<typename Pointer, typename Elem, typename BufferIdx, typename AccessModes>
    struct Accessor<Pointer, Elem, BufferIdx, 1, AccessModes>
    {
        using ReturnType = AccessReturnType<Pointer, Elem, AccessModes>;

        ALPAKA_FN_ACC auto operator[](Vec<DimInt<1>, BufferIdx> i) const -> ReturnType
        {
            return (*this)(i[0]);
        }

        ALPAKA_FN_ACC auto operator[](BufferIdx i) const -> ReturnType
        {
            return (*this)(i);
        }

        ALPAKA_FN_ACC auto operator()(BufferIdx i) const -> ReturnType
        {
            return p[i];
        }

        Pointer p;
        Vec<DimInt<1>, BufferIdx> extents;
    };

    template<typename Pointer, typename Elem, typename BufferIdx, typename AccessModes>
    struct Accessor<Pointer, Elem, BufferIdx, 2, AccessModes>
    {
        using ReturnType = AccessReturnType<Pointer, Elem, AccessModes>;

        ALPAKA_FN_ACC auto operator[](Vec<DimInt<2>, BufferIdx> i) const -> ReturnType
        {
            return (*this)(i[0], i[1]);
        }

        ALPAKA_FN_ACC auto operator()(BufferIdx y, BufferIdx x) const -> ReturnType
        {
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wcast-align"
#endif
            return *(reinterpret_cast<Elem*>(internal::asBytePtr(p) + y * rowPitchInBytes) + x);
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
        }

        Pointer p;
        BufferIdx rowPitchInBytes;
        Vec<DimInt<2>, BufferIdx> extents;
    };

    template<typename Pointer, typename Elem, typename BufferIdx, typename AccessModes>
    struct Accessor<Pointer, Elem, BufferIdx, 3, AccessModes>
    {
        using ReturnType = AccessReturnType<Pointer, Elem, AccessModes>;

        ALPAKA_FN_ACC auto operator[](Vec<DimInt<3>, BufferIdx> i) const -> ReturnType
        {
            return (*this)(i[0], i[1], i[2]);
        }

        ALPAKA_FN_ACC auto operator()(BufferIdx z, BufferIdx y, BufferIdx x) const -> ReturnType
        {
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wcast-align"
#endif
            return *(
                reinterpret_cast<Pointer>(internal::asBytePtr(p) + z * slicePitchInBytes + y * rowPitchInBytes) + x);
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
        }

        Pointer p;
        BufferIdx slicePitchInBytes;
        BufferIdx rowPitchInBytes;
        Vec<DimInt<3>, BufferIdx> extents;
    };

#if 0
    using Image = cudaTextureObject_t;

    template<typename Elem, typename BufferIdx, typename AccessModes>
    struct Accessor<Image, Elem, BufferIdx, 1, AccessModes>
    {
        ALPAKA_FN_ACC auto operator[](Vec<DimInt<1>, BufferIdx> i) const -> Elem
        {
            return (*this)(i[0]);
        }

        ALPAKA_FN_ACC auto operator[](BufferIdx i) const -> Elem
        {
            return (*this)(i);
        }

        ALPAKA_FN_ACC auto operator()(BufferIdx i) const -> Elem
        {
            return tex1Dfetch(texObj, i);
        }

        ALPAKA_FN_ACC auto operator()(float i) const -> Elem
        {
            return tex1D(texObj, i);
        }

        Image texObj;
        Vec<DimInt<1>, BufferIdx> extents;
    };

    template<typename Elem, typename BufferIdx, typename AccessModes>
    struct Accessor<Image, Elem, BufferIdx, 2, AccessModes>
    {
        ALPAKA_FN_ACC auto operator[](Vec<DimInt<2>, BufferIdx> i) const -> Elem
        {
            return (*this)(i[0], i[1]);
        }

        ALPAKA_FN_ACC auto operator()(BufferIdx y, BufferIdx x) const -> Elem
        {
            return tex1Dfetch(texObj, y * rowPitchInValues + x);
        }

        ALPAKA_FN_ACC auto operator()(float y, float x) const -> Elem
        {
            return tex2D(texObj, x, y);
        }

        Image texObj;
        BufferIdx rowPitchInValues;
        Vec<DimInt<2>, BufferIdx> extents;
    };

    template<typename Elem, typename BufferIdx, typename AccessModes>
    struct Accessor<Image, Elem, BufferIdx, 3, AccessModes>
    {
        ALPAKA_FN_ACC auto operator[](Vec<DimInt<3>, BufferIdx> i) const -> Elem
        {
            return (*this)(i[0], i[1], i[2]);
        }

        ALPAKA_FN_ACC auto operator()(BufferIdx z, BufferIdx y, BufferIdx x) const -> Elem
        {
            return tex1Dfetch(texObj, z * slicePitchInValues + y * rowPitchInValues + x);
        }

        ALPAKA_FN_ACC auto operator()(float z, float y, float x) const -> Elem
        {
            return tex3D(texObj, x, y, z);
        }

        Image texObj;
        BufferIdx rowPitchInValues;
        BufferIdx slicePitchInValues;
        Vec<DimInt<3>, BufferIdx> extents;
    };
#endif

    namespace internal
    {
        template<typename... AccessModes>
        struct BuildAccessModeList;

        template<typename AccessMode>
        struct BuildAccessModeList<AccessMode>
        {
            using type = AccessMode;
        };

        template<typename AccessMode1, typename AccessMode2, typename... AccessModes>
        struct BuildAccessModeList<AccessMode1, AccessMode2, AccessModes...>
        {
            using type = std::tuple<AccessMode1, AccessMode2, AccessModes...>;
        };

        template<typename... AccessModes, typename Buf, std::size_t... PitchIs, std::size_t... ExtentIs>
        auto buildAccessor(Buf&& buffer, std::index_sequence<PitchIs...>, std::index_sequence<ExtentIs...>)
        {
            using DBuf = std::decay_t<Buf>;
            using BufferIdx = Idx<DBuf>;
            constexpr auto dim = Dim<DBuf>::value;
            auto p = getPtrNative(buffer);
            using AccessModeList = typename BuildAccessModeList<AccessModes...>::type;
            return Accessor<decltype(p), Elem<DBuf>, BufferIdx, dim, AccessModeList>{
                p,
                getPitchBytes<PitchIs + 1>(buffer)...,
                {extent::getExtent<ExtentIs>(buffer)...}};
        }
    } // namespace internal

    template<typename... AccessModes, typename Buf>
    auto accessWith(Buf&& buffer)
    {
        using Dim = Dim<std::decay_t<Buf>>;
        return internal::buildAccessor<AccessModes...>(
            std::forward<Buf>(buffer),
            std::make_index_sequence<Dim::value - 1>{},
            std::make_index_sequence<Dim::value>{});
    }

    template<typename Buf>
    auto access(Buf&& buffer)
    {
        return accessWith<WriteAccess, ReadAccess>(std::forward<Buf>(buffer));
    }

    template<typename Buf>
    auto readAccess(Buf&& buffer)
    {
        return accessWith<ReadAccess>(std::forward<Buf>(buffer));
    }

    template<typename Buf>
    auto writeAccess(Buf&& buffer)
    {
        return accessWith<WriteAccess>(std::forward<Buf>(buffer));
    }
} // namespace alpaka
