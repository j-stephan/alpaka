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

#include <tuple>

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

    template<typename TMemoryHandle, typename TElem, typename TBufferIdx, std::size_t TDim, typename TAccessModes>
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

        template<typename TPointer, typename TElem, typename TAccessModes>
        struct AccessReturnTypeImpl;

        template<typename TPointer, typename TElem>
        struct AccessReturnTypeImpl<TPointer, TElem, ReadAccess>
        {
            using type = TElem;
        };

        template<typename TPointer, typename TElem>
        struct AccessReturnTypeImpl<TPointer, TElem, WriteAccess>
        {
            using type = WriteOnlyProxy<TElem>;
        };

        template<typename TPointer, typename TElem>
        struct AccessReturnTypeImpl<TPointer, TElem, ReadWriteAccess>
        {
            using type = TElem&;
        };

        template<typename TPointer, typename TElem, typename THeadAccessMode, typename... TTailAccessModes>
        struct AccessReturnTypeImpl<TPointer, TElem, std::tuple<THeadAccessMode, TTailAccessModes...>>
            : AccessReturnTypeImpl<TPointer, TElem, THeadAccessMode>
        {
        };
    } // namespace internal

    template<typename TPointer, typename TElem, typename TAccessModes>
    using AccessReturnType = typename internal::AccessReturnTypeImpl<TPointer, TElem, TAccessModes>::type;

    template<typename TPointer, typename TElem, typename TBufferIdx, typename TAccessModes>
    struct Accessor<TPointer, TElem, TBufferIdx, 1, TAccessModes>
    {
        using ReturnType = AccessReturnType<TPointer, TElem, TAccessModes>;

        ALPAKA_FN_ACC Accessor(TPointer p_, Vec<DimInt<1>, TBufferIdx> extents_) : p(p_), extents(extents_)
        {
        }

        template<typename TOtherAccessModes>
        ALPAKA_FN_ACC Accessor(const Accessor<TPointer, TElem, TBufferIdx, 1, TOtherAccessModes>& other)
            : p(other.p)
            , extents(other.extents)
        {
        }

        ALPAKA_FN_ACC auto operator[](Vec<DimInt<1>, TBufferIdx> i) const -> ReturnType
        {
            return (*this)(i[0]);
        }

        ALPAKA_FN_ACC auto operator[](TBufferIdx i) const -> ReturnType
        {
            return (*this)(i);
        }

        ALPAKA_FN_ACC auto operator()(TBufferIdx i) const -> ReturnType
        {
            return p[i];
        }

        TPointer p;
        Vec<DimInt<1>, TBufferIdx> extents;
    };

    template<typename TPointer, typename TElem, typename TBufferIdx, typename TAccessModes>
    struct Accessor<TPointer, TElem, TBufferIdx, 2, TAccessModes>
    {
        using ReturnType = AccessReturnType<TPointer, TElem, TAccessModes>;

        ALPAKA_FN_ACC Accessor(TPointer p_, TBufferIdx rowPitchInBytes_, Vec<DimInt<2>, TBufferIdx> extents_)
            : p(p_)
            , rowPitchInBytes(rowPitchInBytes_)
            , extents(extents_)
        {
        }

        template<typename TOtherAccessModes>
        ALPAKA_FN_ACC Accessor(const Accessor<TPointer, TElem, TBufferIdx, 2, TOtherAccessModes>& other)
            : p(other.p)
            , rowPitchInBytes(other.rowPitchInBytes)
            , extents(other.extents)
        {
        }

        ALPAKA_FN_ACC auto operator[](Vec<DimInt<2>, TBufferIdx> i) const -> ReturnType
        {
            return (*this)(i[0], i[1]);
        }

        ALPAKA_FN_ACC auto operator()(TBufferIdx y, TBufferIdx x) const -> ReturnType
        {
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wcast-align"
#endif
            return *(reinterpret_cast<TElem*>(internal::asBytePtr(p) + y * rowPitchInBytes) + x);
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
        }

        TPointer p;
        TBufferIdx rowPitchInBytes;
        Vec<DimInt<2>, TBufferIdx> extents;
    };

    template<typename TPointer, typename TElem, typename TBufferIdx, typename TAccessModes>
    struct Accessor<TPointer, TElem, TBufferIdx, 3, TAccessModes>
    {
        using ReturnType = AccessReturnType<TPointer, TElem, TAccessModes>;

        ALPAKA_FN_ACC Accessor(
            TPointer p_,
            TBufferIdx slicePitchInBytes_,
            TBufferIdx rowPitchInBytes_,
            Vec<DimInt<3>, TBufferIdx> extents_)
            : p(p_)
            , slicePitchInBytes(slicePitchInBytes_)
            , rowPitchInBytes(rowPitchInBytes_)
            , extents(extents_)
        {
        }

        template<typename TOtherAccessModes>
        ALPAKA_FN_ACC Accessor(const Accessor<TPointer, TElem, TBufferIdx, 3, TOtherAccessModes>& other)
            : p(other.p)
            , slicePitchInBytes(other.slicePitchInBytes)
            , rowPitchInBytes(other.rowPitchInBytes)
            , extents(other.extents)
        {
        }

        ALPAKA_FN_ACC auto operator[](Vec<DimInt<3>, TBufferIdx> i) const -> ReturnType
        {
            return (*this)(i[0], i[1], i[2]);
        }

        ALPAKA_FN_ACC auto operator()(TBufferIdx z, TBufferIdx y, TBufferIdx x) const -> ReturnType
        {
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wcast-align"
#endif
            return *(
                reinterpret_cast<TPointer>(internal::asBytePtr(p) + z * slicePitchInBytes + y * rowPitchInBytes) + x);
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
        }

        TPointer p;
        TBufferIdx slicePitchInBytes;
        TBufferIdx rowPitchInBytes;
        Vec<DimInt<3>, TBufferIdx> extents;
    };

#if 0
    using Image = cudaTextureObject_t;

    template<typename TElem, typename TBufferIdx, typename TAccessModes>
    struct Accessor<Image, TElem, TBufferIdx, 1, TAccessModes>
    {
        ALPAKA_FN_ACC auto operator[](Vec<DimInt<1>, TBufferIdx> i) const -> TElem
        {
            return (*this)(i[0]);
        }

        ALPAKA_FN_ACC auto operator[](TBufferIdx i) const -> TElem
        {
            return (*this)(i);
        }

        ALPAKA_FN_ACC auto operator()(TBufferIdx i) const -> TElem
        {
            return tex1Dfetch(texObj, i);
        }

        ALPAKA_FN_ACC auto operator()(float i) const -> TElem
        {
            return tex1D(texObj, i);
        }

        Image texObj;
        Vec<DimInt<1>, TBufferIdx> extents;
    };

    template<typename TElem, typename TBufferIdx, typename TAccessModes>
    struct Accessor<Image, TElem, TBufferIdx, 2, TAccessModes>
    {
        ALPAKA_FN_ACC auto operator[](Vec<DimInt<2>, TBufferIdx> i) const -> TElem
        {
            return (*this)(i[0], i[1]);
        }

        ALPAKA_FN_ACC auto operator()(TBufferIdx y, TBufferIdx x) const -> TElem
        {
            return tex1Dfetch(texObj, y * rowPitchInValues + x);
        }

        ALPAKA_FN_ACC auto operator()(float y, float x) const -> TElem
        {
            return tex2D(texObj, x, y);
        }

        Image texObj;
        TBufferIdx rowPitchInValues;
        Vec<DimInt<2>, TBufferIdx> extents;
    };

    template<typename TElem, typename TBufferIdx, typename TAccessModes>
    struct Accessor<Image, TElem, TBufferIdx, 3, TAccessModes>
    {
        ALPAKA_FN_ACC auto operator[](Vec<DimInt<3>, TBufferIdx> i) const -> TElem
        {
            return (*this)(i[0], i[1], i[2]);
        }

        ALPAKA_FN_ACC auto operator()(TBufferIdx z, TBufferIdx y, TBufferIdx x) const -> TElem
        {
            return tex1Dfetch(texObj, z * slicePitchInValues + y * rowPitchInValues + x);
        }

        ALPAKA_FN_ACC auto operator()(float z, float y, float x) const -> TElem
        {
            return tex3D(texObj, x, y, z);
        }

        Image texObj;
        TBufferIdx rowPitchInValues;
        TBufferIdx slicePitchInValues;
        Vec<DimInt<3>, TBufferIdx> extents;
    };
#endif

    namespace internal
    {
        template<typename... TAccessModes>
        struct BuildAccessModeList;

        template<typename TAccessMode>
        struct BuildAccessModeList<TAccessMode>
        {
            using type = TAccessMode;
        };

        template<typename TAccessMode1, typename TAccessMode2, typename... TAccessModes>
        struct BuildAccessModeList<TAccessMode1, TAccessMode2, TAccessModes...>
        {
            using type = std::tuple<TAccessMode1, TAccessMode2, TAccessModes...>;
        };

        template<typename... TAccessModes, typename TBuf, std::size_t... TPitchIs, std::size_t... TExtentIs>
        auto buildAccessor(TBuf&& buffer, std::index_sequence<TPitchIs...>, std::index_sequence<TExtentIs...>)
        {
            using DBuf = std::decay_t<TBuf>;
            using TBufferIdx = Idx<DBuf>;
            constexpr auto dim = Dim<DBuf>::value;
            auto p = getPtrNative(buffer);
            using AccessModeList = typename BuildAccessModeList<TAccessModes...>::type;
            return Accessor<decltype(p), Elem<DBuf>, TBufferIdx, dim, AccessModeList>{
                p,
                getPitchBytes<TPitchIs + 1>(buffer)...,
                {extent::getExtent<TExtentIs>(buffer)...}};
        }

        template<typename T>
        constexpr bool isAccessor = false;

        template<typename TMemoryHandle, typename TElem, typename TBufferIdx, std::size_t Dim, typename TAccessModes>
        constexpr bool isAccessor<Accessor<TMemoryHandle, TElem, TBufferIdx, Dim, TAccessModes>> = true;
    } // namespace internal

    template<
        typename... TAccessModes,
        typename TBuf,
        typename = std::enable_if_t<!internal::isAccessor<std::decay_t<TBuf>>>>
    auto accessWith(TBuf&& buffer)
    {
        using Dim = Dim<std::decay_t<TBuf>>;
        return internal::buildAccessor<TAccessModes...>(
            std::forward<TBuf>(buffer),
            std::make_index_sequence<Dim::value - 1>{},
            std::make_index_sequence<Dim::value>{});
    }

    // TODO: currently only allows constraining down to 1 access mode
    template<
        typename TNewAccessMode,
        typename TMemoryHandle,
        typename TElem,
        typename TBufferIdx,
        std::size_t TDim,
        typename... TPrevAccessModesBefore,
        typename... TPrevAccessModesAfter>
    auto accessWith(const Accessor<
                    TMemoryHandle,
                    TElem,
                    TBufferIdx,
                    TDim,
                    std::tuple<TPrevAccessModesBefore..., TNewAccessMode, TPrevAccessModesAfter...>>& acc)
    {
        return Accessor<TMemoryHandle, TElem, TBufferIdx, TDim, TNewAccessMode>{acc};
    }

    // constraining accessor to the same access mode again just passes through
    template<typename TAccessMode, typename TMemoryHandle, typename TElem, typename TBufferIdx, std::size_t TDim>
    auto accessWith(const Accessor<TMemoryHandle, TElem, TBufferIdx, TDim, TAccessMode>& acc)
    {
        return acc;
    }

    template<typename TBufOrAcc>
    auto access(TBufOrAcc&& bufOrAcc)
    {
        return accessWith<ReadWriteAccess>(std::forward<TBufOrAcc>(bufOrAcc));
    }

    template<typename TBufOrAcc>
    auto readAccess(TBufOrAcc&& bufOrAcc)
    {
        return accessWith<ReadAccess>(std::forward<TBufOrAcc>(bufOrAcc));
    }

    template<typename TBufOrAcc>
    auto writeAccess(TBufOrAcc&& bufOrAcc)
    {
        return accessWith<WriteAccess>(std::forward<TBufOrAcc>(bufOrAcc));
    }
} // namespace alpaka
