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
#include <alpaka/mem/view/Accessor.hpp>
#include <alpaka/meta/Void.hpp>

namespace alpaka
{
    namespace internal
    {
        template<typename T>
        ALPAKA_FN_HOST_ACC auto asBytePtr(T* p)
        {
            return reinterpret_cast<char*>(p);
        }

        template<typename T>
        struct WriteOnlyProxy
        {
            ALPAKA_FN_HOST_ACC WriteOnlyProxy(T& location) : loc(location)
            {
            }

            template<typename U>
            ALPAKA_FN_HOST_ACC auto& operator=(U&& value)
            {
                loc = std::forward<U>(value);
                return *this;
            }

        private:
            T& loc;
        };

        template<typename TElem, typename TAccessModes>
        struct AccessReturnTypeImpl;

        template<typename TElem>
        struct AccessReturnTypeImpl<TElem, ReadAccess>
        {
            using type = TElem;
        };

        template<typename TElem>
        struct AccessReturnTypeImpl<TElem, WriteAccess>
        {
            using type = WriteOnlyProxy<TElem>;
        };

        template<typename TElem>
        struct AccessReturnTypeImpl<TElem, ReadWriteAccess>
        {
            using type = TElem&;
        };

        template<typename TElem, typename THeadAccessMode, typename... TTailAccessModes>
        struct AccessReturnTypeImpl<TElem, std::tuple<THeadAccessMode, TTailAccessModes...>>
            : AccessReturnTypeImpl<TElem, THeadAccessMode>
        {
        };

        template<typename TElem, typename TAccessModes>
        using AccessReturnType = typename internal::AccessReturnTypeImpl<TElem, TAccessModes>::type;
    } // namespace internal

    //! 1D accessor to memory objects represented by a pointer.
    template<typename TElem, typename TBufferIdx, typename TAccessModes>
    struct Accessor<TElem*, TElem, TBufferIdx, 1, TAccessModes>
    {
        using ReturnType = internal::AccessReturnType<TElem, TAccessModes>;

        ALPAKA_FN_HOST_ACC Accessor(TElem* p_, Vec<DimInt<1>, TBufferIdx> extents_) : p(p_), extents(extents_)
        {
        }

        template<typename TOtherAccessModes>
        ALPAKA_FN_HOST_ACC Accessor(const Accessor<TElem*, TElem, TBufferIdx, 1, TOtherAccessModes>& other)
            : p(other.p)
            , extents(other.extents)
        {
        }

        ALPAKA_FN_HOST_ACC auto operator[](Vec<DimInt<1>, TBufferIdx> i) const -> ReturnType
        {
            return (*this)(i[0]);
        }

        ALPAKA_FN_HOST_ACC auto operator[](TBufferIdx i) const -> ReturnType
        {
            return (*this)(i);
        }

        ALPAKA_FN_HOST_ACC auto operator()(TBufferIdx i) const -> ReturnType
        {
            return p[i];
        }

        TElem* p;
        Vec<DimInt<1>, TBufferIdx> extents;
    };

    //! 2D accessor to memory objects represented by a pointer.
    template<typename TElem, typename TBufferIdx, typename TAccessModes>
    struct Accessor<TElem*, TElem, TBufferIdx, 2, TAccessModes>
    {
        using ReturnType = internal::AccessReturnType<TElem, TAccessModes>;

        ALPAKA_FN_HOST_ACC Accessor(TElem* p_, TBufferIdx rowPitchInBytes_, Vec<DimInt<2>, TBufferIdx> extents_)
            : p(p_)
            , rowPitchInBytes(rowPitchInBytes_)
            , extents(extents_)
        {
        }

        template<typename TOtherAccessModes>
        ALPAKA_FN_HOST_ACC Accessor(const Accessor<TElem*, TElem, TBufferIdx, 2, TOtherAccessModes>& other)
            : p(other.p)
            , rowPitchInBytes(other.rowPitchInBytes)
            , extents(other.extents)
        {
        }

        ALPAKA_FN_HOST_ACC auto operator[](Vec<DimInt<2>, TBufferIdx> i) const -> ReturnType
        {
            return (*this)(i[0], i[1]);
        }

        ALPAKA_FN_HOST_ACC auto operator()(TBufferIdx y, TBufferIdx x) const -> ReturnType
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

        TElem* p;
        TBufferIdx rowPitchInBytes;
        Vec<DimInt<2>, TBufferIdx> extents;
    };

    //! 3D accessor to memory objects represented by a pointer.
    template<typename TElem, typename TBufferIdx, typename TAccessModes>
    struct Accessor<TElem*, TElem, TBufferIdx, 3, TAccessModes>
    {
        using ReturnType = internal::AccessReturnType<TElem, TAccessModes>;

        ALPAKA_FN_HOST_ACC Accessor(
            TElem* p_,
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
        ALPAKA_FN_HOST_ACC Accessor(const Accessor<TElem*, TElem, TBufferIdx, 3, TOtherAccessModes>& other)
            : p(other.p)
            , slicePitchInBytes(other.slicePitchInBytes)
            , rowPitchInBytes(other.rowPitchInBytes)
            , extents(other.extents)
        {
        }

        ALPAKA_FN_HOST_ACC auto operator[](Vec<DimInt<3>, TBufferIdx> i) const -> ReturnType
        {
            return (*this)(i[0], i[1], i[2]);
        }

        ALPAKA_FN_HOST_ACC auto operator()(TBufferIdx z, TBufferIdx y, TBufferIdx x) const -> ReturnType
        {
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wcast-align"
#endif
            return *(
                reinterpret_cast<TElem*>(internal::asBytePtr(p) + z * slicePitchInBytes + y * rowPitchInBytes) + x);
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
        }

        TElem* p;
        TBufferIdx slicePitchInBytes;
        TBufferIdx rowPitchInBytes;
        Vec<DimInt<3>, TBufferIdx> extents;
    };

    namespace traits
    {
        namespace internal
        {
            template<typename T, typename SFINAE = void>
#ifdef __cpp_inline_variables
            inline
#endif
                constexpr bool isView
                = false;

            // TODO: replace this by a concept in C++20
            template<typename TView>
#ifdef __cpp_inline_variables
            inline
#endif
                constexpr bool isView<
                    TView,
                    meta::Void<
                        Idx<TView>,
                        Dim<TView>,
                        decltype(alpaka::getPtrNative(std::declval<TView>())),
                        decltype(getPitchBytes<0>(std::declval<TView>())),
                        decltype(extent::getExtent<0>(std::declval<TView>()))>> = true;

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

            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename... TAccessModes,
                typename TViewForwardRef,
                std::size_t... TPitchIs,
                std::size_t... TExtentIs>
            ALPAKA_FN_HOST_ACC auto buildViewAccessor(
                TViewForwardRef&& view,
                std::index_sequence<TPitchIs...>,
                std::index_sequence<TExtentIs...>)
            {
                using TView = std::decay_t<TViewForwardRef>;
                static_assert(isView<TView>, "");
                using TBufferIdx = Idx<TView>;
                constexpr auto dim = Dim<TView>::value;
                using Elem = Elem<TView>;
                auto p = getPtrNative(view);
                static_assert(
                    std::is_same<decltype(p), const Elem*>::value || std::is_same<decltype(p), Elem*>::value,
                    "We assume that getPtrNative() returns a raw pointer to the view's elements");
                static_assert(
                    !std::is_same<decltype(p), const Elem*>::value
                        || std::is_same<std::tuple<TAccessModes...>, std::tuple<alpaka::ReadAccess>>::value,
                    "When getPtrNative() returns a const raw pointer, the access mode must be ReadAccess");
                using AccessModeList = typename BuildAccessModeList<TAccessModes...>::type;
                return Accessor<Elem*, Elem, TBufferIdx, dim, AccessModeList>{
                    const_cast<Elem*>(p), // strip constness, this is handled the the access modes
                    getPitchBytes<TPitchIs + 1>(view)...,
                    {extent::getExtent<TExtentIs>(view)...}};
            }
        } // namespace internal

        //! Builds an accessor from view like memory objects.
        template<typename TView>
        struct BuildAccessor<TView, std::enable_if_t<internal::isView<TView>>>
        {
            template<typename... TAccessModes, typename TViewForwardRef>
            ALPAKA_FN_HOST_ACC static auto buildAccessor(TViewForwardRef&& view)
            {
                using Dim = Dim<std::decay_t<TView>>;
                return internal::buildViewAccessor<TAccessModes...>(
                    std::forward<TViewForwardRef>(view),
                    std::make_index_sequence<Dim::value - 1>{},
                    std::make_index_sequence<Dim::value>{});
            }
        };
    } // namespace traits
} // namespace alpaka
