/* Copyright 2021 Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/dev/DevGenericSycl.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/mem/view/Accessor.hpp>
#include <alpaka/mem/view/AccessorGenericSycl.hpp>
#include <alpaka/mem/view/ViewAccessor.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/view/ViewSubView.hpp>
#include <alpaka/offset/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

#include <sycl/sycl.hpp>

#include <type_traits>
#include <utility>

namespace alpaka
{
    //! A sub-view to a view.
    template<typename TPltf, typename TElem, typename TDim, typename TIdx>
    class ViewSubView<DevGenericSycl<TPltf>, TElem, TDim, TIdx>
    {
        static_assert(!std::is_const<TIdx>::value, "The idx type of the view can not be const!");

        using Dev = alpaka::DevGenericSycl<TPltf>;

    public:
        //! Constructor.
        //! \param view The view this view is a sub-view of.
        //! \param extentElements The extent in elements.
        //! \param relativeOffsetsElements The offsets in elements.
        template<typename TView, typename TOffsets, typename TExtent>
        ViewSubView(
            TView const& view,
            TExtent const& extentElements,
            TOffsets const& relativeOffsetsElements = TOffsets())
            : m_buf{view.m_buf}
            , m_extentElements(extent::getExtentVec(extentElements))
            , m_offsetsElements(getOffsetVec(relativeOffsetsElements))
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            static_assert(
                std::is_same<Dev, alpaka::Dev<TView>>::value,
                "The dev type of TView and the Dev template parameter have to be identical!");

            static_assert(
                std::is_same<TIdx, Idx<TView>>::value,
                "The idx type of TView and the TIdx template parameter have to be identical!");
            static_assert(
                std::is_same<TIdx, Idx<TExtent>>::value,
                "The idx type of TExtent and the TIdx template parameter have to be identical!");
            static_assert(
                std::is_same<TIdx, Idx<TOffsets>>::value,
                "The idx type of TOffsets and the TIdx template parameter have to be identical!");

            static_assert(
                std::is_same<TDim, Dim<TView>>::value,
                "The dim type of TView and the TDim template parameter have to be identical!");
            static_assert(
                std::is_same<TDim, Dim<TExtent>>::value,
                "The dim type of TExtent and the TDim template parameter have to be identical!");
            static_assert(
                std::is_same<TDim, Dim<TOffsets>>::value,
                "The dim type of TOffsets and the TDim template parameter have to be identical!");

            ALPAKA_ASSERT(((m_offsetsElements + m_extentElements) <= extent::getExtentVec(view))
                              .foldrAll(std::logical_and<bool>()));
        }

        //! Constructor.
        //! \param view The view this view is a sub-view of.
        //! \param extentElements The extent in elements.
        //! \param relativeOffsetsElements The offsets in elements.
        template<typename TView, typename TOffsets, typename TExtent>
        ViewSubView(TView& view, TExtent const& extentElements, TOffsets const& relativeOffsetsElements = TOffsets())
            : m_buf{view.m_buf}
            , m_extentElements(extent::getExtentVec(extentElements))
            , m_offsetsElements(getOffsetVec(relativeOffsetsElements))
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            static_assert(
                std::is_same<Dev, alpaka::Dev<TView>>::value,
                "The dev type of TView and the Dev template parameter have to be identical!");

            static_assert(
                std::is_same<TIdx, Idx<TView>>::value,
                "The idx type of TView and the TIdx template parameter have to be identical!");
            static_assert(
                std::is_same<TIdx, Idx<TExtent>>::value,
                "The idx type of TExtent and the TIdx template parameter have to be identical!");
            static_assert(
                std::is_same<TIdx, Idx<TOffsets>>::value,
                "The idx type of TOffsets and the TIdx template parameter have to be identical!");

            static_assert(
                std::is_same<TDim, Dim<TView>>::value,
                "The dim type of TView and the TDim template parameter have to be identical!");
            static_assert(
                std::is_same<TDim, Dim<TExtent>>::value,
                "The dim type of TExtent and the TDim template parameter have to be identical!");
            static_assert(
                std::is_same<TDim, Dim<TOffsets>>::value,
                "The dim type of TOffsets and the TDim template parameter have to be identical!");

            ALPAKA_ASSERT(((m_offsetsElements + m_extentElements) <= extent::getExtentVec(view))
                              .foldrAll(std::logical_and<bool>()));
        }

        //! \param view The view this view is a sub-view of.
        template<typename TView>
        explicit ViewSubView(TView const& view) : ViewSubView(view, view, Vec<TDim, TIdx>::all(0))
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;
        }

        //! \param view The view this view is a sub-view of.
        template<typename TView>
        explicit ViewSubView(TView& view) : ViewSubView(view, view, Vec<TDim, TIdx>::all(0))
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;
        }

    public:
        sycl::buffer<TElem, TDim::value> m_buf; // Same buffer as parent
        Vec<TDim, TIdx> m_extentElements; // The extent of this view.
        Vec<TDim, TIdx> m_offsetsElements; // The offset relative to the parent view.
    };

    namespace internal
    {
        // temporary until we have proper type traits
        template<typename TPltf, typename TElem, typename TDim, typename TIdx>
        inline constexpr bool isAccessor<ViewSubView<DevGenericSycl<TPltf>, TElem, TDim, TIdx>> = true;
    }

    template<typename... TAccessModes, typename TPltf, typename TElem, typename TDim, typename TIdx>
    auto accessWith(ViewSubView<DevGenericSycl<TPltf>, TElem, TDim, TIdx> const& subView)
    {
        constexpr auto SYCLMode = detail::SYCLMode<TAccessModes...>::value;
        using SYCLAcc = sycl::accessor<TElem, static_cast<int>(TDim::value), SYCLMode, sycl::access::target::global_buffer,
                                       sycl::access::placeholder::true_t>;
        using Modes = typename traits::internal::BuildAccessModeList<TAccessModes...>::type;
        using Acc = Accessor<SYCLAcc, TElem, std::size_t, static_cast<std::size_t>(TDim::value), Modes>;

        auto buf = subView.m_buf; // buffers are reference counted, so we can copy to work around constness

        if constexpr(TDim::value == 1)
        {
            auto const range = sycl::range<1>{static_cast<std::size_t>(extent::getWidth(subView.m_extentElements))};
            auto const offset = sycl::id<1>{static_cast<std::size_t>(extent::getWidth(subView.m_offsetsElements))};
            return Acc{SYCLAcc{buf, range, offset}};
        }
        else if constexpr(TDim::value == 2)
        {
            auto const range = sycl::range<2>{static_cast<std::size_t>(extent::getWidth(subView.m_extentElements)),
                                              static_cast<std::size_t>(extent::getHeight(subView.m_extentElements))};
            auto const offset = sycl::id<2>{static_cast<std::size_t>(extent::getWidth(subView.m_offsetsElements)),
                                            static_cast<std::size_t>(extent::getHeight(subView.m_offsetsElements))};
            
            std::cout << "Creating ranged accessor at (" << offset[0] << ", " << offset[1] << ") with dimensions ("
                      << range[0] << ", " << range[1] << ").\n";

            return Acc{SYCLAcc{buf, range, offset}};
        }
        else
        {
            auto const range = sycl::range<3>{static_cast<std::size_t>(extent::getWidth(subView.m_extentElements)),
                                              static_cast<std::size_t>(extent::getHeight(subView.m_extentElements)),
                                              static_cast<std::size_t>(extent::getDepth(subView.m_extentElements))};
            auto const offset = sycl::id<3>{static_cast<std::size_t>(extent::getWidth(subView.m_offsetsElements)),
                                            static_cast<std::size_t>(extent::getHeight(subView.m_offsetsElements)),
                                            static_cast<std::size_t>(extent::getDepth(subView.m_offsetsElements))};

            std::cout << "Creating ranged accessor at (" << offset[0] << ", " << offset[1] << ", " << offset[2] << ") with dimensions ("
                      << range[0] << ", " << range[1] << ", " << range[2] << ").\n";

            return Acc{SYCLAcc{buf, range, offset}};
        }
    }

} // namespace alpaka

#endif
