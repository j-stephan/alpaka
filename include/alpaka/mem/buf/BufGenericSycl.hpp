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

#include <alpaka/core/Sycl.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dev/DevGenericSycl.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/mem/buf/BufCpu.hpp>
#include <alpaka/mem/view/Accessor.hpp>
#include <alpaka/mem/view/AccessorGenericSycl.hpp>
#include <alpaka/mem/view/ViewAccessor.hpp>
#include <alpaka/vec/Vec.hpp>

#include <sycl/sycl.hpp>

#include <memory>
#include <type_traits>

namespace alpaka
{
    //! The SYCL memory buffer.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    struct BufGenericSycl
    {
        static_assert(!std::is_const_v<TElem>,
                      "The elem type of the buffer can not be const because the C++ Standard forbids containers of const elements!");

        static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer can not be const!");

        //! Constructor
        template<typename TExtent>
        ALPAKA_FN_HOST BufGenericSycl(TDev const & dev, sycl::buffer<TElem, TDim::value> buf, TExtent const& extent)
        : m_dev{dev}, m_extentElements{extent::getExtentVecEnd<TDim>(extent)}, m_buf{buf}
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            static_assert(TDim::value == Dim<TExtent>::value,
                          "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be identical!");

            static_assert(std::is_same_v<TIdx, Idx<TExtent>>,
                          "The idx type of TExtent and the TIdx template parameter have to be identical!");
        }

        ALPAKA_FN_HOST ~BufGenericSycl() = default;
        ALPAKA_FN_HOST BufGenericSycl(BufGenericSycl const&) = default;
        ALPAKA_FN_HOST auto operator=(BufGenericSycl const&) -> BufGenericSycl& = default;
        ALPAKA_FN_HOST BufGenericSycl(BufGenericSycl&&) = default;
        ALPAKA_FN_HOST auto operator=(BufGenericSycl&&) -> BufGenericSycl& = default;

        TDev m_dev;
        Vec<TDim, TIdx> m_extentElements;
        sycl::buffer<TElem, TDim::value> m_buf;
    };

    namespace traits
    {
        //#############################################################################
        //! The BufGenericSycl device type trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct DevType<BufGenericSycl<TElem, TDim, TIdx, TDev>>
        {
            using type = TDev;
        };

        //#############################################################################
        //! The BufGenericSycl device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct GetDev<BufGenericSycl<TElem, TDim, TIdx, TDev>>
        {
            ALPAKA_FN_HOST static auto getDev(BufGenericSycl<TElem, TDim, TIdx, TDev> const& buf)
            {
                return buf.m_dev;
            }
        };

        //#############################################################################
        //! The BufGenericSycl dimension getter trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct DimType<BufGenericSycl<TElem, TDim, TIdx, TDev>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The BufGenericSycl memory element type get trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct ElemType<BufGenericSycl<TElem, TDim, TIdx, TDev>>
        {
            using type = TElem;
        };
    }

    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The BufGenericSycl extent get trait specialization.
            template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx, typename TDev>
            struct GetExtent<TIdxIntegralConst, BufGenericSycl<TElem, TDim, TIdx, TDev>>
            {
                static_assert(TDim::value > TIdxIntegralConst::value, "Requested dimension out of bounds");
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(BufGenericSycl<TElem, TDim, TIdx, TDev> const& buf) -> TIdx
                {
                    return buf.m_extentElements[TIdxIntegralConst::value];
                }
            };
        }
    }

    namespace traits
    {
        //#############################################################################
        //! The BufGenericSycl pitch get trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct GetPitchBytes<DimInt<TDim::value - 1u>, BufGenericSycl<TElem, TDim, TIdx, TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPitchBytes(BufGenericSycl<TElem, TDim, TIdx, TDev> const & buf) -> TIdx
            {
                return static_cast<TIdx>(sizeof(TElem)) * static_cast<TIdx>(extent::getWidth(buf.m_extentElements));
            }
        };

        //#############################################################################
        //! The SYCL memory allocation trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TPltf>
        struct BufAlloc<TElem, TDim, TIdx, DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocBuf(DevGenericSycl<TPltf> const & dev, TExtent const & ext)
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                using namespace sycl;

                using buffer = BufGenericSycl<TElem, TDim, TIdx, DevGenericSycl<TPltf>>;
                using handle = sycl::buffer<TElem, TDim::value>;

                // SYCL follows the same index ordering as alpaka. Do not change the index ordering here!
                if constexpr(TDim::value == 1)
                {
                    auto const range = sycl::range<1>{static_cast<std::size_t>(ext[0])};
                    return buffer{dev, handle{range}, ext};
                }
                else if constexpr(TDim::value == 2)
                {
                    auto const range = sycl::range<2>{static_cast<std::size_t>(ext[0]),
                                                      static_cast<std::size_t>(ext[1])};
                    return buffer{dev, handle{range}, ext};
                }
                else
                {
                    auto const range = sycl::range<3>{static_cast<std::size_t>(ext[0]),
                                                      static_cast<std::size_t>(ext[1]),
                                                      static_cast<std::size_t>(ext[2])};
                    return buffer{dev, handle{range}, ext};
                }
            }
        };

        //#############################################################################
        //! The BufGenericSycl SYCL device memory mapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TPltf>
        struct Map<BufGenericSycl<TElem, TDim, TIdx, DevGenericSycl<TPltf>>, DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto map(BufGenericSycl<TElem, TDim, TIdx, DevGenericSycl<TPltf>> const& buf, DevGenericSycl<TPltf> const& dev) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(getDev(buf) != dev)
                    throw std::runtime_error{"Mapping memory from one SYCL device into another SYCL device not implemented!"};

                // If it is already the same device nothing has to be mapped.
            }
        };

        //#############################################################################
        //! The BufGenericSycl SYCL device memory unmapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TPltf>
        struct Unmap<BufGenericSycl<TElem, TDim, TIdx, DevGenericSycl<TPltf>>, DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto unmap(BufGenericSycl<TElem, TDim, TIdx, DevGenericSycl<TPltf>> const& buf, DevGenericSycl<TPltf> const& dev) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(getDev(buf) != dev)
                    throw std::runtime_error{"Unmapping memory mapped from one SYCL device into another SYCL device not implemented!"};

                // If it is already the same device nothing has to be unmapped.
            }
        };

        //#############################################################################
        //! The BufGenericSycl memory pinning trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct Pin<BufGenericSycl<TElem, TDim, TIdx, TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto pin(BufGenericSycl<TElem, TDim, TIdx, TDev>&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // There is no pinning in SYCL.
            }
        };
        //#############################################################################
        //! The BufGenericSycl memory unpinning trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct Unpin<BufGenericSycl<TElem, TDim, TIdx, TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto unpin(BufGenericSycl<TElem, TDim, TIdx, TDev>&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // There is no unpinning in SYCL.
            }
        };

        //#############################################################################
        //! The BufGenericSycl memory pin state trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct IsPinned<BufGenericSycl<TElem, TDim, TIdx, TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto isPinned(BufGenericSycl<TElem, TDim, TIdx, TDev> const&) -> bool
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // There is no pinning in SYCL but we will pretend to do
                // so anyway.
                return true;
            }
        };

        //#############################################################################
        //! The BufGenericSycl memory prepareForAsyncCopy trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct PrepareForAsyncCopy<BufGenericSycl<TElem, TDim, TIdx, TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto prepareForAsyncCopy(BufGenericSycl<TElem, TDim, TIdx, TDev>&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Everything in SYCL is async by default.
            }
        };

        //#############################################################################
        //! The BufGenericSycl offset get trait specialization.
        template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx, typename TDev>
        struct GetOffset<TIdxIntegralConst, BufGenericSycl<TElem, TDim, TIdx, TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getOffset(BufGenericSycl<TElem, TDim, TIdx, TDev> const&) -> TIdx
            {
                return 0u;
            }
        };

        //#############################################################################
        //! The BufGenericSycl idx type trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct IdxType<BufGenericSycl<TElem, TDim, TIdx, TDev>>
        {
            using type = TIdx;
        };

        //#############################################################################
        //! The BufCpu SYCL device memory mapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TPltf>
        struct Map<BufCpu<TElem, TDim, TIdx>, DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto map(BufCpu<TElem, TDim, TIdx>&, DevGenericSycl<TPltf> const&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                // No need for this
            }
        };
        //#############################################################################
        //! The BufGenericSycl device memory unmapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TPltf>
        struct Unmap<BufCpu<TElem, TDim, TIdx>, DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto unmap(BufCpu<TElem, TDim, TIdx>&, DevGenericSycl<TPltf> const&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                // No need for this
            }
        };

        //#############################################################################
        //! The BufCpu pointer on SYCL device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TPltf>
        struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx> const&, DevGenericSycl<TPltf> const&) -> TElem const*
            {
                static_assert(sizeof(TElem) == 0, "SYCL does not map host pointers to devices");
            }

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx>&, DevGenericSycl<TPltf> const&) -> TElem*
            {
                static_assert(sizeof(TElem) == 0, "SYCL does not map host pointers to devices");
            }
        };

    }


    namespace traits
    {
        namespace internal
        {
            template<typename TElem, typename TDim, typename TIdx, typename TDev>
            inline constexpr bool isView<BufGenericSycl<TElem, TDim, TIdx, TDev>> = false;
        }

        //! The customization point for how to build an accessor for a given memory object.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct BuildAccessor<BufGenericSycl<TElem, TDim, TIdx, TDev>>
        {
            template<typename... TAccessModes>
            ALPAKA_FN_HOST_ACC static auto buildAccessor(BufGenericSycl<TElem, TDim, TIdx, TDev> const& buffer)
            {
                constexpr auto SYCLMode = alpaka::detail::SYCLMode<TAccessModes...>::value;
                using SYCLAcc = sycl::accessor<TElem, static_cast<int>(TDim::value), SYCLMode, sycl::access::target::global_buffer,
                                           sycl::access::placeholder::true_t>;
                using Modes = typename traits::internal::BuildAccessModeList<TAccessModes...>::type;
                using Acc = Accessor<SYCLAcc, TElem, TIdx, static_cast<std::size_t>(TDim::value), Modes>;

                auto buf = buffer.m_buf; // buffers are reference counted, so we can copy to work around constness

                return Acc{SYCLAcc{buf}, buffer.m_extentElements};
            }
        };
    }
}

#include <alpaka/mem/buf/sycl/Copy.hpp>
#include <alpaka/mem/buf/sycl/Set.hpp>

#endif
