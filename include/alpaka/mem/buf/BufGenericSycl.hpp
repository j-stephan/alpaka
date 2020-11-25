/* Copyright 2020 Jan Stephan
 * 
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

#include <alpaka/core/Sycl.hpp>
#include <alpaka/vec/Vec.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/mem/buf/BufCpu.hpp>

#include <CL/sycl.hpp>

#include <memory>

namespace alpaka
{
    //#############################################################################
    //! The SYCL memory buffer.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    class BufGenericSycl
    {
        friend struct traits::GetDev<BufGenericSycl<TElem, TDim, TIdx, TDev>>;

        template <typename TIdxIntegralConst>
        friend struct extent::traits::GetExtent<TIndexIntegralConst, BufGenericSycl<TElem, TDim, TIdx, TDev>>;

        friend struct GetPitchBytes<DimInt<TDim::value - 1u>, BufGenericSycl<TElem, TDim, TIdx, TDev>>;

        static_assert(std::is_const<TElem>::value,
                      "The elem type of the buffer can not be const because the C++ Standard forbids containers of const elements!");

        static_assert(!std::is_const<TIdx>::value, "The idx type of the buffer can not be const!");

    private:
        using Elem = TElem;
        using Dim = TDim;

    public:
        //-----------------------------------------------------------------------------
        //! Constructor
        template<typename TExtent>
        ALPAKA_FN_HOST BufGenericSycl(TDev const & dev, TElem* ptr, TIdx const& pitchBytes, TExtent const& extent)
        : m_dev{dev}
        , m_extentElements{extent::getExtentVecEnd<TDim>(extent)}
        , m_ptr{ptr, [m_dev](auto p) { cl::sycl::free(p, m_dev.m_context); }}
        , m_pitchBytes{pitchBytes}
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            static_assert(TDim::value == Dim<TExtent>::value,
                          "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be identical!");

            static_assert(std::is_same<TIdx, Idx<TExtent>>::value,
                          "The idx type of TExtent and the TIdx template parameter have to be identical!");
        }

        ALPAKA_FN_HOST ~BufGenericSycl() = default;
        ALPAKA_FN_HOST BufGenericSycl(BufGenericSycl const&) = default;
        ALPAKA_FN_HOST auto operator=(BufGenericSycl const&) -> BufGenericSycl& = default;
        ALPAKA_FN_HOST BufGenericSycl(BufGenericSycl&&) = default;
        ALPAKA_FN_HOST auto operator=(BufGenericSycl&&) -> BufGenericSycl = default;

    private:
        TDev m_dev;
        Vec<TDim, TIdx> m_extentElements;
        std::shared_ptr<TElem> m_ptr;
        TIdx m_pitchBytes; // SYCL does not support pitched allocations. This will simply return the bytes per row.
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
        //! The BufGenericSycl native pointer get trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct GetPtrNative<BufGenericSycl<TElem, TDim, TIdx, TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrNative(BufGenericSycl<TElem, TDim, TIdx, TDev> const& buf) -> TElem const*
            {
                return buf.m_ptr.get();
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrNative(BufGenericSycl<TDev, TEleme, TDim, TIdx>& buf) -> TElem*
            {
                return buf.m_ptr.get();
            }
        };

        //#############################################################################
        //! The BufGenericSycl pointer on device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct GetPtrDev<BufGenericSycl<TElem, TDim, TIdx, TDev>, TDev>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrDev(BufGenericSycl<TElem, TDim, TIdx, TDev> const& buf, TDev const& dev) -> TElem const*
            {
                if(dev != getDev(buf))
                    throw std::runtime_error{"The buffer is not accessible from the given device!"};

                return getPtrNative(buf);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrDev(BufGenericSycl<TElem, TDim, TIdx, TDev>& buf, TDev const& dev) -> TElem*
            {
                if(dev != getDev(buf))
                    throw std::runtime_error{"The buffer is not accessible from the given device!"};

                return getPtrNative(buf);
            }
        };

        //#############################################################################
        //! The BufGenericSycl pitch get trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct GetPitchBytes<DimInt<TDim::value - 1u>, BufGenericSycl<TElem, TDim, TIdx, TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPitchBytes(BufGenericSycl<TElem, TDim, TIdx, TDev> const & buf) -> TIdx
            {
                return buf.m_pitchBytes;
            }
        };

        //#############################################################################
        //! The SYCL memory allocation trait specialization.
        template<typename TElem, typename TIdx, typename TDim, typename TPltf>
        struct BufAlloc<TElem, TDim, TIdx, DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocBuf(DevGenericSycl<TPltf> const & dev, TExtent const & ext)
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                using namespace cl::sycl;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__;
                // buffer allocation prints the values, keeping this
                // here for consistency
#endif

                auto memPtr = static_cast<TElem*>(nullptr);
                auto pitchBytes = std::size_t{};
                if constexpr(TDim::value == 1)
                {
                    auto const width = extent::getWidth(ext);
                    auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));

                    memPtr = malloc_device<TElem>(width, dev.m_device, dev.m_context);
                    pitchBytes = static_cast<TIdx>(widthBytes);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << __func__
                              << " ew: " << width
                              << " ewb: " << widthBytes
                              << " ptr: " << memPtr
                              << '\n';
#endif
                }
                else if constexpr(TDim::value == 2)
                {
                    auto const width = extent::getWidth(ext);
                    auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));
                    auto const height = extent::getHeight(ext);

                    memPtr = malloc_device<TElem>(width * height, dev.m_device, dev.m_context);
                    pitchBytes = static_cast<TIdx>(widthBytes);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << __func__
                              << " ew: " << width
                              << " eh: " << height
                              << " ewb: " << widthBytes 
                              << " ptr: " << memPtr
                              << " pitch: " << pitchBytes
                              << '\n';
#endif
                }
                else if constexpr(TDim::value == 3)
                {
                    auto const width = extent::getWidth(ext);
                    auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));
                    auto const height = extent::getHeight(ext);
                    auto const depth = extent::getDepth(ext);

                    memPtr = malloc_device<TElem>(width * height * depth, dev.m_device, dev.m_context);
                    pitchBytes = static_cast<TIdx>(widthBytes);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << __func__
                              << " ew: " << width
                              << " eh: " << height
                              << " ed: " << depth
                              << " ewb: " << widthBytes 
                              << " ptr: " << memPtr
                              << " pitch: " << pitchBytes
                              << '\n';
#endif
                }

                return typename BufType<TElem, TDim, TIdx, DevGenericSycl<TPltf>>::type{dev, memPtr, pitchBytes, extent};
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
            ALPAKA_FN_HOST static auto unmap(BufGenericSycl<TPltf> const& buf, DevGenericSycl<TPltf> const& dev) -> void
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
}

#include <alpaka/mem/buf/sycl/Copy.hpp>
#include <alpaka/mem/buf/sycl/Set.hpp>

#endif
