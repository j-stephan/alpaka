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

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Sycl.hpp>
#include <alpaka/dev/DevUniformSycl.hpp>
#include <alpaka/vec/Vec.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/mem/buf/Traits.hpp>

#include <CL/sycl.hpp>

#include <memory>
#include <type_traits>

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    class BufCpu;

    //#############################################################################
    //! The SYCL memory buffer.
    template<typename TElem, typename TDim, typename TIdx>
    class BufUniformSycl
    {
        friend struct traits::GetDev<BufUniformSycl<TElem, TDim, TIdx>>;

        template <typename TIdxIntegralConst>
        friend struct extent::traits::GetExtent<TIndexIntegralConst, BufUniformSycl<TElem, TDim, TIdx>>;

        friend struct GetPitchBytes<DimInt<TDim::value - 1u>, BufUniformSycl<TElem, TDim, TIdx>>;

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
        ALPAKA_FN_HOST BufUniformSycl(DevUniformSycl const & dev, TElem* ptr, TIdx const& pitchBytes,
                                      TExtent const& extent)
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

        ALPAKA_FN_HOST ~BufUniformSycl() = default;
        ALPAKA_FN_HOST BufUniformSycl(BufUniformSycl const&) = default;
        ALPAKA_FN_HOST auto operator=(BufUniformSycl const&) -> BufUniformSycl& = default;
        ALPAKA_FN_HOST BufUniformSycl(BufUniformSycl&&) = default;
        ALPAKA_FN_HOST auto operator=(BufUniformSycl&&) -> BufUniformSycl = default;

    protected:
        DevUniformSycl m_dev;
        Vec<TDim, TIdx> m_extentElements;
        std::shared_ptr<TElem> m_ptr;
        TIdx m_pitchBytes; // SYCL does not support pitched allocations. This will simply return the bytes per row.
    };

    namespace traits
    {
        //#############################################################################
        //! The BufUniformSycl device type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct DevType<BufUniformSycl<TElem, TDim, TIdx>>
        {
            using type = DevUniformSycl;
        };

        //#############################################################################
        //! The BufUniformSycl device get trait specialization.
        template<typename TBuf, typename TElem, typename TDim, typename TIdx>
        struct GetDev<std::enable_if_t<std::is_base_of_v<BufUniformSycl<TElem, TDim, TIdx>, TBuf>, TBuf>>
        {
            ALPAKA_FN_HOST static auto getDev(TBuf const& buf)
            {
                return buf.m_dev;
            }
        };

        //#############################################################################
        //! The BufUniformSycl dimension getter trait specialization.
        template<typename TBuf, typename TElem, typename TDim, typename TIdx>
        struct DimType<std::enable_if_t<std::is_base_of_v<BufUniformSycl<TElem, TDim, TIdx>, TBuf>>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The BufUniformSycl memory element type get trait specialization.
        template<typename TBuf, typename TElem, typename TDim, typename TIdx>
        struct ElemType<std::enable_if_t<std::is_base_of<BufUniformSycl<TElem, TDim, TIdx>, TBuf>>>
        {
            using type = TElem;
        };
    }

    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The BufUniformSycl extent get trait specialization.
            template<typename TIdxIntegralConst, typename TBuf, typename TElem, typename TDim, typename TIdx>
            struct GetExtent<TIdxIntegralConst,
                             std::enable_if_t<std::is_base_of_v<BufUniformSycl<TElem, TDim, TIdx>, TBuf>, TBuf>>
            {
                static_assert(TDim::value > TIdxIntegralConst::value, "Requested dimension out of bounds");
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(TBuf const& buf) -> TIdx
                {
                    return buf.m_extentElements[TIdxIntegralConst::value];
                }
            };
        }
    }

    namespace traits
    {
        //#############################################################################
        //! The BufUniformSycl native pointer get trait specialization.
        template<typename TBuf, typename TElem, typename TDim, typename TIdx>
        struct GetPtrNative<std::enable_if_t<std::is_base_of_v<BufUniformSycl<TElem, TDim, TIdx>, TBuf>, TBuf>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrNative(TBuf const& buf) -> TElem const*
            {
                return buf.m_ptr.get();
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrNative(TBuf& buf) -> TElem*
            {
                return buf.m_ptr.get();
            }
        };

        //#############################################################################
        //! The BufUniformSycl pointer on device get trait specialization.
        template<typename TBuf, typename TDev, typename TElem, typename TDim, typename TIdx>
        struct GetPtrDev<std::enable_if_t<std::is_base_of_v<BufUniformSycl<TElem, TDim, TIdx>, TBuf>,
                         std::enable_if_t<std::is_same_v<BufType<TDev, TElem, TDim, TIdx>::type, TBuf>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrDev(TBuf const& buf, TDev const& dev) -> TElem const*
            {
                if(dev != getDev(buf))
                    throw std::runtime_error{"The buffer is not accessible from the given device!"};

                return getPtrNative(buf);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrDev(TBuf& buf, TDev const& dev) -> TElem*
            {
                if(dev != getDev(buf))
                    throw std::runtime_error{"The buffer is not accessible from the given device!"};

                return getPtrNative(buf);
            }
        };

        //#############################################################################
        //! The BufUniformSycl pitch get trait specialization.
        template<typename TBuf, typename TElem, typename TDim, typename TIdx>
        struct GetPitchBytes<DimInt<TDim::value - 1u>,
                             std::enable_if_t<std::is_base_of_v<BufUniformSycl<TElem, TDim, TIdx>, TBuf>, TBuf>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPitchBytes(TBuf const & buf) -> TIdx
            {
                return buf.m_pitchBytes;
            }
        };

        //#############################################################################
        //! The SYCL memory allocation trait specialization.
        template<typename TElem, typename TIdx, typename TDim, typename TDev>
        struct BufAlloc<TElem, TDim, TIdx, std::enable_if_t<std::is_base_of_v<DevUniformSycl, TDev>, TDev>>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocBuf(TDev const & dev, TExtent const & ext)
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

                return typename BufType<TDev, TElem, TDim, TIdx>::type{dev, memPtr, pitchBytes, extent};
            }
        };

        //#############################################################################
        //! The BufUniformSycl SYCL device memory mapping trait specialization.
        template<typename TBuf, typename TDev, typename TElem, typename TDim, typename TIdx>
        struct Map<std::enable_if_t<std::is_base_of_v<BufUniformSycl<TElem, TDim, TIdx>, TBuf>, TBuf>,
                   std::enable_if_t<std::is_same_v<BufType<TDev>::value, TBuf>, TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto map(TBuf const& buf, TDev const& dev) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(getDev(buf) != dev)
                    throw std::runtime_error{"Mapping memory from one SYCL device into another SYCL device not implemented!"};

                // If it is already the same device nothing has to be mapped.
            }
        };

        //#############################################################################
        //! The BufUniformSycl SYCL device memory unmapping trait specialization.
        template<typename TBuf, typename TDev, typename TElem, typename TDim, typename TIdx>
        struct Unmap<std::enable_if_t<std::is_base_of_v<BufUniformSycl<TElem, TDim, TIdx>, TBuf>, TBuf>,
                   std::enable_if_t<std::is_same_v<BufType<TDev>::value, TBuf>, TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto unmap(TBuf const& buf, TDev const& dev) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(getDev(buf) != dev)
                    throw std::runtime_error{"Unmapping memory mapped from one SYCL device into another SYCL device not implemented!"};

                // If it is already the same device nothing has to be unmapped.
            }
        };

        //#############################################################################
        //! The BufUniformSycl memory pinning trait specialization.
        template<typename TBuf, typename TElem, typename TDim, typename TIdx>
        struct Pin<TBuf, std::enable_if_t<std::is_base_of_v<BufUniformSycl<TElem, TDim, TIdx>, TBuf>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto pin(TBuf&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // There is no pinning in SYCL.
            }
        };
        //#############################################################################
        //! The BufUniformSycl memory unpinning trait specialization.
        template<typename TBuf, typename TElem, typename TDim, typename TIdx>
        struct Unpin<TBuf, std::enable_if_t<std::is_base_of_v<BufUniformSycl<TElem, TDim, TIdx>, TBuf>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto unpin(TBuf&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // There is no unpinning in SYCL.
            }
        };

        //#############################################################################
        //! The BufUniformSycl memory pin state trait specialization.
        template<typename TBuf, typename TElem, typename TDim, typename TIdx>
        struct IsPinned<TBUf, std::enable_if_t<std::is_base_of_v<BufUniformSycl<TElem, TDim, TIdx>, TBuf>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto isPinned(TBuf const&) -> bool
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // There is no pinning in SYCL but we will pretend to do
                // so anyway.
                return true;
            }
        };

        //#############################################################################
        //! The BufUniformSycl memory prepareForAsyncCopy trait specialization.
        template<typename TBuf, typename TElem, typename TDim, typename TIdx>
        struct PrepareForAsyncCopy<TBuf, std::enable_if_t<std::is_base_of_v<BufUniformSycl<TElem, TDim, TIdx>, TBuf>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto prepareForAsyncCopy(TBuf&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Everything in SYCL is async by default.
            }
        };

        //#############################################################################
        //! The BufUniformSycl offset get trait specialization.
        template<typename TIdxIntegralConst, typename TBuf, typename TElem, typename TDim, typename TIdx>
        struct GetOffset<TIdxIntegralConst, TBuf,
                         std::enable_if_t<std::is_base_of_v<BufUniformSycl<TElem, TDim, TIdx>, TBuf>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getOffset(TBuf const&) -> TIdx
            {
                return 0u;
            }
        };

        //#############################################################################
        //! The BufUniformSycl idx type trait specialization.
        template<typename TBuf, typename TElem, typename TDim, typename TIdx>
        struct IdxType<TBuf, std::enable_if_t<std::is_base_of_v<BufUniformSycl<TElem, TDim, TIdx>, TBuf>>>
        {
            using type = TIdx;
        };

        //#############################################################################
        //! The BufCpu SYCL device memory mapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct Map<BufCpu<TElem, TDim, TIdx>, TDev, std::enable_if_t<std::is_base_of_v<DevUniformSycl, TDev>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto map(BufCpu<TElem, TDim, TIdx>&, TDev const&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                // No need for this
            }
        };
        //#############################################################################
        //! The BufUniformSycl device memory unmapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct Unmap<BufCpu<TElem, TDim, TIdx>, TDev, std::enable_if_t<std::is_base_of_v<DevUniformSycl, TDev>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto unmap(BufCpu<TElem, TDim, TIdx>&, TDev const&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                // No need for this
            }
        };

        //#############################################################################
        //! The BufCpu pointer on SYCL device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, TDev, std::enable_if_t<std::is_base_of_v<DevUniformSycl, TDev>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx> const&, TDev const&) -> TElem const*
            {
                static_assert(sizeof(TElem) == 0, "SYCL does not map host pointers to devices");
            }

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx>&, TDev const&) -> TElem*
            {
                static_assert(sizeof(TElem) == 0, "SYCL does not map host pointers to devices");
            }
        };
    }
}

#include <alpaka/mem/buf/sycl/Copy.hpp>
#include <alpaka/mem/buf/sycl/Set.hpp>

#endif
