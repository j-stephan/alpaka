/* Copyright 2019 Jan Stephan
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
#include <alpaka/dev/DevSycl.hpp>
#include <alpaka/vec/Vec.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/mem/buf/Traits.hpp>

#include <memory>

namespace alpaka
{
    namespace dev
    {
        class DevSycl;
    }

    namespace mem
    {
        namespace buf
        {
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            class BufCpu;
        }
    }

    namespace mem
    {
        namespace buf
        {
            //#############################################################################
            //! The SYCL memory buffer.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            class BufSycl
            {
                static_assert(
                    !std::is_const<TElem>::value,
                    "The elem type of the buffer can not be const because the C++ Standard forbids containers of const elements!");
                static_assert(
                    !std::is_const<TIdx>::value,
                    "The idx type of the buffer can not be const!");
            private:
                using Elem = TElem;
                using Dim = TDim;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor
                template<
                    typename TExtent>
                ALPAKA_FN_HOST BufSycl(
                    dev::DevSycl const & dev,
                    cl::sycl::buffer<TElem, TDim::value>&& pMem,
                    TIdx const & pitchBytes,
                    TExtent const & extent) :
                        m_dev(dev),
                        m_extentElements(extent::getExtentVecEnd<TDim>(extent)),
                        m_spMem(std::move(pMem)),
                        m_pitchBytes(pitchBytes)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        TDim::value == dim::Dim<TExtent>::value,
                        "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be identical!");
                    static_assert(
                        std::is_same<TIdx, idx::Idx<TExtent>>::value,
                        "The idx type of TExtent and the TIdx template parameter have to be identical!");
                }

            public:
                dev::DevSycl m_dev;
                vec::Vec<TDim, TIdx> m_extentElements;
                cl::sycl::buffer<TElem, TDim::value> m_spMem;
                TIdx m_pitchBytes;
            };
        }
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The BufSycl device type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct DevType<
                mem::buf::BufSycl<TElem, TDim, TIdx>>
            {
                using type = dev::DevSycl;
            };

            //#############################################################################
            //! The BufSycl device get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetDev<
                mem::buf::BufSycl<TElem, TDim, TIdx>>
            {
                ALPAKA_FN_HOST static auto getDev(
                    mem::buf::BufSycl<TElem, TDim, TIdx> const & buf)
                -> dev::DevSycl
                {
                    return buf.m_dev;
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The BufSycl dimension getter trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct DimType<
                mem::buf::BufSycl<TElem, TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace elem
    {
        namespace traits
        {
            //#############################################################################
            //! The BufSycl memory element type get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct ElemType<
                mem::buf::BufSycl<TElem, TDim, TIdx>>
            {
                using type = TElem;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The BufSycl extent get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetExtent<
                TIdxIntegralConst,
                mem::buf::BufSycl<TElem, TDim, TIdx>,
                typename std::enable_if<(TDim::value > TIdxIntegralConst::value)>::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(
                    mem::buf::BufSycl<TElem, TDim, TIdx> const & extent)
                -> TIdx
                {
                    return extent.m_extentElements[TIdxIntegralConst::value];
                }
            };
        }
    }
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The BufSycl native pointer get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPtrNative<
                    mem::buf::BufSycl<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    // FIXME: Find a workaround
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufSycl<TElem, TDim, TIdx> const & buf)
                    -> TElem const *
                    {
                        static_assert(1 == 0, "SYCL does not expose device pointers to host code");
                        throw std::runtime_error{"SYCL does not expose device pointers to host code"};
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufSycl<TElem, TDim, TIdx> & buf)
                    -> TElem *
                    {
                        static_assert(1 == 0, "SYCL does not expose device pointers to host code");
                        throw std::runtime_error{"SYCL does not expose device pointers to host code"};
                    }
                };

                //#############################################################################
                //! The BufSycl pointer on device get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPtrDev<
                    mem::buf::BufSycl<TElem, TDim, TIdx>,
                    dev::DevSycl>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufSycl<TElem, TDim, TIdx> const & buf,
                        dev::Sycl const & dev)
                    -> TElem const *
                    {
                        static_assert(1 == 0, "SYCL does not expose device pointers to host code");
                        throw std::runtime_error{"SYCL does not expose device pointers to host code"};
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufSycl<TElem, TDim, TIdx> & buf,
                        dev::DevSycl const & dev)
                    -> TElem *
                    {
                        static_assert(1 == 0, "SYCL does not expose device pointers to host code");
                        throw std::runtime_error{"SYCL does not expose device pointers to host code"};
                    }
                };
                //#############################################################################
                //! The BufSycl pitch get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPitchBytes<
                    dim::DimInt<TDim::value - 1u>,
                    mem::buf::BufSycl<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        mem::buf::BufSycl<TElem, TDim, TIdx> const & buf)
                    -> TIdx
                    {
                        return buf.m_pitchBytes;
                    }
                };
            }
        }
        namespace buf
        {
            namespace traits
            {
                //#############################################################################
                //! The SYCL memory allocation trait specialization.
                template<
                    typename TElem,
                    typename TIdx,
                    typename TDim>
                struct Alloc<
                    TElem,
                    TDim,
                    TIdx,
                    dev::DevSycl>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevSycl const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufSycl<TElem, TDim, TIdx>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto const width(extent::getWidth(extent));
                        auto const widthBytes(width * static_cast<TIdx>(sizeof(TElem)));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << __func__
                            << " ew: " << width
                            << " ewb: " << widthBytes
                            // << " ptr: " << memPtr // not supported
                            << std::endl;
#endif
                        return mem::buf::BufSycl<TElem, dim::DimInt<1u>, TIdx> {
                            dev,
                            cl::sycl::buffer<TElem, TDim::value>{cl::sycl::range<TDim::value>{width}},
                            static_cast<TIdx>(widthBytes),
                            extent};
                    }
                };

                //#############################################################################
                //! The BufSycl SYCL device memory mapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Map<
                    mem::buf::BufSycl<TElem, TDim, TIdx>,
                    dev::DevSycl>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto map(
                        mem::buf::BufSycl<TElem, TDim, TIdx> const & buf,
                        dev::DevSycl const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // SYCL maps all buffers to all SYCL devices in the
                        // same context. Nothing to do here.
                    }
                };

                //#############################################################################
                //! The BufSycl SYCL device memory unmapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Unmap<
                    mem::buf::BufSycl<TElem, TDim, TIdx>,
                    dev::DevSycl>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unmap(
                        mem::buf::BufSycl<TElem, TDim, TIdx> const & buf,
                        dev::DevSycl const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // There is no unmapping in SYCL.
                    }
                };
                //#############################################################################
                //! The BufSycl memory pinning trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Pin<
                    mem::buf::BufSycl<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto pin(
                        mem::buf::BufSycl<TElem, TDim, TIdx> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // There is no pinning in SYCL.
                    }
                };
                //#############################################################################
                //! The BufSycl memory unpinning trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Unpin<
                    mem::buf::BufSycl<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unpin(
                        mem::buf::BufSycl<TElem, TDim, TIdx> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // There is no unpinning in SYCL.
                    }
                };

                //#############################################################################
                //! The BufSycl memory pin state trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct IsPinned<
                    mem::buf::BufSycl<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto isPinned(
                        mem::buf::BufSycl<TElem, TDim, TIdx> const &)
                    -> bool
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // There is no pinning in SYCL but we will lie to the user.
                        return true;
                    }
                };

                //#############################################################################
                //! The BufSycl memory prepareForAsyncCopy trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct PrepareForAsyncCopy<
                    mem::buf::BufSycl<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto prepareForAsyncCopy(
                        mem::buf::BufSycl<TElem, TDim, TIdx> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Everything in SYCL is async by default.
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The BufSycl offset get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetOffset<
                TIdxIntegralConst,
                mem::buf::BufSycl<TElem, TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getOffset(
                   mem::buf::BufSycl<TElem, TDim, TIdx> const &)
                -> TIdx
                {
                    return 0u;
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The BufSycl idx type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct IdxType<
                mem::buf::BufSycl<TElem, TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufCpu.
    namespace mem
    {
        namespace buf
        {
            namespace traits
            {
                //#############################################################################
                //! The BufCpu SYCL device memory mapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Map<
                    mem::buf::BufCpu<TElem, TDim, TIdx>,
                    dev::DevSycl>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto map(
                        mem::buf::BufCpu<TElem, TDim, TIdx> & buf,
                        dev::DevSycl const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                        // No need for this
                    }
                };
                //#############################################################################
                //! The BufSycl device memory unmapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Unmap<
                    mem::buf::BufCpu<TElem, TDim, TIdx>,
                    dev::DevSycl>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unmap(
                        mem::buf::BufCpu<TElem, TDim, TIdx> & buf,
                        dev::DevSycl const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                        // No need for this
                    }
                };
            }
        }
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The BufCpu pointer on SYCL device get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPtrDev<
                    mem::buf::BufCpu<TElem, TDim, TIdx>,
                    dev::DevSycl>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim, TIdx> const & buf,
                        dev::DevSycl const &)
                    -> TElem const *
                    {
                        static_assert(1 == 0, "SYCL does not expose device pointers to host code");
                        throw std::runtime_error{"SYCL does not expose device pointers to host code"};
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim, TIdx> & buf,
                        dev::DevSycl const &)
                    -> TElem *
                    {
                        static_assert(1 == 0, "SYCL does not expose device pointers to host code");
                        throw std::runtime_error{"SYCL does not expose device pointers to host code"};
                    }
                };
            }
        }
    }
}

#include <alpaka/mem/buf/sycl/Copy.hpp>
#include <alpaka/mem/buf/sycl/Set.hpp>

#endif
