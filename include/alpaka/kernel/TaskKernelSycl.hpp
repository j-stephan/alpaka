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

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/queue/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccSycl.hpp>
#include <alpaka/dev/DevSycl.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/queue/QueueSyclNonBlocking.hpp>
#include <alpaka/queue/QueueSyclBlocking.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <alpaka/acc/Traits.hpp>
    #include <alpaka/dev/Traits.hpp>
    #include <alpaka/workdiv/WorkDivHelpers.hpp>
#endif

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Sycl.hpp>
#include <alpaka/meta/ApplyTuple.hpp>
#include <alpaka/meta/Metafunctions.hpp>

#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif
#include <experimental/type_traits>

namespace alpaka
{
    namespace kernel
    {
        namespace sycl
        {

            //#############################################################################
            //! The SYCL accelerator execution task.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            class TaskKernelSycl final :
                public workdiv::WorkDivMembers<TDim, TIdx>
            {
            public:
                static_assert(
                    meta::Conjunction<
                        std::is_trivially_copyable<
                            TKernelFnObj>,
                        std::is_trivially_copyable<
                            TArgs>...
                        >::value,
                    "The given kernel function object and its arguments have to fulfill is_trivially_copyable!");
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FN_HOST TaskKernelSycl(
                    TWorkDiv && workDiv,
                    TKernelFnObj const & kernelFnObj,
                    TArgs const & ... args) :
                        workdiv::WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv)),
                        m_kernelFnObj(kernelFnObj),
                        m_args(args...)
                {
                    static_assert(
                        dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                        "The work division and the execution task have to be of the same dimensionality!");
                }
                //-----------------------------------------------------------------------------
                TaskKernelSycl(TaskKernelSycl const &) = default;
                //-----------------------------------------------------------------------------
                TaskKernelSycl(TaskKernel &&) = default;
                //-----------------------------------------------------------------------------
                auto operator=(TaskKernelSycl const &) -> TaskKernelSycl & = default;
                //-----------------------------------------------------------------------------
                auto operator=(TaskKernelSycl &&) -> TaskKernelSycl & = default;
                //-----------------------------------------------------------------------------
                ~TaskKernelSycl() = default;

                TKernelFnObj m_kernelFnObj;
                std::tuple<TArgs...> m_args;
                
                auto operator()(cl::sycl::handler& cgh)
                {
                    /*static_assert(
                        std::is_same_v<std::result_of_t<
                            TKernelFnObj(acc::AccSycl<TDim, TIdx> const &, TArgs const & ...)>, void>,
                        "The TKernelFnObj is required to return void!");*/
 
                    // add Accelerator to variadic arguments
                    const auto acc = acc::AccSycl<TDim, TIdx>{extent, work_item};
                    auto kernel_args = std::tuple_cat(std::tie(acc), m_args);

                    // bind buffer to accessor
                    for(auto&& arg : {kernel_args...})
                        require_acc(cgh, std::forward<decltype(arg)>(arg), special);

                    cgh.parallel_for<class sycl_kernel>(
                            cl::sycl::nd_range<dim::Dim<TDim>::value> {
                            // TODO: Global and local work size
                            },
                    [=](cl::sycl::nd_item<dim::Dim<TDim>::value> work_item)
                    {
                        auto transformed_args = std::make_tuple(std::apply([](auto&&... args)
                        {
                            ((return get_pointer(std::forward<decltype(args)>(args), special)), ...);
                        }, kernel_args));

                        std::apply(m_kernelFnObj, transformed_args);
                    });
                }

            private:
                struct general {};
                struct special {};
                template <typename> struct acc { using type = int; };

                template <typename Val,
                          typename acc<decltype(std::declval<Val>().is_placeholder())>::type = 0>
                auto require_acc(cl::sycl::handler& cgh, Val&& val, special)
                {
                    cgh.require(val);
                }

                template <typename Val>
                auto require_acc(cl::sycl::handler& cgh, Val&& val, general)
                {
                    // do nothing
                }

                template <typename Val,
                          typename acc<decltype(std::declval<Val>().is_placeholder())>::type = 0>
                auto get_pointer(Val&& val, special)
                {
                    return val.get_pointer();
                }

                template <typename Val>
                auto get_pointer(Val&& val, general)
                {
                    return val;
                }

            };
        }
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL execution task accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                kernel::TaskKernelSycl<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccSycl<TDim, TIdx>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL execution task device type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                kernel::TaskKernelSycl<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = dev::DevSycl;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL execution task dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                kernel::TaskKernelSycl<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TDim;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                kernel::TaskKernelSycl<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = pltf::PltfSycl;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL execution task idx type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct IdxType<
                kernel::TaskKernelSycl<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TIdx;
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL asynchronous kernel enqueue trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                queue::QueueSyclAsync,
                kernel::TaskKernelSycl<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueSyclAsync & queue,
                    kernel::TaskKernelSycl<TDim, TIdx, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    auto const gridBlockExtent(
                        workdiv::getWorkDiv<Grid, Blocks>(task));
                    auto const blockThreadExtent(
                        workdiv::getWorkDiv<Block, Threads>(task));
                    auto const threadElemExtent(
                        workdiv::getWorkDiv<Thread, Elems>(task));

                    dim3 const gridDim(kernel::sycl::detail::convertVecToSyclExtent(gridBlockExtent));
                    dim3 const blockDim(kernel::sycl::detail::convertVecToSyclExtent(blockThreadExtent));
                    kernel::cuda::detail::checkVecOnly3Dim(threadElemExtent);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << __func__
                        << " global extent: " <<  gridDim.z << " " <<  gridDim.y << " " <<  gridDim.x
                        << " local extent: " <<  blockDim.z << " " <<  blockDim.y << " " <<  blockDim.x
                        << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                    if(!workdiv::isValidWorkDiv<acc::AccSycl<TDim, TIdx>>(dev::getDev(queue), task))
                    {
                        throw std::runtime_error("The given work division is not valid or not supported by the device of type " + acc::getAccName<acc::AccGpuCudaRt<TDim, TIdx>>() + "!");
                    }
#endif

                    // Get the size of the block shared dynamic memory.
                    auto const blockSharedMemDynSizeBytes(
                        meta::apply(
                            [&](TArgs const & ... args)
                            {
                                return
                                    kernel::getBlockSharedMemDynSizeBytes<
                                        acc::AccGpuCudaRt<TDim, TIdx>>(
                                            task.m_kernelFnObj,
                                            blockThreadExtent,
                                            threadElemExtent,
                                            args...);
                            },
                            task.m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the block shared memory idx.
                    std::cout << __func__
                        << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the function attributes.
                    cudaFuncAttributes funcAttrs;
                    cudaFuncGetAttributes(&funcAttrs, kernel::cuda::detail::cudaKernel<TDim, TIdx, TKernelFnObj, TArgs...>);
                    std::cout << __func__
                        << " binaryVersion: " << funcAttrs.binaryVersion
                        << " constSizeBytes: " << funcAttrs.constSizeBytes << " B"
                        << " localSizeBytes: " << funcAttrs.localSizeBytes << " B"
                        << " maxThreadsPerBlock: " << funcAttrs.maxThreadsPerBlock
                        << " numRegs: " << funcAttrs.numRegs
                        << " ptxVersion: " << funcAttrs.ptxVersion
                        << " sharedSizeBytes: " << funcAttrs.sharedSizeBytes << " B"
                        << std::endl;
#endif

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            queue.m_spQueueImpl->m_dev.m_iDevice));
                    // Enqueue the kernel execution.
                    // \NOTE: No const reference (const &) is allowed as the parameter type because the kernel launch language extension expects the arguments by value.
                    // This forces the type of a float argument given with std::forward to this function to be of type float instead of e.g. "float const & __ptr64" (MSVC).
                    // If not given by value, the kernel launch code does not copy the value but the pointer to the value location.
                    meta::apply(
                        [&](TArgs ... args)
                        {
                            kernel::cuda::detail::cudaKernel<TDim, TIdx, TKernelFnObj, TArgs...><<<
                                gridDim,
                                blockDim,
                                static_cast<std::size_t>(blockSharedMemDynSizeBytes),
                                queue.m_spQueueImpl->m_CudaQueue>>>(
                                    threadElemExtent,
                                    task.m_kernelFnObj,
                                    args...);
                        },
                        task.m_args);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // Wait for the kernel execution to finish but do not check error return of this call.
                    // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
                    cudaStreamSynchronize(
                        queue.m_spQueueImpl->m_CudaQueue);
                    std::string const kernelName("'execution of kernel: '" + std::string(typeid(TKernelFnObj).name()) + "' failed with");
                    ::alpaka::cuda::detail::cudaRtCheckLastError(kernelName.c_str(), __FILE__, __LINE__);
#endif
                }
            };
            //#############################################################################
            //! The CUDA synchronous kernel enqueue trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                queue::QueueCudaRtSync,
                kernel::TaskKernelGpuCudaRt<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaRtSync & queue,
                    kernel::TaskKernelGpuCudaRt<TDim, TIdx, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtent.prod()) < available memory idx

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    //std::size_t printfFifoSize;
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << __func__ << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
                    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printfFifoSize*10);
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << __func__ << "INFO: printfFifoSize: " <<  printfFifoSize << std::endl;
#endif
                    auto const gridBlockExtent(
                        workdiv::getWorkDiv<Grid, Blocks>(task));
                    auto const blockThreadExtent(
                        workdiv::getWorkDiv<Block, Threads>(task));
                    auto const threadElemExtent(
                        workdiv::getWorkDiv<Thread, Elems>(task));

                    dim3 const gridDim(kernel::cuda::detail::convertVecToCudaDim(gridBlockExtent));
                    dim3 const blockDim(kernel::cuda::detail::convertVecToCudaDim(blockThreadExtent));
                    kernel::cuda::detail::checkVecOnly3Dim(threadElemExtent);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << __func__ << "gridDim: " <<  gridDim.z << " " <<  gridDim.y << " " <<  gridDim.x << std::endl;
                    std::cout << __func__ << "blockDim: " <<  blockDim.z << " " <<  blockDim.y << " " <<  blockDim.x << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                    if(!workdiv::isValidWorkDiv<acc::AccGpuCudaRt<TDim, TIdx>>(dev::getDev(queue), task))
                    {
                        throw std::runtime_error("The given work division is not valid or not supported by the device of type " + acc::getAccName<acc::AccGpuCudaRt<TDim, TIdx>>() + "!");
                    }
#endif

                    // Get the size of the block shared dynamic memory.
                    auto const blockSharedMemDynSizeBytes(
                        meta::apply(
                            [&](TArgs const & ... args)
                            {
                                return
                                    kernel::getBlockSharedMemDynSizeBytes<
                                        acc::AccGpuCudaRt<TDim, TIdx>>(
                                            task.m_kernelFnObj,
                                            blockThreadExtent,
                                            threadElemExtent,
                                            args...);
                            },
                            task.m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the block shared memory idx.
                    std::cout << __func__
                        << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the function attributes.
                    cudaFuncAttributes funcAttrs;
                    cudaFuncGetAttributes(&funcAttrs, kernel::cuda::detail::cudaKernel<TDim, TIdx, TKernelFnObj, TArgs...>);
                    std::cout << __func__
                        << " binaryVersion: " << funcAttrs.binaryVersion
                        << " constSizeBytes: " << funcAttrs.constSizeBytes << " B"
                        << " localSizeBytes: " << funcAttrs.localSizeBytes << " B"
                        << " maxThreadsPerBlock: " << funcAttrs.maxThreadsPerBlock
                        << " numRegs: " << funcAttrs.numRegs
                        << " ptxVersion: " << funcAttrs.ptxVersion
                        << " sharedSizeBytes: " << funcAttrs.sharedSizeBytes << " B"
                        << std::endl;
#endif

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            queue.m_spQueueImpl->m_dev.m_iDevice));
                    // Enqueue the kernel execution.
                    // \NOTE: No const reference (const &) is allowed as the parameter type because the kernel launch language extension expects the arguments by value.
                    // This forces the type of a float argument given with std::forward to this function to be of type float instead of e.g. "float const & __ptr64" (MSVC).
                    // If not given by value, the kernel launch code does not copy the value but the pointer to the value location.
                    meta::apply(
                        [&](TArgs ... args)
                        {
                            kernel::cuda::detail::cudaKernel<TDim, TIdx, TKernelFnObj, TArgs...><<<
                                gridDim,
                                blockDim,
                                static_cast<std::size_t>(blockSharedMemDynSizeBytes),
                                queue.m_spQueueImpl->m_CudaQueue>>>(
                                    threadElemExtent,
                                    task.m_kernelFnObj,
                                    args...);
                        },
                        task.m_args);

                    // Wait for the kernel execution to finish but do not check error return of this call.
                    // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
                    cudaStreamSynchronize(
                        queue.m_spQueueImpl->m_CudaQueue);
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    std::string const kernelName("'execution of kernel: '" + std::string(typeid(TKernelFnObj).name()) + "' failed with");
                    ::alpaka::cuda::detail::cudaRtCheckLastError(kernelName.c_str(), __FILE__, __LINE__);
#endif
                }
            };
        }
    }
}

#endif
