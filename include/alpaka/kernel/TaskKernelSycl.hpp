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

#include <stl-tuple/STLTuple.hpp> // computecpp-sdk

namespace alpaka
{
    namespace kernel
    {
        namespace sycl
        {
            namespace detail
            {
                template <typename Name>
                struct kernel {}; // make ComputeCpp happy by declaring the
                                  // kernel name in a globally visible namespace

                struct general {};
                struct special : general {};
                template <typename> struct acc_t { using type = int; };

                template <typename TDim,
                          typename TBuf,
                          typename acc_t<typename TBuf::is_alpaka_sycl_buffer_wrapper>::type = 0>
                inline auto get_access(cl::sycl::handler& cgh, TBuf buf, special)
                {
                    using buf_type = typename TBuf::buf_type;
                    using value_type = typename buf_type::value_type;
                    if constexpr(std::is_const_v<buf_type>)
                    {
                        return cl::sycl::accessor<value_type, TDim::value,
                                                  cl::sycl::access::mode::read,
                                                  cl::sycl::access::target::global_buffer>{*(buf.buf), cgh};
                    }
                    else
                    {
                        return cl::sycl::accessor<value_type, TDim::value,
                                                  cl::sycl::access::mode::read_write,
                                                  cl::sycl::access::target::global_buffer>{*(buf.buf), cgh};
                    }
                }

                template <typename TDim, typename TBuf>
                inline auto get_access(cl::sycl::handler& cgh, TBuf buf, general)
                {
                    return buf;
                }

                template <typename TAccessor,
                          typename acc_t<decltype(std::declval<TAccessor>().get_pointer())>::type = 0>
                inline auto get_pointer(TAccessor accessor, special)
                {
                    return static_cast<typename TAccessor::value_type*>(accessor.get_pointer());
                }

                template <typename TAccessor>
                inline auto get_pointer(TAccessor accessor, general)
                {
                    return accessor;
                }

                // we only need the device tuple for copying the arguments into
                // device code. Once we are done we can use a std::tuple again.
                template <typename... TArgs, std::size_t... Is>
                constexpr auto make_std_args(utility::tuple::Tuple<TArgs...> args,
                                             std::index_sequence<Is...>)
                {
                    return std::make_tuple(utility::tuple::get<Is>(args)...);
                }

                template <typename... TArgs, std::size_t... Is>
                constexpr auto make_device_args(std::tuple<TArgs...> args,
                                                std::index_sequence<Is...>)
                {
                    return utility::tuple::make_tuple(std::get<Is>(args)...);
                }

                template <typename TKernelFnObj, typename... TArgs>
                constexpr auto kernel_returns_void(TKernelFnObj,
                                                   std::tuple<TArgs...> const &)
                {
                    return std::is_same_v<std::result_of_t<
                            TKernelFnObj(TArgs const & ...)>, void>;
                }

                template <typename TKernelFnObj, typename... TArgs>
                constexpr auto kernel_is_trivial(TKernelFnObj,
                                                 std::tuple<TArgs...> const &)
                {
                    return std::conjunction_v<
                            std::is_trivially_copyable<TKernelFnObj>,
                            std::is_trivially_copyable<TArgs>...>;
                }

                template <typename... TArgs, std::size_t... Is>
                constexpr auto transform(std::tuple<TArgs...> args,
                                         std::index_sequence<Is...>)
                {
                    return std::make_tuple(get_pointer(std::get<Is>(args),
                                                       special{})...);
                }

                template <typename TDim, typename... TArgs, std::size_t... Is>
                constexpr auto bind_buffers(cl::sycl::handler& cgh,
                                            std::tuple<TArgs...> args,
                                            std::index_sequence<Is...>)
                {
                    return std::make_tuple(get_access<TDim>(cgh,
                                                            std::get<Is>(args),
                                                            special{})...);
                }
            } // namespace detail
        } // namespace sycl
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

            static_assert(TDim::value > 0 && TDim::value <= 3,
                          "Invalid kernel dimensionality");
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST TaskKernelSycl(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) :
                    workdiv::WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj{kernelFnObj},
                    m_args{args...}
            {
                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the execution task have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            TaskKernelSycl(TaskKernelSycl const &) = default;
            //-----------------------------------------------------------------------------
            TaskKernelSycl(TaskKernelSycl &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelSycl const &) -> TaskKernelSycl & = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelSycl &&) -> TaskKernelSycl & = default;
            //-----------------------------------------------------------------------------
            ~TaskKernelSycl() = default;

            TKernelFnObj m_kernelFnObj;
            std::tuple<TArgs...> m_args;

            inline auto operator()(cl::sycl::handler& cgh)
            { 
                // create shared predicate counter -- needed for block
                // synchronization with predicates
                auto pred_counter = cl::sycl::accessor<int, 0,
                                                       cl::sycl::access::mode::atomic,
                                                       cl::sycl::access::target::local>{cgh};
                
                // bind all buffers to their accessors
                auto std_accessor_args = sycl::detail::bind_buffers<TDim>(
                                            cgh,
                                            m_args,
                                            std::make_index_sequence<sizeof...(TArgs)>{});

                // transform to device tuple so we can copy the arguments into
                // device code. We can't use std::tuple for this step because
                // it isn't standard layout and thus prohibited to be copied
                // into SYCL device code.
                auto accessor_args = sycl::detail::make_device_args(
                                        std_accessor_args,
                                        std::make_index_sequence<sizeof...(TArgs)>{});

                const auto work_groups = workdiv::WorkDivMembers<TDim, TIdx>::m_gridBlockExtent;
                const auto group_items = workdiv::WorkDivMembers<TDim, TIdx>::m_blockThreadExtent;
                const auto item_elements = workdiv::WorkDivMembers<TDim, TIdx>::m_threadElemExtent;

                const auto global_size = get_global_size(work_groups, group_items);
                const auto local_size = get_local_size(group_items);

                // copy-by-value so we don't access 'this' on the device
                auto k_func = m_kernelFnObj;

                using kernel_type = sycl::detail::kernel<TKernelFnObj>;
                cgh.parallel_for<kernel_type>(
                        cl::sycl::nd_range<TDim::value> {
                            global_size, local_size
                        },
                [=](cl::sycl::nd_item<TDim::value> work_item)
                {
                    // now that we've imported the tuple into device code we
                    // can use std::tuple again
                    auto std_args = sycl::detail::make_std_args(
                                        accessor_args,
                                        std::make_index_sequence<sizeof...(TArgs)>{});

                    // transform accessors to pointers
                    auto transformed_args = sycl::detail::transform(
                                        std_args,
                                        std::make_index_sequence<sizeof...(TArgs)>{});

                    // add alpaka accelerator to variadic arguments
                    using acc_type = acc::AccSycl<TDim, TIdx>;
                    auto kernel_args = std::tuple_cat(std::make_tuple(
                                        acc::AccSycl<TDim, TIdx>{item_elements,
                                                                 work_item,
                                                                 pred_counter}),
                                        transformed_args);

                    // Now we can check for correctness.
                    static_assert(sycl::detail::kernel_returns_void(k_func, kernel_args),
                                  "The TKernelFnObj is required to return void!");

                    // FIXME: Kernels are not trivial in SYCL. Do we need this
                    // check at all in the SYCL backend?
                    /*static_assert(sycl::detail::kernel_is_trivial(k_func, kernel_args),
                                  "The given kernel function object and its arguments have to fulfill is_trivially_copyable!");*/

                    std::apply(k_func, kernel_args);
                });
            }

        private:
            auto get_global_size(const vec::Vec<TDim, TIdx>& work_groups,
                                 const vec::Vec<TDim, TIdx>& group_items)
            {

                if constexpr(TDim::value == 1)
                    return cl::sycl::range<1>{work_groups[0] * group_items[0]};
                else if constexpr(TDim::value == 2)
                {
                    return cl::sycl::range<2>{work_groups[0] * group_items[0],
                                              work_groups[1] * group_items[1]};
                }
                else
                {
                    return cl::sycl::range<3>{work_groups[0] * group_items[0],
                                              work_groups[1] * group_items[1],
                                              work_groups[2] * group_items[2]};
                }
            }

            auto get_local_size(const vec::Vec<TDim, TIdx>& group_items)
            {
                if constexpr(TDim::value == 1)
                    return cl::sycl::range<1>{group_items[0]};
                else if constexpr(TDim::value == 2)
                    return cl::sycl::range<2>{group_items[0], group_items[1]};
                else
                {
                    return cl::sycl::range<3>{group_items[0], group_items[1],
                                              group_items[2]};
                }
            }
        };
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
}

#endif
