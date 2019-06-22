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

#include <boost/hana/for_each.hpp>
#include <boost/hana/prepend.hpp>
#include <boost/hana/transform.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/ext/std/tuple.hpp>

#include <stl-tuple/STLTuple.hpp> // computecpp-sdk

namespace alpaka
{
    namespace kernel
    {
        namespace sycl
        {
            namespace detail
            {
                struct kernel {}; // make ComputeCpp happy

                struct general {};
                struct special : general {};
                template <typename> struct acc_t { using type = int; };

                template <typename Val,
                          typename acc_t<decltype(std::declval<Val>().is_placeholder())>::type = 0>
                inline auto require(cl::sycl::handler& cgh, Val&& val, special)
                {
                    cgh.require(val);
                }

                template <typename Val>
                inline auto require(cl::sycl::handler& cgh, Val&& val, general)
                {
                    // do nothing
                }

                template <typename Val,
                          typename acc_t<decltype(std::declval<Val>().get_pointer())>::type = 0>
                inline auto get_pointer(Val&& val, special)
                {
                    return static_cast<typename std::remove_reference_t<Val>::value_type*>(val.get_pointer());
                }

                template <typename Val>
                inline auto get_pointer(Val&& val, general)
                {
                    return std::forward<Val>(val);
                }

                // Calling this inside the kernel as Hana is a bit tricky
                template <typename... TArgs, std::size_t... Is>
                constexpr auto make_std_args(utility::tuple::Tuple<TArgs...> args,
                                              std::index_sequence<Is...>)
                {
                    return std::make_tuple(utility::tuple::get<Is>(args)...);
                };

                template <typename... TArgs, std::size_t... Is>
                constexpr auto make_hana_args(utility::tuple::Tuple<TArgs...> args,
                                              std::index_sequence<Is...>)
                {
                    return boost::hana::make_tuple(utility::tuple::get<Is>(args)...);
                };

                template <typename TKernelFnObj, typename... TArgs>
                constexpr auto kernel_returns_void(TKernelFnObj, std::tuple<TArgs...> const &)
                {
                    return std::is_same_v<std::result_of_t<
                            TKernelFnObj(TArgs const & ...)>, void>;
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
            // apparently kernels are no longer trivially copyable during the
            // second phase of ComputeCpp's SYCL compilation
            /*static_assert(
                meta::Conjunction<
                    std::is_trivially_copyable<TKernelFnObj>,
                    std::is_trivially_copyable<TArgs>...
                    >::value,
                "The given kernel function object and its arguments have to fulfill is_trivially_copyable!");*/

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
            utility::tuple::Tuple<TArgs...> m_args;

            inline auto operator()(cl::sycl::handler& cgh)
            { 
                // bind all buffers to their accessors
                // use boost::hana::tuple since apply() is explicitly deleted
                // for utility::tuple::Tuple
                boost::hana::for_each(sycl::detail::make_hana_args(
                                        m_args,
                                        std::make_index_sequence<sizeof...(TArgs)>{}),
                [&](auto&& val)
                {
                    sycl::detail::require(cgh,
                                          std::forward<decltype(val)>(val),
                                          sycl::detail::special{});
                });

                const auto work_groups = workdiv::WorkDivMembers<TDim, TIdx>::m_gridBlockExtent;
                const auto group_items = workdiv::WorkDivMembers<TDim, TIdx>::m_blockThreadExtent;
                const auto item_elements = workdiv::WorkDivMembers<TDim, TIdx>::m_threadElemExtent;

                const auto global_size = get_global_size(work_groups, group_items);
                const auto local_size = get_local_size(group_items);

                // copy-by-value so we don't access 'this' on the device
                auto k_func = m_kernelFnObj;
                auto k_args = m_args;

                cgh.parallel_for<sycl::detail::kernel>(
                        cl::sycl::nd_range<TDim::value> {
                            global_size, local_size
                        },
                [=](cl::sycl::nd_item<TDim::value> work_item)
                {
                    // now that we've imported the tuple into device code we
                    // can use std::tuple and Hana
                    auto hana_args = sycl::detail::make_std_args(
                                        k_args,
                                        std::make_index_sequence<sizeof...(TArgs)>{});

                    auto transformed_args = boost::hana::transform(hana_args,
                    [](auto&& val)
                    {
                        return sycl::detail::get_pointer(
                                std::forward<decltype(val)>(val), 
                                sycl::detail::special{});
                    });

                    // add Accelerator to variadic arguments
                    auto kernel_args = boost::hana::prepend(
                            transformed_args, 
                            acc::AccSycl<TDim, TIdx>{item_elements, work_item});

                    static_assert(sycl::detail::kernel_returns_void(k_func, kernel_args),
                                  "The TKernelFnObj is required to return void!");

                    boost::hana::unpack(kernel_args, k_func);
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
