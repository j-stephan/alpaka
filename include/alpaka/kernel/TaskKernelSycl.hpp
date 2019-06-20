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

#include <boost/hana/ap.hpp>
#include <boost/hana/prepend.hpp>
#include <boost/hana/transform.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/ext/std/tuple.hpp>
#include <boost/hana/fwd/for_each.hpp>

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
                          typename acc_t<decltype(std::declval<Val>().get_pointer())>::type = 0>
                auto get_pointer(Val&& val, special)
                {
                    auto ptr = val.get_pointer();
                    return static_cast<typename decltype(ptr)::element_type*>(ptr);
                    // return static_cast<typename std::remove_reference_t<Val>::pointer_t>(val.get_pointer());
                    // return static_cast<typename std::remove_reference_t<Val>::value_type*>(val.get_pointer());
                }

                template <typename Val>
                auto get_pointer(Val&& val, general)
                {
                    return std::forward<Val>(val);
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
            static_assert(
                meta::Conjunction<
                    std::is_trivially_copyable<TKernelFnObj>,
                    std::is_trivially_copyable<TArgs>...
                    >::value,
                "The given kernel function object and its arguments have to fulfill is_trivially_copyable!");

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

            auto operator()(cl::sycl::handler& cgh)
            { 
                // bind all buffers to their accessors
                boost::hana::for_each(m_args, [&](auto&& val)
                {
                    sycl::detail::require_acc(cgh, std::forward<decltype(val)>(val), sycl::detail::special{});
                });

                const auto work_groups = workdiv::WorkDivMembers<TDim, TIdx>::m_gridBlockExtent;
                const auto group_items = workdiv::WorkDivMembers<TDim, TIdx>::m_blockThreadExtent;
                const auto item_elements = workdiv::WorkDivMembers<TDim, TIdx>::m_threadElemExtent;

                const auto global_size = get_global_size(work_groups, group_items);
                const auto local_size = get_local_size(group_items);

                // copy-by-value so we don't access 'this' on the device
                auto k_func = m_kernelFnObj;
                // create Hana tuple from std tuple
                auto k_args = boost::hana::unpack(m_args, [](auto&&... args)
                { 
                    return boost::hana::make_tuple(args...); 
                });                

                cgh.parallel_for<sycl::detail::kernel>(
                        cl::sycl::nd_range<TDim::value> {
                            global_size, local_size
                        },
                [=](cl::sycl::nd_item<TDim::value> work_item)
                {
                    auto transformed_args = boost::hana::transform(k_args,
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

                    // TODO: Find a way to expand the kernel_args tuple (again...) and
                    // do the static_assert
                    /*static_assert(
                        std::is_same_v<std::result_of_t<
                            TKernelFnObj(acc::AccSycl<TDim, TIdx> const &, TArgs const & ...)>, void>,
                        "The TKernelFnObj is required to return void!");*/

                    // std::apply(k_obj, kernel_args);
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
