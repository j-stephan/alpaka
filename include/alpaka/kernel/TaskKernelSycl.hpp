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
    namespace detail
    {
        template <typename TName>
        struct kernel {}; // SYCL kernel names must be globally visible

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
                return buf.buf.template get_access<cl::sycl::access::mode::read,
                                                   cl::sycl::access::target::global_buffer>(cgh);
            }
            else
            {
                return buf.buf.template get_access<cl::sycl::access::mode::read_write,
                                                   cl::sycl::access::target::global_buffer>(cgh);
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

        template <typename TAccessor,
                  typename acc_t<decltype(std::declval<TAccessor>().get_pointer())>::type = 0>
        inline auto get_dummy_pointer(TAccessor accessor, special)
        {
            return static_cast<typename TAccessor::value_type*>(nullptr);
        }

        template <typename TAccessor>
        inline auto get_dummy_pointer(TAccessor accessor, general)
        {
            return accessor;
        }

        template <typename... TArgs, std::size_t... Is>
        constexpr auto make_device_args(std::tuple<TArgs...> args,
                                        std::index_sequence<Is...>)
        {
            return utility::tuple::make_tuple(std::get<Is>(args)...);
        }

        template <typename TKernelFnObj, typename... TArgs>
        constexpr auto kernel_returns_void(TKernelFnObj,
                                           utility::tuple::Tuple<TArgs...> const &)
        {
            return std::is_same_v<std::result_of_t<
                    TKernelFnObj(TArgs const & ...)>, void>;
        }

        template <typename... TArgs, std::size_t... Is>
        constexpr auto transform(utility::tuple::Tuple<TArgs...> args,
                                 std::index_sequence<Is...>)
        {
            return utility::tuple::make_tuple(get_pointer(utility::tuple::get<Is>(args),
                                              special{})...);
        }

        template <typename... TArgs, std::size_t... Is>
        constexpr auto make_dummies(std::tuple<TArgs...> args,
                                    std::index_sequence<Is...>)
        {
            return std::make_tuple(get_dummy_pointer(std::get<Is>(args),
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

        template <typename TFunc, typename... TArgs, std::size_t... Is>
        constexpr auto apply_impl(TFunc&& f,
                                  utility::tuple::Tuple<TArgs...> t,
                                  std::index_sequence<Is...>)
        {
            f(utility::tuple::get<Is>(t)...);
        }

        template <typename TFunc, typename... TArgs>
        constexpr auto apply(TFunc&& f, utility::tuple::Tuple<TArgs...> t)
        {
            apply_impl(std::forward<TFunc>(f),
                       t,
                       std::make_index_sequence<sizeof...(TArgs)>{});
        }
    } // namespace detail

    //#############################################################################
    //! The SYCL accelerator execution task.
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelSycl final : public WorkDivMembers<TDim, TIdx>
    {
    public:

        static_assert(TDim::value > 0 && TDim::value <= 3, "Invalid kernel dimensionality");
        //-----------------------------------------------------------------------------
        template<typename TWorkDiv>
        ALPAKA_FN_HOST TaskKernelSycl(TWorkDiv && workDiv, TKernelFnObj const & kernelFnObj, TArgs const & ... args)
        : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
        , m_kernelFnObj{kernelFnObj}
        , m_args{args...}
        {
            static_assert(
                Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
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
            auto pred_counter = cl::sycl::accessor<int, 0, cl::sycl::access::mode::atomic, cl::sycl::access::target::local>{cgh};


            // bind all buffers to their accessors
            auto std_accessor_args = detail::bind_buffers<TDim>(cgh, m_args, std::make_index_sequence<sizeof...(TArgs)>{});

            // transform to device tuple so we can copy the arguments into
            // device code. We can't use std::tuple for this step because
            // it isn't standard layout and thus prohibited to be copied
            // into SYCL device code.
            auto accessor_args = detail::make_device_args(std_accessor_args, std::make_index_sequence<sizeof...(TArgs)>{});

            const auto work_groups = WorkDivMembers<TDim, TIdx>::m_gridBlockExtent;
            const auto group_items = WorkDivMembers<TDim, TIdx>::m_blockThreadExtent;
            const auto item_elements = WorkDivMembers<TDim, TIdx>::m_threadElemExtent;

            const auto global_size = get_global_size(work_groups, group_items);
            const auto local_size = get_local_size(group_items);

            // transform accessors to dummy pointers
            auto dummy_args = detail::make_dummies(
                                    std_accessor_args, // intentional, only for shared mem allocation
                                    std::make_index_sequence<sizeof...(TArgs)>{});
            
            // allocate shared memory -- needs at least 1 byte to make XRT happy
            const auto shared_mem_bytes = std::max(1ul, std::apply(
            [&](const auto & ... args)
            {
                return getBlockSharedMemDynSizeBytes<AccSycl<TDim, TIdx>>(m_kernelFnObj, group_items, item_elements,
                                                                          args...);
            }, dummy_args));

            // copy-by-value so we don't access 'this' on the device
            auto k_func = m_kernelFnObj;

            auto shared_accessor = cl::sycl::accessor<unsigned char, 1,
                                                      cl::sycl::access::mode::read_write,
                                                      cl::sycl::access::target::local>{
                                                          cl::sycl::range<1>{shared_mem_bytes},
                                                          cgh};

            cgh.parallel_for<detail::kernel<TKernelFnObj>>(cl::sycl::nd_range<TDim::value>{global_size, local_size},
            [=](cl::sycl::nd_item<TDim::value> work_item)
            {
                // transform accessors to pointers
                auto transformed_args = detail::transform(accessor_args, std::make_index_sequence<sizeof...(TArgs)>{});

                // add alpaka accelerator to variadic arguments
                using acc_type = AccSycl<TDim, TIdx>;
                auto kernel_args = utility::tuple::append(utility::tuple::make_tuple(
                                    AccSycl<TDim, TIdx>{item_elements, work_item, shared_accessor, pred_counter}),
                                    transformed_args);

                // Now we can check for correctness.
                static_assert(detail::kernel_returns_void(k_func, kernel_args),
                              "The TKernelFnObj is required to return void!");

                detail::apply(k_func, kernel_args);
            });
        }

    private:
        auto get_global_size(const Vec<TDim, TIdx>& work_groups, const Vec<TDim, TIdx>& group_items)
        {
            using namespace cl::sycl;

            if constexpr(TDim::value == 1)
                return range<1>{work_groups[0] * group_items[0]};
            else if constexpr(TDim::value == 2)
                return range<2>{work_groups[0] * group_items[0], work_groups[1] * group_items[1]};
            else
            {
                return range<3>{work_groups[0] * group_items[0], work_groups[1] * group_items[1],
                                work_groups[2] * group_items[2]};
            }
        }

        auto get_local_size(const Vec<TDim, TIdx>& group_items)
        {
            using namespace cl::sycl;

            if constexpr(TDim::value == 1)
                return range<1>{group_items[0]};
            else if constexpr(TDim::value == 2)
                return range<2>{group_items[0], group_items[1]};
            else
                return range<3>{group_items[0], group_items[1], group_items[2]};
        }
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL execution task accelerator type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelSycl<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = AccSycl<TDim, TIdx>;
        };

        //#############################################################################
        //! The SYCL execution task device type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelSycl<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = DevSycl;
        };

        //#############################################################################
        //! The SYCL execution task dimension getter trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelSycl<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The SYCL execution task platform type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PltfType<TaskKernelSycl<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = PltfSycl;
        };

        //#############################################################################
        //! The SYCL execution task idx type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelSycl<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };
    }
}

#endif
