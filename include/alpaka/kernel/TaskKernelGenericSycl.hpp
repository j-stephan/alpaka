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

#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>
#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Sycl.hpp>

#include <stl-tuple/STLTuple.hpp> // computecpp-sdk

#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>


namespace alpaka
{
    namespace detail
    {
        template <typename TName>
        struct kernel {}; // SYCL kernel names must be globally visible

        template <typename... TArgs, std::size_t... Is>
        constexpr auto make_device_args(std::tuple<TArgs...> args, std::index_sequence<Is...>)
        {
            return utility::tuple::make_tuple(std::get<Is>(args)...);
        }

        template <typename TKernelFnObj, typename... TArgs>
        constexpr auto kernel_returns_void(TKernelFnObj, utility::tuple::Tuple<TArgs...> const &)
        {
            return std::is_same_v<std::result_of_t<TKernelFnObj(TArgs const & ...)>, void>;
        }

        template <typename TFunc, typename... TArgs, std::size_t... Is>
        constexpr auto apply_impl(TFunc&& f, utility::tuple::Tuple<TArgs...> t, std::index_sequence<Is...>)
        {
            f(utility::tuple::get<Is>(t)...);
        }

        template <typename TFunc, typename... TArgs>
        constexpr auto apply(TFunc&& f, utility::tuple::Tuple<TArgs...> t)
        {
            apply_impl(std::forward<TFunc>(f), t, std::make_index_sequence<sizeof...(TArgs)>{});
        }
    } // namespace detail

    //#############################################################################
    //! The SYCL accelerator execution task.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelGenericSycl final : public WorkDivMembers<TDim, TIdx>
    {
    public:
        static_assert(TDim::value > 0 && TDim::value <= 3, "Invalid kernel dimensionality");
        //-----------------------------------------------------------------------------
        template<typename TWorkDiv>
        ALPAKA_FN_HOST TaskKernelGenericSycl(TWorkDiv && workDiv, TKernelFnObj const & kernelFnObj,
                                          TArgs const & ... args)
        : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
        , m_kernelFnObj{kernelFnObj}
        , m_args{args...}
        {
            static_assert(
                Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                "The work division and the execution task have to be of the same dimensionality!");
        }

        //-----------------------------------------------------------------------------
        TaskKernelGenericSycl(TaskKernelGenericSycl const &) = default;
        //-----------------------------------------------------------------------------
        TaskKernelGenericSycl(TaskKernelGenericSycl &&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(TaskKernelGenericSycl const &) -> TaskKernelGenericSycl & = default;
        //-----------------------------------------------------------------------------
        auto operator=(TaskKernelGenericSycl &&) -> TaskKernelGenericSycl & = default;
        //-----------------------------------------------------------------------------
        ~TaskKernelGenericSycl() = default;

        inline auto operator()(cl::sycl::handler& cgh)
        { 
            using namespace cl::sycl;

            // create shared predicate counter -- needed for block synchronization with predicates
            auto pred_counter_acc = accessor<int, 0, access::mode::read_write, access::target::local>{cgh};

            // transform to device tuple so we can copy the arguments into device code. We can't use std::tuple for
            // this step because it isn't standard layout and thus prohibited to be copied into SYCL device code.
            auto device_args = make_device_args(m_args, std::make_index_sequence<sizeof...(TArgs)>{});

            const auto work_groups = WorkDivMembers<TDim, TIdx>::m_gridBlockExtent;
            const auto group_items = WorkDivMembers<TDim, TIdx>::m_blockThreadExtent;
            const auto item_elements = WorkDivMembers<TDim, TIdx>::m_threadElemExtent;

            const auto global_size = get_global_size(work_groups, group_items);
            const auto local_size = get_local_size(group_items);

            // allocate shared memory -- needs at least 1 byte to make XRT happy
            const auto shared_mem_bytes = std::max(1ul, std::apply([&](auto const& ... args)
            {
                return getBlockSharedMemDynSizeBytes<TAcc>(m_kernelFnObj, group_items, item_elements, args...);
            }, m_args));

            auto shared_accessor = accessor<unsigned char, 1, access::mode::read_write, access::target::local>{
                                                range<1>{shared_mem_bytes}, cgh};

            // copy-by-value so we don't access 'this' on the device
            auto k_func = m_kernelFnObj;

            // wait for previous kernels to complete
            cgh.depends_on(m_dependencies);

            cgh.parallel_for<kernel<TKernelFnObj>>(nd_range<TDim::value>{global_size, local_size},
            [=](nd_item<TDim::value> work_item)
            {
                auto pred_counter = atomic_ref<int, memory_order::relaxed, memory_scope::work_group,
                                               access::address_space::local_space>{pred_counter_acc};

                // add alpaka accelerator to variadic arguments
                auto kernel_args = utility::tuple::append(utility::tuple::make_tuple(
                                    TAcc{item_elements, work_item, shared_accessor, pred_counter}),
                                    device_args);

                // Now we can check for correctness.
                static_assert(kernel_returns_void(k_func, kernel_args), "The TKernelFnObj must return void!");
                apply(k_func, kernel_args);
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

        TKernelFnObj m_kernelFnObj;
        std::tuple<TArgs...> m_args;
        std::vector<cl::sycl::event> m_dependencies = {};
        std::shared_ptr<std::shared_mutex> mutex_ptr{std::make_shared<std::shared_mutex>()};
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL execution task accelerator type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TAcc;
        };

        //#############################################################################
        //! The SYCL execution task device type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = DevType<TAcc>::type;
        };

        //#############################################################################
        //! The SYCL execution task platform type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PltfType<TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = PltfType<TAcc>::type;
        };

        //#############################################################################
        //! The SYCL execution task dimension getter trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The SYCL execution task idx type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelGenericSycl<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };
    }
}

#endif
