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

#include <alpaka/atomic/Op.hpp>
#include <alpaka/atomic/Traits.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/meta/DependentFalseType.hpp>

#include <CL/sycl.hpp>

#include <type_traits>

namespace alpaka
{
    //#############################################################################
    //! The SYCL accelerator atomic ops.
    //
    //  Atomics can used in the hierarchy level grids, blocks and threads.
    //  Atomics are not guaranteed to be safe between devices
    class AtomicUniformSycl
    {
    public:

        //-----------------------------------------------------------------------------
        AtomicUniformSycl() = default;
        //-----------------------------------------------------------------------------
        AtomicUniformSycl(AtomicUniformSycl const &) = default;
        //-----------------------------------------------------------------------------
        AtomicUniformSycl(AtomicUniformSycl &&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AtomicUniformSycl const &) -> AtomicUniformSycl & = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AtomicUniformSycl &&) -> AtomicUniformSycl & = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~AtomicUniformSycl() = default;
    };

    namespace detail
    {
        template <typename THierarchy>
        struct SyclMemoryScope {};

        template <>
        struct SyclMemoryScope<hierarchy::Grids>
        {
            static constexpr auto value = cl::sycl::memory_scope::device;
        };

        template<>
        struct SyclMemoryScope<hierarchy::Blocks>
        {
            static constexpr auto value = cl::sycl::memory_scope::work_group;
        };

        template<>
        struct SyclMemoryScope<hierarchy::Threads>
        {
            static constexpr auto value = cl::sycl::memory_scope::work_item;
        };
    }

    namespace traits
    {
        //-----------------------------------------------------------------------------
        // Add.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicAdd, AtomicUniformSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics do not support this type");

            //-----------------------------------------------------------------------------
            //
            static auto atomicOp(AtomicUniformSycl const &, T * const addr, T const & value) -> T
            {
                using namespace cl::sycl;

                auto ref = atomic_ref<memory_order_relaxed, detail::SyclMemoryScope<THierarchy>::value>{*addr};
                return ref.fetch_add(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Sub.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicSub, AtomicUniformSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics do not support this type");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicUniformSycl const &, T * const addr, T const & value) -> T
            {
                using namespace cl::sycl;

                auto ref = atomic_ref<memory_order_relaxed, detail::SyclMemoryScope<THierarchy>::value>{*addr};
                return ref.fetch_sub(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Min.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicMin, AtomicUniformSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics don't support this type");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicUniformSycl const &, T * const addr, T const & value) -> T
            {
                using namespace cl::sycl;

                auto ref = atomic_ref<memory_order_relaxed, detail::SyclMemoryScope<THierarchy>::value>{*addr};
                return ref.fetch_min(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Max.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicMax, AtomicUniformSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics don't support this type");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicUniformSycl const &, T * const addr, T const & value) -> T
            {
                using namespace cl::sycl;

                auto ref = atomic_ref<memory_order_relaxed, detail::SyclMemoryScope<THierarchy>::value>{*addr};
                return ref.fetch_max(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Exch.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicExch, AtomicUniformSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics don't support this type");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicUniformSycl const &, T * const addr, T const & value) -> T
            {
                using namespace cl::sycl;

                auto ref = atomic_ref<memory_order_relaxed, detail::SyclMemoryScope<THierarchy>::value>{*addr};
                return ref.exchange(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Inc.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicInc, AtomicUniformSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics do not support this type");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicUniformSycl const &, T * const addr, T const & value) -> T
            {
                using namespace cl::sycl;

                auto ref = atomic_ref<memory_order_relaxed, detail::SyclMemoryScope<THierarchy>::value>{*addr};
                return ref.fetch_add(T{1});
            }
        };

        //-----------------------------------------------------------------------------
        // Dec.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicDec, AtomicUniformSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics do not support this type");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicUniformSycl const &, T * const addr, T const & value) -> T
            {
                using namespace cl::sycl;

                auto ref = atomic_ref<memory_order_relaxed, detail::SyclMemoryScope<THierarchy>::value>{*addr};
                return ref.fetch_sub(T{1});
            }
        };

        //-----------------------------------------------------------------------------
        // And.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicAnd, AtomicUniformSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T>, "Bitwise operations are not supported on non-integral types.");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicUniformSycl const &, T * const addr, T const & value) -> T
            {
                using namespace cl::sycl;

                auto ref = atomic_ref<memory_order_relaxed, detail::SyclMemoryScope<THierarchy>::value>{*addr};
                return ref.fetch_and(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Or.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicOr, AtomicUniformSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T>, "Bitwise operations are not supported on non-integral types.");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicUniformSycl const &, T * const addr, T const & value) -> T
            {
                using namespace cl::sycl;

                auto ref = atomic_ref<memory_order_relaxed, detail::SyclMemoryScope<THierarchy>::value>{*addr};
                return ref.fetch_or(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Xor.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicXor, AtomicUniformSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T>, "Bitwise operations are not supported on non-integral types.");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicUniformSycl const &, T * const addr, T const & value) -> T
            {
                using namespace cl::sycl;

                auto ref = atomic_ref<memory_order_relaxed, detail::SyclMemoryScope<THierarchy>::value>{*addr};
                return ref.fetch_xor(value);
            }
        };

        //-----------------------------------------------------------------------------
        // Cas.

        //-----------------------------------------------------------------------------
        //! The SYCL accelerator atomic operation.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicCas, AtomicUniformSycl, T, THierarchy>
        {
            static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                          "SYCL atomics don't support this type");
            //-----------------------------------------------------------------------------
            static auto atomicOp(AtomicUniformSycl const &, T * const addr, T const & compare, T const & value) -> T
            {
                using namespace cl::sycl;

                auto ref = atomic_ref<memory_order_relaxed, detail::SyclMemoryScope<THierarchy>::value>{*addr};

                // SYCL stores the value in *addr to the "compare" parameter if the values are not equal. Since
                // alpaka's interface does not expect this we need to copy "compare" to this function and forget it
                // afterwards.
                auto tmp = compare;

                // We always want to return the old value at the end.
                const auto old = ref.load();

                // This returns a bool telling us if the exchange happened or not. Useless in this case.
                ref.compare_exchange_strong(tmp, value);

                return old;
            }
        };
    }
}

#endif