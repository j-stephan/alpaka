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

#include <alpaka/core/Unused.hpp>
#include <alpaka/atomic/Op.hpp>
#include <alpaka/atomic/Traits.hpp>
#include <alpaka/meta/DependentFalseType.hpp>

#include <climits>

namespace alpaka
{
    namespace atomic
    {
        //#############################################################################
        //! The SYCL accelerator atomic ops.
        //
        //  Atomics can used in the hierarchy level grids, blocks and threads.
        //  Atomics are not guaranteed to be safe between devices
        class AtomicSycl
        {
        public:

            //-----------------------------------------------------------------------------
            AtomicSycl() = default;
            //-----------------------------------------------------------------------------
            AtomicSycl(AtomicSycl const &) = delete;
            //-----------------------------------------------------------------------------
            AtomicSycl(AtomicSycl &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(AtomicSycl const &) -> AtomicSycl & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(AtomicSycl &&) -> AtomicSycl & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AtomicSycl() = default;
        };

        namespace traits
        {
            /*
             * FIXME: The following ops are specialized for each possible type
             * permitted in SYCL. Since the API call is the same for all types
             * this could easily be abstracted away by a template parameter.
             * Unfortunately, some SYCL implementations mess this up as they add
             * the memory space to the type information, in which case the type
             * T becomes something like __global int. Ironically, this annotated
             * type breaks the SYCL API, because we can't use it for the type
             * parameter of cl::sycl::atomic. The value operand will also turn
             * have this type which makes no sense at all.
             *
             * There is also no way to extract the __global, __local etc. from
             * the type, so we can't overload for global_ptr, local_ptr and so
             * on. Tests have shown that cl::sycl::global_ptr will happily
             * swallow a __local int at compile time (and probably break at
             * runtime). OTOH, this wouldn't be a good idea anyway, since the
             * type annotations are implementation-defined.
             */

            //#############################################################################
            //! The specializations to execute the requested atomic ops of the SYCL accelerator.

            //-----------------------------------------------------------------------------
            // Add.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicSycl,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &, 
                    int * const addr,
                    int const & value)
                -> int
                {
                    // TODO: solve shared memory case
                    auto addr_ptr = cl::sycl::global_ptr<int>{addr};
                    auto atomic_addr = cl::sycl::atomic<int>{addr_ptr};

                    return cl::sycl::atomic_fetch_add(atomic_addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicSycl,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &, 
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    // TODO: solve shared memory case
                    auto addr_ptr = cl::sycl::global_ptr<unsigned int>{addr};
                    auto atomic_addr = cl::sycl::atomic<unsigned int>{addr_ptr};

                    return cl::sycl::atomic_fetch_add(atomic_addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicSycl,
                long,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &, 
                    long * const addr,
                    long const & value)
                -> long
                {
                    // TODO: solve shared memory case
                    auto addr_ptr = cl::sycl::global_ptr<long>{addr};
                    auto atomic_addr = cl::sycl::atomic<long>{addr_ptr};

                    return cl::sycl::atomic_fetch_add(atomic_addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicSycl,
                unsigned long,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &, 
                    unsigned long * const addr,
                    unsigned long const & value)
                -> unsigned long
                {
                    // TODO: Check for __global
                    // TODO: solve shared memory case
                    auto addr_ptr = cl::sycl::global_ptr<unsigned long>{addr};
                    auto atomic_addr = cl::sycl::atomic<unsigned long>{addr_ptr};

                    return cl::sycl::atomic_fetch_add(atomic_addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicSycl,
                long long,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &, 
                    long long * const addr,
                    long long const & value)
                -> long long
                {
                    // TODO: solve shared memory case
                    auto addr_ptr = cl::sycl::global_ptr<long long>{addr};
                    auto atomic_addr = cl::sycl::atomic<long long>{addr_ptr};

                    return cl::sycl::atomic_fetch_add(atomic_addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicSycl,
                unsigned long long,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &, 
                    unsigned long long * const addr,
                    unsigned long long const & value)
                -> unsigned long long
                {
                    // TODO: solve shared memory case
                    auto addr_ptr = cl::sycl::global_ptr<unsigned long long>{addr};
                    auto atomic_addr = cl::sycl::atomic<unsigned long long>{addr_ptr};

                    return cl::sycl::atomic_fetch_add(atomic_addr, value);
                }

                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    cl::sycl::global_ptr<unsigned long long> addr,
                    unsigned long long const & value)
                -> unsigned long long
                {
                    auto atomic_addr = cl::sycl::atomic<unsigned long long>{addr};
                    return cl::sycl::atomic_fetch_add(atomic_addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            // Sub.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Sub,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO: solve shared memory case
                    // TODO
                    // return cl::sycl::atomic_fetch_sub(atomic<T, addressSpace> object, T operand)
                }
            };

            //-----------------------------------------------------------------------------
            // Min.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
               typename T,
               typename THierarchy>
            struct AtomicOp<
                op::Min,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // return cl::sycl::atomic_fetch_min(atomic<T, addressSpace> object, T operand)
                }
            };

            //-----------------------------------------------------------------------------
            // Max.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Max,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // return cl::sycl::atomic_fetch_max(atomic<T, addressSpace> object, T operand)
                }
            };

            //-----------------------------------------------------------------------------
            // Exch.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // return cl::sycl::atomic_exchange(atomic<T, addressSpace> object, T operand)
                }
            };

            //-----------------------------------------------------------------------------
            // Inc.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Inc,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // work-around with compare_and_exchange
                }
            };

            //-----------------------------------------------------------------------------
            // Dec.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Dec,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // work-around with compare_and_exchange
                }
            };

            //-----------------------------------------------------------------------------
            // And.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // return cl::sycl::atomic_fetch_and(atomic<T, addressSpace> object, T operand)
                }
            };

            //-----------------------------------------------------------------------------
            // Or.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // return cl::sycl::atomic_fetch_or(atomic<T, addressSpace> object, T operand)
                }
            };

            //-----------------------------------------------------------------------------
            // Xor.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // return cl::sycl::atomic_fetch_xor(atomic<T, addressSpace> object, T operand)
                }
            };

            //-----------------------------------------------------------------------------
            // Cas.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Cas,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & compare,
                    T const & value)
                -> T
                {
                    // TODO
                    // return cl::sycl::atomic_compare_exchange_strong(atomic<T, addressSpace object, T& expected, T desired)
                }
            };
        }
    }
}

#endif
