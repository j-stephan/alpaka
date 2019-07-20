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
             * Unfortunately, SYCL implementations disagree on how to deduce
             * the address spaces of raw pointers. Implementations relying on
             * OpenCL 2.0 (Intel) are able to place pointers in the 'generic'
             * address space and then look up the actual address space.
             * Implementations relying on OpenCL 1.2 (ComputeCpp) can't do this
             * and therefore have to add an address space attribute (__global,
             * __local, __constant or __private) directly to the pointer type.
             * This turns a 'T*' into a '__global T*' or something, making the
             * latter a different type.
             *
             * This will cause template instantiation to break. Additionally
             * this also breaks cl::sycl::atomic because address space
             * attributes are forbidden for the type parameter. In Alpaka's case
             * the value operand will also turn into '__global T' which could
             * also cause some trouble if template instantiation worked as it
             * should.
             *
             * There is also no way to extract the __global, __local etc. from
             * the type, so we can't overload for global_ptr, local_ptr and so
             * on. Tests have shown that cl::sycl::global_ptr will happily
             * swallow a __local int at compile time (and probably break at
             * runtime). OTOH, this wouldn't be a good idea anyway, since the
             * type annotations are implementation-defined.
             *
             * Another problem is the choice of the correct address space once
             * template instantiation works. Overloading the atomic ops for
             * SYCL's global_ptr, local_ptr etc. types unfortunately doesn't
             * work, because at this stage of compilation the pointers are
             * still 'generic' and the overloads therefore ambigous.
             *
             * Because of these we currently support global atomics and
             * nothing else.
             * 
             * tl;dr: We currently only support global atomics because SYCL's
             * handling of raw pointers is limited by a) OpenCL 1.2 and b) a
             * missing way of extracting the address space from the raw pointer.
             */

            //#############################################################################
            //! The specializations to execute the requested atomic ops of the SYCL accelerator.
            // See: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
            // on how to implement everything with cl::sycl::compare_exchange_strong.

            //-----------------------------------------------------------------------------
            // Add.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                              "SYCL atomics do not support this type");
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &, 
                    T * const addr,
                    T const & value)
                -> T
                {
                    auto addr_ptr = cl::sycl::global_ptr<T>{addr};
                    if constexpr(std::is_integral_v<T>)
                    {
                        auto atomic_addr = cl::sycl::atomic<
                                            T,
                                            cl::sycl::access::address_space::global_space>{addr_ptr};

                        return cl::sycl::atomic_fetch_add(atomic_addr, value);
                    }
                    else if constexpr(std::is_same_v<T, double>)
                        return simulate_atomic_add<unsigned long long>(addr_ptr, value);
                    else if constexpr(std::is_same_v<T, float>)
                        return simulate_atomic_add<unsigned int>(addr_ptr, value);
                    else if constexpr(std::is_same_v<T, cl::sycl::half>)
                        return simulate_atomic_add<unsigned short>(addr_ptr, value);
                }

                template <class TInt>
                static auto simulate_atomic_add(cl::sycl::global_ptr<T> const addr,
                                                T const & value)
                {
                    auto raw_ptr = static_cast<T*>(addr);
                    auto int_raw_ptr = reinterpret_cast<TInt*>(raw_ptr);
                    auto int_ptr = cl::sycl::global_ptr<TInt>{int_raw_ptr};
                    auto atomic_addr = cl::sycl::atomic<
                                        TInt,
                                        cl::sycl::access::address_space::global_space>{int_ptr};
                    auto old = *int_raw_ptr;
                    auto assumed = TInt{};

                    do
                    {
                        assumed = old;
                        old = cl::sycl::atomic_compare_exchange_strong(
                                atomic_addr, assumed,
                                *(reinterpret_cast<TInt*>(value + *(reinterpret_cast<T*>(&assumed)))));
                    } while (assumed != old);

                    return *(reinterpret_cast<T*>(&old));
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
                static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                              "SYCL atomics don't support this type");
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    auto addr_ptr = cl::sycl::global_ptr<T>{addr};
                    if constexpr(std::is_integral_v<T>)
                    {
                        auto atomic_addr = cl::sycl::atomic<
                                            T,
                                            cl::sycl::access::address_space::global_space>{addr_ptr};
                        return cl::sycl::atomic_fetch_sub(atomic_addr, value);
                    }
                    else if constexpr(std::is_same_v<double, T>)
                        return simulate_atomic_sub<unsigned long long>(addr_ptr, value);
                    else if constexpr(std::is_same_v<float, T>)
                        return simulate_atomic_sub<unsigned int>(addr_ptr, value);
                    else if constexpr(std::is_same_v<cl::sycl::half, T>)
                        return simulate_atomic_sub<unsigned short>(addr_ptr, value);
                }

                template <class TInt>
                static auto simulate_atomic_sub(cl::sycl::global_ptr<T> const addr,
                                                T const & value)
                {
                    auto raw_ptr = static_cast<T*>(addr);
                    auto int_raw_ptr = reinterpret_cast<TInt*>(raw_ptr);
                    auto int_ptr = cl::sycl::global_ptr<TInt>{int_raw_ptr};
                    auto atomic_addr = cl::sycl::atomic<
                                        TInt,
                                        cl::sycl::access::address_space::global_space>{int_ptr};
                    auto old = *int_raw_ptr;
                    auto assumed = TInt{};

                    do
                    {
                        assumed = old;
                        old = cl::sycl::atomic_compare_exchange_strong(
                                atomic_addr, assumed,
                                *(reinterpret_cast<TInt*>(value - *(reinterpret_cast<T*>(&assumed)))));
                    } while (assumed != old);

                    return *(reinterpret_cast<T*>(&old));
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
                static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                              "SYCL atomics don't support this type");
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    auto addr_ptr = cl::sycl::global_ptr<T>{addr};
                    if constexpr(std::is_integral_v<T>)
                    {
                        auto atomic_addr = cl::sycl::atomic<
                                            T,
                                            cl::sycl::access::address_space::global_space>{addr_ptr};
                        return cl::sycl::atomic_fetch_min(atomic_addr, value);
                    }
                    else if constexpr(std::is_same_v<double, T>)
                        return simulate_atomic_min<unsigned long long>(addr_ptr, value);
                    else if constexpr(std::is_same_v<float, T>)
                        return simulate_atomic_min<unsigned int>(addr_ptr, value);
                    else if constexpr(std::is_same_v<cl::sycl::half, T>)
                        return simulate_atomic_min<unsigned short>(addr_ptr, value);
                }

                template <class TInt>
                static auto simulate_atomic_min(cl::sycl::global_ptr<T> const addr,
                                                T const & value)
                {
                    auto raw_ptr = static_cast<T*>(addr);
                    auto int_raw_ptr = reinterpret_cast<TInt*>(raw_ptr);
                    auto int_ptr = cl::sycl::global_ptr<TInt>{int_raw_ptr};
                    auto atomic_addr = cl::sycl::atomic<
                                        TInt,
                                        cl::sycl::access::address_space::global_space>{int_ptr};
                    auto old = *int_raw_ptr;
                    auto assumed = TInt{};

                    do
                    {
                        assumed = old;
                        auto min_val = cl::sycl::fmin(value, *(reinterpret_cast<T*>(&assumed)));
                        old = cl::sycl::atomic_compare_exchange_strong(
                                atomic_addr, assumed,
                                *(reinterpret_cast<TInt*>(&min_val)));
                    } while (assumed != old);

                    return *(reinterpret_cast<T*>(&old));
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
                static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                              "SYCL atomics don't support this type");
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    auto addr_ptr = cl::sycl::global_ptr<T>{addr};
                    if constexpr(std::is_integral_v<T>)
                    {
                        auto atomic_addr = cl::sycl::atomic<
                                            T,
                                            cl::sycl::access::address_space::global_space>{addr_ptr};
                        return cl::sycl::atomic_fetch_max(atomic_addr, value);
                    }
                    else if constexpr(std::is_same_v<double, T>)
                        return simulate_atomic_max<unsigned long long>(addr_ptr, value);
                    else if constexpr(std::is_same_v<float, T>)
                        return simulate_atomic_max<unsigned int>(addr_ptr, value);
                    else if constexpr(std::is_same_v<cl::sycl::half, T>)
                        return simulate_atomic_max<unsigned short>(addr_ptr, value);
                }

                template <class TInt>
                static auto simulate_atomic_max(cl::sycl::global_ptr<T> const addr,
                                                T const & value)
                {
                    auto raw_ptr = static_cast<T*>(addr);
                    auto int_raw_ptr = reinterpret_cast<TInt*>(raw_ptr);
                    auto int_ptr = cl::sycl::global_ptr<TInt>{int_raw_ptr};
                    auto atomic_addr = cl::sycl::atomic<
                                        TInt,
                                        cl::sycl::access::address_space::global_space>{int_ptr};
                    auto old = *int_raw_ptr;
                    auto assumed = TInt{};

                    do
                    {
                        assumed = old;
                        auto max_val = cl::sycl::fmax(value, *(reinterpret_cast<T*>(&assumed)));
                        old = cl::sycl::atomic_compare_exchange_strong(
                                atomic_addr, assumed,
                                *(reinterpret_cast<TInt*>(&max_val)));
                    } while (assumed != old);

                    return *(reinterpret_cast<T*>(&old));
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
                static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                              "SYCL atomics don't support this type");
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    auto addr_ptr = cl::sycl::global_ptr<T>{addr};
                    if constexpr(std::is_integral_v<T> || std::is_same_v<float, T>)
                    {
                        auto atomic_addr = cl::sycl::atomic<
                                            T,
                                            cl::sycl::access::address_space::global_space>{addr_ptr};
                        return cl::sycl::atomic_exchange(atomic_addr, value);
                    }
                    else if constexpr(std::is_same_v<double, T>)
                        return simulate_atomic_exchange<unsigned long long>(addr_ptr, value);
                    else if constexpr(std::is_same_v<cl::sycl::half, T>)
                        return simulate_atomic_exchange<unsigned short>(addr_ptr, value);
                }

                template <class TInt>
                static auto simulate_atomic_exchange(cl::sycl::global_ptr<T> const addr,
                                                     T const & value)
                {
                    auto raw_ptr = static_cast<T*>(addr);
                    auto int_raw_ptr = reinterpret_cast<TInt*>(raw_ptr);
                    auto int_ptr = cl::sycl::global_ptr<TInt>{int_raw_ptr};
                    auto atomic_addr = cl::sycl::atomic<
                                        TInt,
                                        cl::sycl::access::address_space::global_space>{int_ptr};
                    auto old = *int_raw_ptr;
                    auto assumed = TInt{};

                    do
                    {
                        assumed = old;
                        auto val = value;
                        old = cl::sycl::atomic_compare_exchange_strong(
                                atomic_addr, assumed,
                                *(reinterpret_cast<TInt*>(&val)));
                    } while (assumed != old);

                    return *(reinterpret_cast<T*>(&old));
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
                static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                              "SYCL atomics don't support this type");
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    using TInt = std::conditional_t<std::is_integral_v<T>, T,
                                    std::conditional_t<std::is_same_v<double, T>, unsigned long long,
                                    std::conditional_t<std::is_same_v<float, T>, unsigned int,
                                    std::conditional_t<std::is_same_v<cl::sycl::half, T>, unsigned short, void>>>>;

                    auto int_raw_ptr = reinterpret_cast<TInt*>(addr);
                    auto int_ptr = cl::sycl::global_ptr<TInt>{int_raw_ptr};
                    auto atomic_addr = cl::sycl::atomic<
                                        TInt,
                                        cl::sycl::access::address_space::global_space>{int_ptr};
                    auto old = *int_raw_ptr;
                    auto assumed = TInt{};

                    do
                    {
                        assumed = old;
                        auto old_val = *(reinterpret_cast<T*>(&assumed));
                        auto new_val = (old_val >= value) ? 0 : (old_val + 1);
                        old = cl::sycl::atomic_compare_exchange_strong(
                                atomic_addr, assumed,
                                *(reinterpret_cast<TInt*>(&new_val)));
                    } while (assumed != old);

                    return *(reinterpret_cast<T*>(&old));
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
                static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                              "SYCL atomics don't support this type");
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    using TInt = std::conditional_t<std::is_integral_v<T>, T,
                                    std::conditional_t<std::is_same_v<double, T>, unsigned long long,
                                    std::conditional_t<std::is_same_v<float, T>, unsigned int,
                                    std::conditional_t<std::is_same_v<cl::sycl::half, T>, unsigned short, void>>>>;

                    auto int_raw_ptr = reinterpret_cast<TInt*>(addr);
                    auto int_ptr = cl::sycl::global_ptr<TInt>{int_raw_ptr};
                    auto atomic_addr = cl::sycl::atomic<
                                        TInt,
                                        cl::sycl::access::address_space::global_space>{int_ptr};
                    auto old = *int_raw_ptr;
                    auto assumed = TInt{};

                    do
                    {
                        assumed = old;
                        auto old_val = *(reinterpret_cast<T*>(&assumed));
                        auto new_val = ((old_val == 0) || (old_val > value)) ? value : (old_val - 1);
                        old = cl::sycl::atomic_compare_exchange_strong(
                                atomic_addr, assumed,
                                *(reinterpret_cast<TInt*>(&new_val)));
                    } while (assumed != old);

                    return *(reinterpret_cast<T*>(&old));
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
                static_assert(std::is_integral_v<T>,
                              "Bitwise operations are unsupported on non-integral types.");
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    auto addr_ptr = cl::sycl::global_ptr<T>{addr};
                    auto atomic_addr = cl::sycl::atomic<T>{addr_ptr};
                    return cl::sycl::atomic_fetch_and(atomic_addr, value);
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
                static_assert(std::is_integral_v<T>,
                              "Bitwise operations are unsupported on non-integral types.");
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    auto addr_ptr = cl::sycl::global_ptr<T>{addr};
                    auto atomic_addr = cl::sycl::atomic<T>{addr_ptr};
                    return cl::sycl::atomic_fetch_or(atomic_addr, value);
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
                static_assert(std::is_integral_v<T>,
                              "Bitwise operations are unsupported on non-integral types.");
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    auto addr_ptr = cl::sycl::global_ptr<T>{addr};
                    auto atomic_addr = cl::sycl::atomic<T>{addr_ptr};
                    return cl::sycl::atomic_fetch_xor(atomic_addr, value);
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
                static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                              "SYCL atomics don't support this type");
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & compare,
                    T const & value)
                -> T
                {
                    auto addr_ptr = cl::sycl::global_ptr<T>{addr};
                    if constexpr(std::is_integral_v<T>)
                    {
                        auto atomic_addr = cl::sycl::atomic<T>{addr_ptr};
                        return cl::sycl::atomic_compare_exchange_strong(atomic_addr,
                                                                        compare,
                                                                        value);
                    }
                    else if constexpr(std::is_same_v<double, T>)
                        return simulate_atomic_compare_exchange<unsigned long long>(addr, compare, value);
                    else if constexpr(std::is_same_v<float, T>)
                        return simulate_atomic_compare_exchange<unsigned int>(addr, compare, value);
                    else if constexpr(std::is_same_v<cl::sycl::half, T>)
                        return simulate_atomic_compare_exchange<unsigned short>(addr, compare, value);
                }

                template <class TInt>
                static auto simulate_atomic_compare_exchange(
                        cl::sycl::global_ptr<T> const addr,
                        T const & compare,
                        T const & value)
                {
                    auto raw_ptr = static_cast<T*>(addr);
                    auto int_raw_ptr = reinterpret_cast<TInt*>(raw_ptr);
                    auto int_ptr = cl::sycl::global_ptr<TInt>{int_raw_ptr};
                    auto atomic_addr = cl::sycl::atomic<
                                        TInt,
                                        cl::sycl::access::address_space::global_space>{int_ptr};
                    auto old = *int_raw_ptr;
                    auto assumed = TInt{};

                    do
                    {
                        assumed = old;
                        auto old_val = *(reinterpret_cast<T*>(&assumed));
                        auto new_val = (old_val == compare) ? value : old_val;
                        old = cl::sycl::atomic_compare_exchange_strong(
                                atomic_addr, assumed,
                                *(reinterpret_cast<TInt*>(&new_val)));
                    } while (assumed != old);

                    return *(reinterpret_cast<T*>(&old));
                }
            };
        }
    }
}

#endif
