/* Copyright 2019 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once
namespace alpaka
{
    namespace mem
    {
        //! The accessor specifics.
        namespace access
        {
            //! The accessor mode.
            enum class mode
            {
                read = 1024, // ensure SYCL compatibility
                write,
                read_write,
                discard_write,
                discard_read_write,
                atomic
            };

            //! The accessor target.
            enum class target
            {
                global = 2014, // ensure SYCL compatibility
                constant,
                shared
            };

            //! The accessor traits.
            namespace traits
            {
                //! The device buffer accessor trait.
                template <typename TView,
                          access::mode AccessMode,
                          access::target AccessTarget,
                          typename Sfinae = void>
                struct GetAccess;
            }

            //-----------------------------------------------------------------------------
            //! Gets the accessor of the memory view.
            //!
            //! \param view The memory view.
            //! \return The accessor.
            template<
                typename TView,
                access::mode AccessMode,
                access::target AccessTarget = access::target::global>
            ALPAKA_FN_HOST auto getAccess(
                TView const & view)
            {
                return traits::GetAccess<TView, AccessMode, AccessTarget>::getAccess(view);
            }

        }
    }
}
