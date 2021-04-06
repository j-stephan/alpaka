/* Copyright 2021 Bernhard Manfred Gruber

 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#pragma once

#if 0
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#        include <alpaka/mem/view/Accessor.hpp>

namespace alpaka
{
    using Image = cudaTextureObject_t;

    template<typename TElem, typename TBufferIdx, typename TAccessModes>
    struct Accessor<Image, TElem, TBufferIdx, 1, TAccessModes>
    {
        ALPAKA_FN_HOST_ACC auto operator[](Vec<DimInt<1>, TBufferIdx> i) const -> TElem
        {
            return (*this)(i[0]);
        }

        ALPAKA_FN_HOST_ACC auto operator[](TBufferIdx i) const -> TElem
        {
            return (*this)(i);
        }

        ALPAKA_FN_HOST_ACC auto operator()(TBufferIdx i) const -> TElem
        {
            return tex1Dfetch(texObj, i);
        }

        ALPAKA_FN_HOST_ACC auto operator()(float i) const -> TElem
        {
            return tex1D(texObj, i);
        }

        Image texObj;
        Vec<DimInt<1>, TBufferIdx> extents;
    };

    template<typename TElem, typename TBufferIdx, typename TAccessModes>
    struct Accessor<Image, TElem, TBufferIdx, 2, TAccessModes>
    {
        ALPAKA_FN_HOST_ACC auto operator[](Vec<DimInt<2>, TBufferIdx> i) const -> TElem
        {
            return (*this)(i[0], i[1]);
        }

        ALPAKA_FN_HOST_ACC auto operator()(TBufferIdx y, TBufferIdx x) const -> TElem
        {
            return tex1Dfetch(texObj, y * rowPitchInValues + x);
        }

        ALPAKA_FN_HOST_ACC auto operator()(float y, float x) const -> TElem
        {
            return tex2D(texObj, x, y);
        }

        Image texObj;
        TBufferIdx rowPitchInValues;
        Vec<DimInt<2>, TBufferIdx> extents;
    };

    template<typename TElem, typename TBufferIdx, typename TAccessModes>
    struct Accessor<Image, TElem, TBufferIdx, 3, TAccessModes>
    {
        ALPAKA_FN_HOST_ACC auto operator[](Vec<DimInt<3>, TBufferIdx> i) const -> TElem
        {
            return (*this)(i[0], i[1], i[2]);
        }

        ALPAKA_FN_HOST_ACC auto operator()(TBufferIdx z, TBufferIdx y, TBufferIdx x) const -> TElem
        {
            return tex1Dfetch(texObj, z * slicePitchInValues + y * rowPitchInValues + x);
        }

        ALPAKA_FN_HOST_ACC auto operator()(float z, float y, float x) const -> TElem
        {
            return tex3D(texObj, x, y, z);
        }

        Image texObj;
        TBufferIdx rowPitchInValues;
        TBufferIdx slicePitchInValues;
        Vec<DimInt<3>, TBufferIdx> extents;
    };

    namespace traits
    {
        //! Builds an accessor for alpaka images.
        template<>
        struct BuildAccessor<Image>
        {
            template<typename... TAccessModes>
            ALPAKA_FN_HOST_ACC static auto buildAccessor(Image&& image)
            {
                // TODO
            }
        };
    } // namespace traits
} // namespace alpaka
#    endif
#endif