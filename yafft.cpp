/*
 * Copyright 2009 Annan Harley
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and 
 *   limitations under the License.
 */

#include "yafft.hpp"
#include "yafft_impl.hpp"
#include "yafft_windowfn_impl.hpp"

namespace yafft  {

  template <typename FP_T>
  boost::shared_ptr<iyafft_windowfn<FP_T> > get_windowfn(const window_t wt) {

    boost::shared_ptr<iyafft_windowfn<FP_T> > ptr;

    switch (wt) {
    case HAMMING:
      ptr.reset(new hamming_windowfn<FP_T>());
      break;

    case HANN:
      ptr.reset(new hann_windowfn<FP_T>());
      break;

    case GAUSS:
      ptr.reset(new gauss_windowfn<FP_T>());
      break;

    case BLACKMAN:
      ptr.reset(new blackman_windowfn<FP_T>());
      break;
    }

    return ptr;
  }

  template iyafft_float_windowfn_ptr get_windowfn<float>(const window_t wt);

  template <typename FP_T>
  boost::shared_ptr<iyafft<FP_T> > get_fft(const size_t N) {

    typedef boost::shared_ptr<iyafft<FP_T> > iyafft_ptr;

    switch (N) {
      /*
    case 2: return iyafft_float_ptr(new typename fft_t_float<2>::t());
    case 4: return iyafft_float_ptr(new typename fft_t_float<4>::t());      
      */
    case 8: return iyafft_ptr(new typename fft_t<FP_T, 8>::pow2());
    case 16: return iyafft_ptr(new typename fft_t<FP_T, 16>::pow2());
    case 32: return iyafft_ptr(new typename fft_t<FP_T, 32>::pow2());
    case 64: return iyafft_ptr(new typename fft_t<FP_T, 64>::pow2());
    case 128: return iyafft_ptr(new typename fft_t<FP_T, 128>::pow2());
    case 256: return iyafft_ptr(new typename fft_t<FP_T, 256>::pow2());
    case 512: return iyafft_ptr(new typename fft_t<FP_T, 512>::pow2());
    case 1024: return iyafft_ptr(new typename fft_t<FP_T, 1024>::pow2());
    case 2048: return iyafft_ptr(new typename fft_t<FP_T, 2048>::pow2());
    case 4096: return iyafft_ptr(new typename fft_t<FP_T, 4096>::pow2());
    case 8192: return iyafft_ptr(new typename fft_t<FP_T, 8192>::pow2());
    case 16384: return iyafft_ptr(new typename fft_t<FP_T, 16384>::pow2());
    case 32768: return iyafft_ptr(new typename fft_t<FP_T, 32768>::pow2());
    case 65536: return iyafft_ptr(new typename fft_t<FP_T, 65536>::pow2());
    case 131072: return iyafft_ptr(new typename fft_t<FP_T, 131072>::pow2());
    case 262144: return iyafft_ptr(new typename fft_t<FP_T, 262144>::pow2());
    case 524288: return iyafft_ptr(new typename fft_t<FP_T, 524288>::pow2());
    case 1048576: return iyafft_ptr(new typename fft_t<FP_T, 1048576>::pow2());
    case 2097152: return iyafft_ptr(new typename fft_t<FP_T, 2097152>::pow2());
    default: return iyafft_ptr(new typename fft_t<FP_T, 0>::npow2(N));
    }
  }

  template boost::shared_ptr<iyafft<float> > get_fft<float>(const size_t N);
}
