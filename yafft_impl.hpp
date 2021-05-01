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

#ifndef _YAFFT_IMPL_HPP_
#define _YAFFT_IMPL_HPP_

#include <complex>

#include "yafft.hpp"

namespace yafft {
  namespace impl {
     
    // as noted in the other implementation, there are several stages
    // in how i choose to do an fft
    //
    // - swizzling into the correct format
    // - doing stage 1, which is just an add/sub phase, as the twiddle is 1.0f
    // - doing stage 2, which is just add/sub for the front half of the data,
    //    and sub/add for the second half, as the twiddle there is -1.0f
    // - stage 3 is the meat of the operation, we iterate down the twiddle set
    //    as we loop over the data (in place operations on the work array)
    // - stage 4 only exists for floats - we need to do some simd shuffling
    //    to get at some of the elements.  more twiddle op/mults
    // - stage 5 is a bit more simd shuffling and the final multiplies
    // - the unswizzling
    //
    // I should say that I thought about ways that I might be able to put the
    //  unswizzling operations in stages 4/5 so that we didn't need to have
    //  an explicit unswizzle step, but I couldn't figure out any way to do it
    //  without destroying the good cache behaviour that the code 
    //  currently has
    
    // this is the bottom of the tree, where we're just doing 8 complex
    // numbers, arranged [r,r,r,r,i,i,i,i,r,r,r,r,i,i,i,i]
    template<class SIMD_OPS,
	     class PRINT_OPS,
	     size_t N_stop>
    struct fft_bottom {
      
      typedef typename SIMD_OPS::SIMD_T simd_t;
      typedef typename SIMD_OPS::VAL_T val_t;
      
      static inline void
      fn(val_t * const in,
	 val_t * const twiddles,
	 const size_t level_offset,
	 const size_t twiddle_offset,
	 simd_t r1,
	 simd_t i1,
	 simd_t r2,
	 simd_t i2) __attribute__((always_inline)) {

	const size_t im = N_stop >> 1;
	size_t twiddles_base = (twiddle_offset * 2) + level_offset;

	val_t * const in_p_im = in + im;
	val_t * const in_p_N_stop = in + N_stop;
	val_t * const in_p_N_stop_p_im = in_p_N_stop + im;

	PRINT_OPS::print_simd_cpx("stage4 preshuffle 1", r1, i1);
	PRINT_OPS::print_simd_cpx("stage4 preshuffle 2", r2, i2);

	simd_t r3 = SIMD_OPS::shuf_3_2_3_2(r1, r2);
	simd_t i3 = SIMD_OPS::shuf_3_2_3_2(i1, i2);

	PRINT_OPS::print_simd_cpx("stage4 shuffle 1", r3, i3);
	
	simd_t r4 = SIMD_OPS::shuf_1_0_1_0(r1, r2);
	simd_t i4 = SIMD_OPS::shuf_1_0_1_0(i1, i2);

	PRINT_OPS::print_simd_cpx("stage4 shuffle 2", r4, i4);
	
	simd_t rt1 = SIMD_OPS::load(twiddles+twiddles_base);
	simd_t it1 = SIMD_OPS::load(twiddles+twiddles_base+im);
	twiddles_base += N_stop;

	PRINT_OPS::print_simd_cpx("stage4 twiddles", rt1, it1);
	
	simd_t r5 = SIMD_OPS::sub(SIMD_OPS::mul(r3, rt1),
				  SIMD_OPS::mul(i3, it1));
	simd_t i5 = SIMD_OPS::add(SIMD_OPS::mul(r3, it1),
				  SIMD_OPS::mul(i3, rt1));

	simd_t r6 = SIMD_OPS::add(r4, r5);
	simd_t i6 = SIMD_OPS::add(i4, i5);

	PRINT_OPS::print_simd_cpx("stage4 out 1", r6, i6);
	
	simd_t r7 = SIMD_OPS::sub(r4, r5);
	simd_t i7 = SIMD_OPS::sub(i4, i5);

	PRINT_OPS::print_simd_cpx("stage4 out 2", r7, i7);
	
	// next stage
	simd_t r8_tmp = SIMD_OPS::shuf_3_1_3_1(r6, r7);
	simd_t i8_tmp = SIMD_OPS::shuf_3_1_3_1(i6, i7);

	simd_t r8 = SIMD_OPS::shuf_3_1_2_0(r8_tmp, r8_tmp);
	simd_t i8 = SIMD_OPS::shuf_3_1_2_0(i8_tmp, i8_tmp);

	PRINT_OPS::print_simd_cpx("stage5 shuffle 1", r8, i8);
	
	simd_t r9_tmp = SIMD_OPS::shuf_2_0_2_0(r6, r7);
	simd_t i9_tmp = SIMD_OPS::shuf_2_0_2_0(i6, i7);

	simd_t r9 = SIMD_OPS::shuf_3_1_2_0(r9_tmp, r9_tmp);
	simd_t i9 = SIMD_OPS::shuf_3_1_2_0(i9_tmp, i9_tmp);

	PRINT_OPS::print_simd_cpx("stage5 shuffle 2", r9, i9);
	
	simd_t rt2 = SIMD_OPS::load(twiddles+twiddles_base);
	simd_t it2 = SIMD_OPS::load(twiddles+twiddles_base+im);
	
	PRINT_OPS::print_simd_cpx("stage5 twiddles", rt2, it2);
	
	simd_t r10 = SIMD_OPS::sub(SIMD_OPS::mul(r8, rt2),
				   SIMD_OPS::mul(i8, it2));
	simd_t i10 = SIMD_OPS::add(SIMD_OPS::mul(r8, it2),
 				   SIMD_OPS::mul(i8, rt2));

	simd_t or1 = SIMD_OPS::add(r9, r10);
	simd_t oi1 = SIMD_OPS::add(i9, i10);

	SIMD_OPS::store(in, or1);
	SIMD_OPS::store(in_p_im, oi1);

	PRINT_OPS::print_val("stage5 out 1-1", in);
	PRINT_OPS::print_val("stage5 out 1-2", in_p_im);
	
	simd_t or2 = SIMD_OPS::sub(r9, r10);
	simd_t oi2 = SIMD_OPS::sub(i9, i10);

	SIMD_OPS::store(in_p_N_stop, or2);
	SIMD_OPS::store(in_p_N_stop_p_im, oi2);	

	PRINT_OPS::print_val("stage5 out 2-1", in_p_N_stop);	
	PRINT_OPS::print_val("stage5 out 2-2", in_p_N_stop_p_im);	
      }
    };

    // the generic third stage loop
    template<class SIMD_OPS,
	     class PRINT_OPS,
	     size_t N,
	     size_t N_stop,
	     size_t STOP,
	     size_t SPECIAL_TWIDDLES>
    struct fft_coreloop {
      
      typedef typename SIMD_OPS::SIMD_T simd_t;
      typedef typename SIMD_OPS::VAL_T val_t;
      
      static inline void
      fn(val_t * const in,
	 val_t * const twiddles,
	 const size_t level_offset,
	 const size_t twiddle_offset) {
	
	const size_t im = N_stop >> 1;
	val_t * const twiddles_tmp = twiddles + twiddle_offset;
	
	simd_t rt_xmm = SIMD_OPS::load(twiddles_tmp);
	simd_t it_xmm = SIMD_OPS::load(twiddles_tmp + im);	
	
	val_t * const in_p_im = in + im;
	
	val_t * const in_p_N = in + N;
	val_t * const in_p_N_p_im = in_p_N + im;
	  
	for (size_t i = 0; i < N; i += N_stop) {

	  simd_t r1_xmm = SIMD_OPS::load(in_p_N + i);
	  simd_t i1_xmm = SIMD_OPS::load(in_p_N_p_im + i);
	  
	  PRINT_OPS::print_simd_cpx("fftcoreloop back in",
				    r1_xmm, 
				    i1_xmm);
	  
	  PRINT_OPS::print_simd_cpx("fftcoreloop twiddles",
				    rt_xmm, 
				    it_xmm);
	  
	  simd_t r2_xmm = SIMD_OPS::sub(_mm_mul_ps(r1_xmm, rt_xmm),
					_mm_mul_ps(i1_xmm, it_xmm));
	  simd_t i2_xmm = SIMD_OPS::add(_mm_mul_ps(r1_xmm, it_xmm),
					_mm_mul_ps(i1_xmm, rt_xmm));
	  
	  simd_t r3_xmm = SIMD_OPS::load(in + i);
	  simd_t i3_xmm = SIMD_OPS::load(in_p_im + i);
	  
	  PRINT_OPS::print_simd_cpx("fftcoreloop front in",
				    r3_xmm, 
				    i3_xmm);
	  
	  SIMD_OPS::store(in + i, SIMD_OPS::add(r3_xmm, r2_xmm));
	  SIMD_OPS::store(in_p_im + i, SIMD_OPS::add(i3_xmm, i2_xmm));
	  
	  PRINT_OPS::print_val_cpx("fftcoreloop front out",
				   in + i, 
				   in_p_im + i);
	  
	  SIMD_OPS::store(in_p_N + i, SIMD_OPS::sub(r3_xmm, r2_xmm));
	  SIMD_OPS::store(in_p_N_p_im + i, SIMD_OPS::sub(i3_xmm, i2_xmm));
	  
	  PRINT_OPS::print_val_cpx("fftcoreloop back out",
				   in_p_N + i, 
				   in_p_N_p_im + i);
	  
	}
	
	const size_t next_twiddle_offset = twiddle_offset << 1;
	
	fft_coreloop<SIMD_OPS,
	  PRINT_OPS,
	  (N >> 1), 
	  N_stop,
	  N_stop == (N >> 1),
	  0>::fn(in,
		 twiddles,
		 level_offset,
		 next_twiddle_offset);
      
	fft_coreloop<SIMD_OPS,
	  PRINT_OPS,
	  (N >> 1), 
	  N_stop,
	  N_stop == (N >> 1),
	  0>::fn(in + N,
		 twiddles,
		 level_offset,
		 next_twiddle_offset + N_stop);
      }
    };

    // generic stop case    
    template<class SIMD_OPS,
	     class PRINT_OPS,
	     size_t N,
	     size_t N_stop,
	     size_t SPECIAL_TWIDDLES>
    struct fft_coreloop<SIMD_OPS,
			PRINT_OPS,
			N,
			N_stop,
			1,
			SPECIAL_TWIDDLES> {
      
      typedef typename SIMD_OPS::SIMD_T simd_t;
      typedef typename SIMD_OPS::VAL_T val_t;
      
      static inline void
      fn(val_t * const in,
	 val_t * const twiddles,
	 const size_t level_offset,
	 const size_t twiddle_offset)  __attribute__((always_inline)) {
	
	const size_t im = N_stop >> 1;
	val_t * const twiddles_tmp = twiddles + twiddle_offset;
	
	simd_t rt_xmm = SIMD_OPS::load(twiddles_tmp);
	simd_t it_xmm = SIMD_OPS::load(twiddles_tmp + im);	
	
	val_t * const in_p_im = in + im;
	
	val_t * const in_p_N_stop = in + N_stop;
	val_t * const in_p_N_stop_p_im = in_p_N_stop + im;
	  
	simd_t r1_xmm = SIMD_OPS::load(in_p_N_stop);
	simd_t i1_xmm = SIMD_OPS::load(in_p_N_stop_p_im);
	  
	PRINT_OPS::print_simd_cpx("fftcoreloop back in",
				  r1_xmm, 
				  i1_xmm);
	
	PRINT_OPS::print_simd_cpx("fftcoreloop twiddles",
				  rt_xmm, 
				  it_xmm);
	
	simd_t r2_xmm = SIMD_OPS::sub(_mm_mul_ps(r1_xmm, rt_xmm),
				      _mm_mul_ps(i1_xmm, it_xmm));
	simd_t i2_xmm = SIMD_OPS::add(_mm_mul_ps(r1_xmm, it_xmm),
				      _mm_mul_ps(i1_xmm, rt_xmm));
	
	simd_t r3_xmm = SIMD_OPS::load(in);
	simd_t i3_xmm = SIMD_OPS::load(in_p_im);
	  
	PRINT_OPS::print_simd_cpx("fftcoreloop front in",
				  r3_xmm, 
				  i3_xmm);

	fft_bottom<SIMD_OPS, 
	  PRINT_OPS, 
	  N_stop>::fn(in,
		      twiddles,
		      level_offset,
		      twiddle_offset,
		      SIMD_OPS::add(r3_xmm, r2_xmm),
		      SIMD_OPS::add(i3_xmm, i2_xmm),
		      SIMD_OPS::sub(r3_xmm, r2_xmm),
		      SIMD_OPS::sub(i3_xmm, i2_xmm));
      }
    };

    // the add loop
    template<class SIMD_OPS,
	     class PRINT_OPS,
	     size_t N,
	     size_t N_stop>
    struct fft_coreloop<SIMD_OPS,
			PRINT_OPS,
			N,
			N_stop,
			0,
			2> {
      
      typedef typename SIMD_OPS::SIMD_T simd_t;
      typedef typename SIMD_OPS::VAL_T val_t;
      
      static inline void
      fn(val_t * const in,
	 val_t * const twiddles,
	 const size_t level_offset,
	 const size_t twiddle_offset) {
	
	const size_t im = N_stop >> 1;
	
	val_t * const in_p_im = in + im;

	val_t * const in_p_N = in + N;
	val_t * const in_p_N_p_im = in_p_N + im;

	for (size_t i = 0; i < N; i += N_stop) {
	  
	  simd_t rin1 = SIMD_OPS::load(in + i);
	  simd_t iin1 = SIMD_OPS::load(in_p_im + i);
	  
	  PRINT_OPS::print_simd_cpx("fftcoreloop add front in",
				    rin1, 
				    iin1);

	  simd_t rin2 = SIMD_OPS::load(in_p_N + i);
	  simd_t iin2 = SIMD_OPS::load(in_p_N_p_im + i);
	  
	  PRINT_OPS::print_simd_cpx("fftcoreloop add back in",
				    rin2, 
				    iin2);

	  SIMD_OPS::store(in + i, SIMD_OPS::add(rin1, rin2));
	  SIMD_OPS::store(in_p_im + i, SIMD_OPS::add(iin1, iin2));

	  PRINT_OPS::print_val_cpx("fftcoreloop add front out",
				   in + i, 
				   in_p_im + i);

	  SIMD_OPS::store(in_p_N + i, SIMD_OPS::sub(rin1, rin2));
	  SIMD_OPS::store(in_p_N_p_im + i, SIMD_OPS::sub(iin1, iin2));

	  PRINT_OPS::print_val_cpx("fftcoreloop add back out",
				   in_p_N + i, 
				   in_p_N_p_im + i);
	}
	
	const size_t next_twiddle_offset = twiddle_offset << 1;
	
	fft_coreloop<SIMD_OPS,
	  PRINT_OPS,
	  (N >> 1), 
	  N_stop,
	  N_stop == (N >> 1), 
	  2>::fn(in,
		 twiddles,
		 level_offset,
		 next_twiddle_offset);
	
	fft_coreloop<SIMD_OPS,
	  PRINT_OPS,
	  (N >> 1), 
	  N_stop,
	  N_stop == (N >> 1), 
	  1>::fn(in + N,
		 twiddles,
		 level_offset,
		 next_twiddle_offset + N_stop);
      }
    };

    // the add loop's stop case
    template<class SIMD_OPS,
	     class PRINT_OPS,
	     size_t N,
	     size_t N_stop>
    struct fft_coreloop<SIMD_OPS,
			PRINT_OPS,
			N,
			N_stop,
			1,
			2> {
      
      typedef typename SIMD_OPS::SIMD_T simd_t;
      typedef typename SIMD_OPS::VAL_T val_t;
      
      static inline void
      fn(val_t * const in,
	 val_t * const twiddles,
	 const size_t level_offset,
	 const size_t twiddle_offset)  __attribute__((always_inline)) {
	
	const size_t im = N_stop >> 1;
	
	val_t * const in_p_im = in + im;

	val_t * const in_p_N_stop = in + N_stop;
	val_t * const in_p_N_stop_p_im = in_p_N_stop + im;

	simd_t rin1 = SIMD_OPS::load(in);
	simd_t iin1 = SIMD_OPS::load(in_p_im);
	  
	PRINT_OPS::print_simd_cpx("fftcoreloop stop add front in",
				  rin1, 
				  iin1);
	
	simd_t rin2 = SIMD_OPS::load(in_p_N_stop);
	simd_t iin2 = SIMD_OPS::load(in_p_N_stop_p_im);
	
	PRINT_OPS::print_simd_cpx("fftcoreloop stop add back in",
				  rin2, 
				  iin2);
 
	fft_bottom<SIMD_OPS, 
	  PRINT_OPS,
	  N_stop>::fn(in,
		      twiddles,
		      level_offset,
		      twiddle_offset,
		      SIMD_OPS::add(rin1, rin2),
		      SIMD_OPS::add(iin1, iin2),
		      SIMD_OPS::sub(rin1, rin2),
		      SIMD_OPS::sub(iin1, iin2));

      }
    };

    // the subtraction branch
    template<class SIMD_OPS,
	     class PRINT_OPS,
	     size_t N,
	     size_t N_stop>
    struct fft_coreloop<SIMD_OPS,
			PRINT_OPS,
			N,
			N_stop,
			0,
			1>  {
      
      typedef typename SIMD_OPS::SIMD_T simd_t;
      typedef typename SIMD_OPS::VAL_T val_t;
      
      static inline void
      fn(val_t * const in,
	 val_t * const twiddles,
	 const size_t level_offset,
	 const size_t twiddle_offset) {
	
	const size_t im = N_stop >> 1;
	
	val_t * const in_p_im = in + im;

	val_t * const in_p_N = in + N;
	val_t * const in_p_N_p_im = in_p_N + im;

	for (size_t i = 0; i < N; i += N_stop) {
	  
	  simd_t rin1 = SIMD_OPS::load(in + i);
	  simd_t iin1 = SIMD_OPS::load(in_p_im + i);
	  
	  PRINT_OPS::print_simd_cpx("fftcoreloop sub front in",
				    rin1, 
				    iin1);

	  simd_t rin2 = SIMD_OPS::load(in_p_N + i);
	  simd_t iin2 = SIMD_OPS::load(in_p_N_p_im + i);
	  
	  PRINT_OPS::print_simd_cpx("fftcoreloop sub back in",
				    rin2, 
				    iin2);

	  SIMD_OPS::store(in + i, SIMD_OPS::add(rin1, iin2));
	  SIMD_OPS::store(in_p_im + i, SIMD_OPS::sub(iin1, rin2));

	  PRINT_OPS::print_val_cpx("fftcoreloop sub front out",
				   in + i, 
				   in_p_im + i);

	  SIMD_OPS::store(in_p_N + i, SIMD_OPS::sub(rin1, iin2));
	  SIMD_OPS::store(in_p_N_p_im + i, SIMD_OPS::add(iin1, rin2));

	  PRINT_OPS::print_val_cpx("fftcoreloop sub back out",
				   in_p_N + i, 
				   in_p_N_p_im + i);
	}
	
	const size_t next_twiddle_offset = twiddle_offset << 1;
	
	fft_coreloop<SIMD_OPS,
	  PRINT_OPS,
	  (N >> 1), 
	  N_stop,
	  N_stop == (N >> 1), 
	  0>::fn(in,
		 twiddles,
		 level_offset,
		 next_twiddle_offset);
      
	fft_coreloop<SIMD_OPS,
	  PRINT_OPS,
	  (N >> 1), 
	  N_stop,
	  N_stop == (N >> 1), 
	  0>::fn(in + N,
		 twiddles,
		 level_offset,
		 next_twiddle_offset + N_stop);
      }
    };
        
    // the sub loop's stop case
    template<class SIMD_OPS,
	     class PRINT_OPS,
	     size_t N,
	     size_t N_stop>
    struct fft_coreloop<SIMD_OPS,
			PRINT_OPS,
			N,
			N_stop,
			1,
			1> {
      
      typedef typename SIMD_OPS::SIMD_T simd_t;
      typedef typename SIMD_OPS::VAL_T val_t;
      
      static inline void
      fn(val_t * const in,
	 val_t * const twiddles,
	 const size_t level_offset,
	 const size_t twiddle_offset)  __attribute__((always_inline)) {
	
	const size_t im = N_stop >> 1;
	
	val_t * const in_p_im = in + im;

	val_t * const in_p_N_stop = in + N_stop;
	val_t * const in_p_N_stop_p_im = in_p_N_stop + im;

	simd_t rin1 = SIMD_OPS::load(in);
	simd_t iin1 = SIMD_OPS::load(in_p_im);
	  
	PRINT_OPS::print_simd_cpx("fftcoreloop stop sub front in",
				  rin1, 
				  iin1);
	
	simd_t rin2 = SIMD_OPS::load(in_p_N_stop);
	simd_t iin2 = SIMD_OPS::load(in_p_N_stop_p_im);
	
	PRINT_OPS::print_simd_cpx("fftcoreloop stop sub back in",
				  rin2, 
				  iin2);

	fft_bottom<SIMD_OPS, 
	  PRINT_OPS,
	  N_stop>::fn(in,
		      twiddles,
		      level_offset,
		      twiddle_offset,
		      SIMD_OPS::add(rin1, iin2),
		      SIMD_OPS::sub(iin1, rin2),
		      SIMD_OPS::sub(rin1, iin2),
		      SIMD_OPS::add(iin1, rin2));
      }
    };

    // these two front-ends should be sufficient
    // as far as the float vs double thing goes,
    // the difference should be that the major loop stage
    // goes one iteration longer for doubles, as we
    // need to do one less stage of simd element shuffling
    // (2 64bit elements instead of 4 32bit) for the final
    // leg - this hint will be set as an enum in SIMD_OPS 

    // I used to have a separate unswizzling stage, but realized
    // during the cache testing that it was silly to do so considering
    // how simple the first stage of an FFT is
    
    template <class SIMD_OPS,
	      class PRINT_OPS,
	      class STDCPX_OPS,
	      class DIRECTION_OPS,
	      template <class _IN_T, 
			class _OUT_T, 
			class _PRINT_OPS, 
			size_t _N, 
			class _DIRECTION_OPS,
			yafft::mult_t M_T> class UNSWIZZLE_OPS,
	      size_t N,
	      size_t N_stop>
    struct std_complex_fft {

      typedef typename SIMD_OPS::SIMD_T simd_t;
      typedef typename SIMD_OPS::VAL_T val_t;

      static inline void
      fn(std::complex<val_t> * const in_cpx,
	 val_t * const work,
	 val_t * const twiddles,
	 std::complex<val_t> * const out) {
	
	val_t * const in = (val_t*) in_cpx;
	const size_t im = N_stop >> 1;

	simd_t bitflips = SIMD_OPS::load_bitflip1();

	val_t * const in_p_im = in + im;

	val_t * const in_p_N = in + N;
	val_t * const in_p_N_p_im = in_p_N + im;

	val_t * const work_p_im = work + im;

	val_t * const work_p_N = work + N;
	val_t * const work_p_N_p_im = work_p_N + im;

	for (size_t i = 0; i < N; i += N_stop) {

	  simd_t in1 = SIMD_OPS::load(in + i);
	  simd_t in2 = SIMD_OPS::load(in_p_im + i);
	  
	  PRINT_OPS::print_simd("stage1 A in1", in1); 
	  PRINT_OPS::print_simd("stage1 A in2", in2); 

	  simd_t in3 = SIMD_OPS::load(in_p_N + i);
	  simd_t in4 = SIMD_OPS::load(in_p_N_p_im + i);
	  
	  PRINT_OPS::print_simd("stage1 B in1",  in3);
	  PRINT_OPS::print_simd("stage1 B in2",  in4);

	  simd_t tmp1, tmp2;
	  
	  tmp1 = STDCPX_OPS::real(in1, in2);
	  tmp2 = STDCPX_OPS::real(in3, in4);

	  PRINT_OPS::print_simd("stage1 real tmp1", tmp1);
	  PRINT_OPS::print_simd("stage1 real tmp2", tmp2);

	  SIMD_OPS::store(work + i, SIMD_OPS::add(tmp1, tmp2));
	  SIMD_OPS::store(work_p_N + i, SIMD_OPS::sub(tmp1, tmp2));
	  
	  PRINT_OPS::print_val("stage1 real out1", work + i);
	  PRINT_OPS::print_val("stage1 real out2", work_p_N + i);

	  tmp1 = DIRECTION_OPS::bitflip(STDCPX_OPS::imag(in1, in2), bitflips);
	  tmp2 = DIRECTION_OPS::bitflip(STDCPX_OPS::imag(in3, in4), bitflips);

	  PRINT_OPS::print_simd("stage1 imag tmp1", tmp1);
	  PRINT_OPS::print_simd("stage1 imag tmp2", tmp2);

	  SIMD_OPS::store(work_p_im + i, SIMD_OPS::add(tmp1, tmp2));
	  SIMD_OPS::store(work_p_N_p_im + i, SIMD_OPS::sub(tmp1, tmp2));

	  PRINT_OPS::print_val("stage1 imag out1", work_p_im + i); 
	  PRINT_OPS::print_val("stage1 imag out2", work_p_N_p_im + i);
	}

	PRINT_OPS::print_val_cpx_arr("stage1 output",
				     work,
				     N << 1);

	fft_coreloop<SIMD_OPS,
	  PRINT_OPS,
	  (N >> 1), 
	  N_stop,
	  (N_stop == (N >> 1)), 
	  2>::fn(work,
		 twiddles,
		 ((N == (N_stop << 1)) ? 0 : N),
		 0);
	
	fft_coreloop<SIMD_OPS,
	  PRINT_OPS,
	  (N >> 1), 
	  N_stop,
	  (N_stop == (N >> 1)), 
	  1>::fn(work + N,
		 twiddles,
		 ((N == (N_stop << 1)) ? 0 : N),
		 N_stop);
	
	PRINT_OPS::print_std_cpx("final work output",
				 reinterpret_cast<std::complex<val_t>*>(work),
				 N);

	UNSWIZZLE_OPS<val_t, 
	  val_t, 
	  PRINT_OPS, 
	  N,
	  DIRECTION_OPS,
	  yafft::NONE>::fn(work,
			   reinterpret_cast<val_t*>(out));
      }
    };

    template <class SIMD_OPS,
	      class PRINT_OPS,
	      class STDCPX_OPS,
	      class DIRECTION_OPS,
	      size_t N,
	      size_t N_stop>
    struct std_complex_fft_mul_nounswizzle {

      typedef typename SIMD_OPS::SIMD_T simd_t;
      typedef typename SIMD_OPS::VAL_T val_t;

      static inline void
      fn(std::complex<val_t> * const in_cpx,
	 val_t * const work,
	 val_t * const twiddles,
	 val_t * const mul) {
	
	val_t * const in = (val_t*) in_cpx;
	const size_t im = N_stop >> 1;

	simd_t bitflips = SIMD_OPS::load_bitflip1();

	val_t * const in_p_im = in + im;

	val_t * const in_p_N = in + N;
	val_t * const in_p_N_p_im = in_p_N + im;

	val_t * const mul_p_im = mul + im;
	
	val_t * const mul_p_N = mul + N;
	val_t * const mul_p_N_p_im = mul_p_N + im;

	val_t * const work_p_im = work + im;

	val_t * const work_p_N = work + N;
	val_t * const work_p_N_p_im = work_p_N + im;

	for (size_t i = 0; i < N; i += N_stop) {

	  simd_t in1 = SIMD_OPS::load(in + i);
	  simd_t in2 = SIMD_OPS::load(in_p_im + i);
	  
	  PRINT_OPS::print_simd("stage1 A in1", in1); 
	  PRINT_OPS::print_simd("stage1 A in2", in2); 

	  simd_t in3 = SIMD_OPS::load(in_p_N + i);
	  simd_t in4 = SIMD_OPS::load(in_p_N_p_im + i);
	  
	  PRINT_OPS::print_simd("stage1 B in1",  in3);
	  PRINT_OPS::print_simd("stage1 B in2",  in4);

	  simd_t rout1_tmp = STDCPX_OPS::real(in1, in2);
	  simd_t iout1_tmp = DIRECTION_OPS::bitflip(STDCPX_OPS::imag(in1, in2), 
						    bitflips);

	  PRINT_OPS::print_simd("stage1 real rout1_tmp", rout1_tmp);
	  PRINT_OPS::print_simd("stage1 imag iout1_tmp", iout1_tmp);

	  simd_t rmul1 = SIMD_OPS::load(mul + i);
	  simd_t imul1 = SIMD_OPS::load(mul_p_im + i);

	  PRINT_OPS::print_simd("stage1 rmul1", rmul1);
	  PRINT_OPS::print_simd("stage1 imul1", imul1);

	  simd_t rout1 = SIMD_OPS::sub(SIMD_OPS::mul(rout1_tmp, rmul1),
				       SIMD_OPS::mul(iout1_tmp, imul1));

	  simd_t iout1 = SIMD_OPS::add(SIMD_OPS::mul(rout1_tmp, imul1),
				       SIMD_OPS::mul(iout1_tmp, rmul1));

	  PRINT_OPS::print_simd("stage1 real rout1", rout1);
	  PRINT_OPS::print_simd("stage1 imag iout1", iout1);

	  simd_t rout2_tmp = STDCPX_OPS::real(in3, in4);
	  simd_t iout2_tmp = DIRECTION_OPS::bitflip(STDCPX_OPS::imag(in3, in4),
						    bitflips);

	  PRINT_OPS::print_simd("stage1 real rout2_tmp", rout2_tmp);
	  PRINT_OPS::print_simd("stage1 imag iout2_tmp", iout2_tmp);

	  simd_t rmul2 = SIMD_OPS::load(mul_p_N + i);
	  simd_t imul2 = SIMD_OPS::load(mul_p_N_p_im + i);

	  PRINT_OPS::print_simd("stage1 rmul2", rmul2);
	  PRINT_OPS::print_simd("stage1 imul2", imul2);

	  simd_t rout2 = SIMD_OPS::sub(SIMD_OPS::mul(rout2_tmp, rmul2),
				       SIMD_OPS::mul(iout2_tmp, imul2));

	  simd_t iout2 = SIMD_OPS::add(SIMD_OPS::mul(rout2_tmp, imul2),
				       SIMD_OPS::mul(iout2_tmp, rmul2));

	  PRINT_OPS::print_simd("stage1 real rout2", rout2);
	  PRINT_OPS::print_simd("stage1 imag iout2", iout2);

	  SIMD_OPS::store(work + i, SIMD_OPS::add(rout1, rout2));
	  PRINT_OPS::print_val("stage1 real out1", work + i);

	  SIMD_OPS::store(work_p_im + i, SIMD_OPS::add(iout1, iout2));
	  PRINT_OPS::print_val("stage1 imag out1", work_p_im + i); 
 
	  SIMD_OPS::store(work_p_N + i, SIMD_OPS::sub(rout1, rout2));	  
	  PRINT_OPS::print_val("stage1 real out2", work_p_N + i);

	  SIMD_OPS::store(work_p_N_p_im + i, SIMD_OPS::sub(iout1, iout2));
	  PRINT_OPS::print_val("stage1 imag out2", work_p_N_p_im + i);
	}

	PRINT_OPS::print_val_cpx_arr("stage1 output",
				     work,
				     N << 1);

	fft_coreloop<SIMD_OPS,
	  PRINT_OPS,
	  (N >> 1), 
	  N_stop,
	  (N_stop == (N >> 1)), 
	  2>::fn(work,
		 twiddles,
		 ((N == (N_stop << 1)) ? 0 : N),
		 0);
	
	fft_coreloop<SIMD_OPS,
	  PRINT_OPS,
	  (N >> 1), 
	  N_stop,
	  (N_stop == (N >> 1)), 
	  1>::fn(work + N,
		 twiddles,
		 ((N == (N_stop << 1)) ? 0 : N),
		 N_stop);
	
	PRINT_OPS::print_std_cpx("final work output",
				 reinterpret_cast<std::complex<val_t>*>(work),
				 N);
      }
    };

    template <class SIMD_OPS,
	      class PRINT_OPS,
	      class STDCPX_OPS,
	      class DIRECTION_OPS,
	      template <class _IN_T, 
			class _OUT_T, 
			class _PRINT_OPS, 
			size_t _N, 
			class _DIRECTION_OPS,
			yafft::mult_t M_T> class UNSWIZZLE_OPS,
	      size_t N,
	      size_t N_stop>
    struct std_complex_fft_chirp_mul {

      typedef typename SIMD_OPS::SIMD_T simd_t;
      typedef typename SIMD_OPS::VAL_T val_t;
      typedef typename SIMD_OPS::VAL_P_PAIR_T val_p_pair_t;

      static inline void
      fn(val_t * const chirp_fft,
	 val_t * const work,
	 val_t * const twiddles,
	 val_t * const chirp,
	 std::complex<val_t> * const out) {

	val_p_pair_t val_p_pair(work, chirp_fft);

	UNSWIZZLE_OPS<val_p_pair_t, 
	  val_t, 
	  PRINT_OPS,
	  N, 
	  typename SIMD_OPS::forward_ops, 
	  yafft::PRE>::fn(&val_p_pair,
			   reinterpret_cast<val_t*>(out));

	PRINT_OPS::print_std_cpx("unswizzle_post_mul output",
				 (std::complex<val_t>*) out,
				 N);

	val_t * const in = (val_t*) out;
	const size_t im = N_stop >> 1;

	simd_t bitflips = SIMD_OPS::load_bitflip1();

	val_t * const in_p_im = in + im;

	val_t * const in_p_N = in + N;
	val_t * const in_p_N_p_im = in_p_N + im;

	val_t * const work_p_im = work + im;

	val_t * const work_p_N = work + N;
	val_t * const work_p_N_p_im = work_p_N + im;

	for (size_t i = 0; i < N; i += N_stop) {

	  simd_t in1 = SIMD_OPS::load(in + i);
	  simd_t in2 = SIMD_OPS::load(in_p_im + i);
	  
	  PRINT_OPS::print_simd("stage1 A in1", in1); 
	  PRINT_OPS::print_simd("stage1 A in2", in2); 

	  simd_t in3 = SIMD_OPS::load(in_p_N + i);
	  simd_t in4 = SIMD_OPS::load(in_p_N_p_im + i);
	  
	  PRINT_OPS::print_simd("stage1 B in1",  in3);
	  PRINT_OPS::print_simd("stage1 B in2",  in4);

	  simd_t tmp1, tmp2;
	  
	  tmp1 = STDCPX_OPS::real(in1, in2);
	  tmp2 = STDCPX_OPS::real(in3, in4);

	  PRINT_OPS::print_simd("stage1 real tmp1", tmp1);
	  PRINT_OPS::print_simd("stage1 real tmp2", tmp2);

	  SIMD_OPS::store(work + i, SIMD_OPS::add(tmp1, tmp2));
	  SIMD_OPS::store(work_p_N + i, SIMD_OPS::sub(tmp1, tmp2));
	  
	  PRINT_OPS::print_val("stage1 real out1", work + i);
	  PRINT_OPS::print_val("stage1 real out2", work_p_N + i);

	  tmp1 = DIRECTION_OPS::bitflip(STDCPX_OPS::imag(in1, in2), bitflips); 
	  tmp2 = DIRECTION_OPS::bitflip(STDCPX_OPS::imag(in3, in4), bitflips);

	  PRINT_OPS::print_simd("stage1 imag tmp1", tmp1);
	  PRINT_OPS::print_simd("stage1 imag tmp2", tmp2);

	  SIMD_OPS::store(work_p_im + i, SIMD_OPS::add(tmp1, tmp2));
	  SIMD_OPS::store(work_p_N_p_im + i, SIMD_OPS::sub(tmp1, tmp2));

	  PRINT_OPS::print_val("stage1 imag out1", work_p_im + i); 
	  PRINT_OPS::print_val("stage1 imag out2", work_p_N_p_im + i);
	}

	PRINT_OPS::print_val_cpx_arr("stage1 output",
				     work,
				     N << 1);

	fft_coreloop<SIMD_OPS,
	  PRINT_OPS,
	  (N >> 1), 
	  N_stop,
	  (N_stop == (N >> 1)), 
	  2>::fn(work,
		 twiddles,
		 ((N == (N_stop << 1)) ? 0 : N),
		 0);
	
	fft_coreloop<SIMD_OPS,
	  PRINT_OPS,
	  (N >> 1), 
	  N_stop,
	  (N_stop == (N >> 1)), 
	  1>::fn(work + N,
		 twiddles,
		 ((N == (N_stop << 1)) ? 0 : N),
		 N_stop);
	
	PRINT_OPS::print_val_cpx_arr("final work output",
				     work,
				     N << 1);

	PRINT_OPS::print_val_cpx_arr("final chirp output",
				     chirp,
				     N << 1);

	val_p_pair_t val_p_pair2(work, chirp);

	UNSWIZZLE_OPS<val_p_pair_t, 
	  val_t, 
	  PRINT_OPS, 
	  N,
	  DIRECTION_OPS,
	  yafft::POST>::fn(&val_p_pair2,
			   reinterpret_cast<val_t*>(out));
      }
    };

    template <size_t N,
	      class SIMD_OPS,
	      class PRINT_OPS,
	      class STDCPX_OPS,
	      template <class _IN_T, 
			class _OUT_T, 
			class _PRINT_OPS, 
			size_t _N, 
			class _DIRECTION_OPS,
			yafft::mult_t _M_T> class UNSWIZZLE_OPS>
    class fft_impl_pow2 : public iyafft_pow2<typename SIMD_OPS::VAL_T> {

      template <class __SIMD_OPS,
		class __PRINT_OPS,
		class __STDCPX_OPS,
		template <class __IN_T,
			  class __OUT_T,
			  class ___PRINT_OPS, 
			  size_t ___N, 
			  class ___DIRECTION_OPS,
			  typename yafft::mult_t __M_T> class __UNSWIZZLE_OPS> 
      friend class fft_impl_npow2;
  
      typedef fft_impl_pow2<N, SIMD_OPS, PRINT_OPS, 
			    STDCPX_OPS, UNSWIZZLE_OPS> fft_impl_t;

      typedef typename SIMD_OPS::SIMD_T simd_t;
      typedef typename SIMD_OPS::VAL_T val_t;

      typedef typename std::complex<val_t> cpx_val_t;
      
    public:
      
      fft_impl_pow2() : twiddles_(NULL), work_(NULL) {}

      ~fft_impl_pow2() { if (twiddles_) free(twiddles_); }
      
      bool init() {      
	// I'm just doing this on Intel for now, so let's stick with 
	// 64bit cache line alignment, instead of doing fancy stuff 
	// in here for discovery.	
	const size_t cacheline_sz = 64;
	size_t cachelines = 0;

	// we support a minimum of size 8 FFTs with the complex FFT code
	// size N twiddle bank for complex FFT
	const size_t twiddles_sz = (N == 16) ? (N * 2) : (N * 3);
	const size_t twiddles_sz_bytes = twiddles_sz * sizeof(val_t);
	const bool twiddles_sz_aligned = !(twiddles_sz_bytes % cacheline_sz);
	const size_t twiddles_cachelines = 
	  (twiddles_sz_bytes / cacheline_sz) + (twiddles_sz_aligned ? 0 : 1);
	cachelines += twiddles_cachelines; 
	
	// size N/2 twiddle bank for the real FFT - note we only support 
	// N >= 16 real FFTs
	const size_t N_o_2 = N >> 1;
	const size_t real_twiddles_sz = (N_o_2 == 16) ? (N_o_2 * 2) : (N_o_2 * 3);
	const size_t real_twiddles_sz_bytes = real_twiddles_sz * sizeof(val_t);
	const bool real_twiddles_sz_aligned = 
	  !(real_twiddles_sz_bytes % cacheline_sz);
	const size_t real_twiddles_cachelines = 
	  (real_twiddles_sz_bytes / cacheline_sz) + 
	  (real_twiddles_sz_aligned ? 0 : 1); 
	cachelines += real_twiddles_cachelines; 

	// the last little array chunk for handling real FFT unpacking
	// in the forward direction
	const size_t real_angles_sz = N >> 2;
	const size_t real_angles_sz_bytes = real_angles_sz * sizeof(cpx_val_t);
	const bool real_angles_sz_aligned = !(real_angles_sz_bytes % cacheline_sz);
	const size_t real_angles_cachelines = 
	  (real_angles_sz_bytes / cacheline_sz) + (real_angles_sz_aligned ? 0 : 1);
	cachelines += real_angles_cachelines;

	// work array
	const size_t work_sz = N * 2;
	const size_t work_sz_bytes = work_sz * sizeof(val_t);
	const bool work_sz_aligned = !(work_sz_bytes % cacheline_sz);
	const size_t work_cachelines = 
	  (work_sz_bytes / cacheline_sz) + (work_sz_aligned ? 0 : 1);
	cachelines += work_cachelines;

	void * v_p = NULL;
	// try for the big allocation
	if (yafft::utils::alloc_aligned(&v_p, cacheline_sz, 
					cacheline_sz * cachelines))
	  return false;

	unsigned char * const buffer_p = (unsigned char*) v_p;	

	twiddles_ = (val_t*) buffer_p;
	gen_twiddles<val_t, SIMD_OPS::STOP>(N, twiddles_);
	
	real_twiddles_ = (val_t*) (((unsigned char*) twiddles_) +
				   (twiddles_cachelines * cacheline_sz));
	
	gen_twiddles<val_t, SIMD_OPS::STOP>(N_o_2, real_twiddles_);

	real_angles_ = (cpx_val_t*) (((unsigned char*) real_twiddles_) +
				     (real_twiddles_cachelines * cacheline_sz));

	for (size_t i = 1; i < real_angles_sz; ++i) {
	  const float ang = (i * M_PI) / (N_o_2);
	  real_angles_[i - 1].real() = cosf(ang);
	  real_angles_[i - 1].imag() = -sinf(ang);
	}

	work_ = (val_t*) (((unsigned char*) real_angles_) + 
			  (real_angles_cachelines * cacheline_sz));
	
	return true;
      }
      
      // complex to complex
      void run_forward(cpx_val_t * const in,
		       cpx_val_t * const out) {

	std_complex_fft<SIMD_OPS,
	  PRINT_OPS,
	  STDCPX_OPS,
	  typename SIMD_OPS::forward_ops,
	  UNSWIZZLE_OPS,
	  N,
	  SIMD_OPS::STOP>::fn(in,
			      work_,
			      twiddles_,
			      out);
      }

      // real to complex      
      void run_forward(val_t * const in,
		       cpx_val_t * const out) {

	std_complex_fft<SIMD_OPS,
	  PRINT_OPS,
	  STDCPX_OPS,
	  typename SIMD_OPS::forward_ops,
	  UNSWIZZLE_OPS,
	  N >> 1,
	  SIMD_OPS::STOP>::fn((cpx_val_t*) in,
			      work_,
			      real_twiddles_,
			      out);

	const size_t N_o_2 = N >> 1;
	const val_t half = 1.0f / 2.0f;
	
	const val_t fr = out[0].real(), fi = out[0].imag();
	out[0].real() = fr + fi;
	out[N_o_2].real() = fr - fi;
	out[0].imag() = out[N_o_2].imag() = 0;
	
	for (size_t i = 1; i < (N_o_2>>1) + 1; ++i) {
	  const val_t r1 = out[i].real(), r2 = out[N_o_2 - i].real();
	  const val_t i1 = out[i].imag(), i2 = out[N_o_2 - i].imag();
	  
	  const val_t xr = (r1 + r2) * half;
	  const val_t yi = (r2 - r1) * half;
	  
	  const val_t yr = (i1 + i2) * half;
	  const val_t xi = (i1 - i2) * half;
	  
	  const val_t wr = real_angles_[i-1].real();
	  const val_t wi = real_angles_[i-1].imag();
	  
	  const val_t dr = yr*wr - yi*wi;
	  const val_t di = yr*wi + yi*wr;
	  
	  const size_t N_m_i = N - i;

	  out[i].real() = xr + dr;
	  out[N_m_i].real() = out[i].real();
	  out[i].imag() = xi + di;
	  out[N_m_i].imag() = -out[i].imag();
	  
	  const size_t N_o_2_m_i = N_o_2 - i;
	  const size_t N_m_N_o_2_m_i = N - N_o_2_m_i;

	  out[N_o_2_m_i].real() = xr - dr;
	  out[N_m_N_o_2_m_i].real() = out[N_o_2_m_i].real();
	  out[N_o_2_m_i].imag() = -xi + di;
	  out[N_m_N_o_2_m_i].imag() = -out[N_o_2_m_i].imag();
	}
      }

      // complex to complex
      void run_reverse(cpx_val_t * const in,
		       cpx_val_t * const out) {
	
	std_complex_fft<SIMD_OPS,
	  PRINT_OPS,
	  STDCPX_OPS,
	  typename SIMD_OPS::reverse_ops,
	  UNSWIZZLE_OPS,
	  N,
	  SIMD_OPS::STOP>::fn(in,
			      work_,
			      twiddles_,
			      out);  
      }

      void run_reverse(cpx_val_t * const in,
		       val_t * const out) {

	const size_t N_o_2 = N >> 1;
	const float half = 1.0f / 2.0f;

	// we don't want to clobber the input array because we're nice,
	// so copy it over to the output array.  we might be gaining some
	// benefits by avoiding the writes to the array we're not really
	// going to be using, though ideally we could do all of this in
	// the first stage of the FFT and so just do the writes onto the
	// work array.
	cpx_val_t * const out_cpx = (cpx_val_t*) out;
	
	const float 
	  fr = in[0].real() * half, 
	  fi = in[0].imag() * half;

	out_cpx[0].real() = fr + fi;
	out_cpx[0].imag() = fr - fi;

	for (size_t i = 1; i < ((N_o_2 >> 1) + 1); ++i) {

	  const size_t N_o_2_m_i = N_o_2 - i;

	  // references to the upper and lower bounds
	  cpx_val_t& in_cpx_l = in[i];
	  cpx_val_t& in_cpx_u = in[N_o_2_m_i];

	  // references to the upper and lower bounds
	  cpx_val_t& out_cpx_l = out_cpx[i];
	  cpx_val_t& out_cpx_u = out_cpx[N_o_2_m_i];

	  const val_t 
	    r1 = in_cpx_l.real(),
	    i1 = in_cpx_l.imag(),
	    r2 = in_cpx_u.real(),
	    i2 = in_cpx_u.imag();
	  
	  const val_t xr = (r1 + r2) * half;
	  const val_t yi = -((r2 - r1) * half);
	  
	  const val_t yr = (i1 + i2) * half;
	  const val_t xi = (i1 - i2) * half;
	  
	  const val_t wr = real_angles_[i-1].real();
	  const val_t wi = real_angles_[i-1].imag();
	  
	  const val_t dr = yr*wr - yi*wi;
	  const val_t di = yr*wi + yi*wr;
	  
	  out_cpx_l.real() = xr - dr;
	  out_cpx_l.imag() = xi + di;

	  out_cpx_u.real() = xr + dr;
	  out_cpx_u.imag() = di - xi;
	}
	
	std_complex_fft<SIMD_OPS,
	  PRINT_OPS,
	  STDCPX_OPS,
	  typename SIMD_OPS::reverse_ops,
	  UNSWIZZLE_OPS,
	  N >> 1,
	  SIMD_OPS::STOP>::fn(out_cpx,
			      work_,
			      real_twiddles_,
			      out_cpx);

	const val_t norm_factor = 1.0f / (N >> 1);

	// normalize
	for (size_t i = 0; i < N; ++i) {
	  out[i] *= norm_factor;
	}
      }

    private:

      inline val_t * const get_work() { return work_; }
      
      // just have a fake unswizzle class here
      template <typename _IN_T,
		typename _OUT_T,
		typename _PRINT_OPS,
		size_t _N,
		typename _DIRECTION_OPS,
		yafft::mult_t _M_T>
      struct unswizzle_no_ops {
	static inline void 
	fn(_IN_T * const in, _OUT_T * const out) {}
      };

      void run_forward_nounswizzle(cpx_val_t * const in) {
	
	cpx_val_t * out = NULL;

	std_complex_fft<SIMD_OPS,
	  PRINT_OPS,
	  STDCPX_OPS,
	  typename SIMD_OPS::forward_ops,
	  unswizzle_no_ops,
	  N,
	  SIMD_OPS::STOP>::fn(in,
			      work_,
			      twiddles_,
			      out);
      }
      
      // these are for the non-power of two FFTs

      // this runs the fft, but doesn't write it to the output
      // array - the output array will be consumed by the 
      // 'run forward unswizzled' function below      
      void run_forward_mul_noout(cpx_val_t * const in,
				 val_t * const mul) {
	
	std_complex_fft_mul_nounswizzle<SIMD_OPS,
	  PRINT_OPS,
	  STDCPX_OPS,
	  typename SIMD_OPS::forward_ops,
	  N,
	  SIMD_OPS::STOP>::fn(in,
			      work_,
			      twiddles_,
			      mul);
      }

      void run_reverse_mul_noout(cpx_val_t * const in,
				 val_t * const mul) {
	
	std_complex_fft_mul_nounswizzle<SIMD_OPS,
	  PRINT_OPS,
	  STDCPX_OPS,
	  typename SIMD_OPS::reverse_ops,
	  N,
	  SIMD_OPS::STOP>::fn(in,
			      work_,
			      twiddles_,
			      mul);
      }

      // this runs the reverse FFT, but expects the signal value to already
      // be in the work_ array (see run_forward_mul_noout) above
      // it also runs some multiplications in the course of doing loads,
      // to eke out a bit of performance by avoiding redundant load/stores
      void run_reverse_mul_noin(val_t * const chirp_fft, 
				val_t * const chirp,
				cpx_val_t * const out) {
	
	std_complex_fft_chirp_mul<SIMD_OPS,
	  PRINT_OPS,
	  STDCPX_OPS,
	  typename SIMD_OPS::reverse_ops,
	  UNSWIZZLE_OPS,
	  N,
	  SIMD_OPS::STOP>::fn(chirp_fft,
			      work_,
			      twiddles_,
			      chirp,
			      out);
      }

      void run_forward_mul_noin(val_t * const chirp_fft, 
				val_t * const chirp,
				cpx_val_t * const out) {
	
	std_complex_fft_chirp_mul<SIMD_OPS,
	  PRINT_OPS,
	  STDCPX_OPS,
	  typename SIMD_OPS::forward_ops,
	  UNSWIZZLE_OPS,
	  N,
	  SIMD_OPS::STOP>::fn(chirp_fft,
			      work_,
			      twiddles_,
			      chirp,
			      out);
      }

      fft_impl_pow2(const fft_impl_pow2&) {}

      val_t * twiddles_;
      val_t * real_twiddles_;
      cpx_val_t * real_angles_;
      val_t * work_;
    };

    static inline size_t
    next_pow2(size_t s) {
      if (s == 0) return 1;
      s--;
      for (size_t i = 1; i < sizeof(size_t)*8; i <<= 1) s = s | s >> i;
      return s + 1;
    }

    template <class SIMD_OPS,
	      class PRINT_OPS,
	      class STDCPX_OPS,
	      template <class _IN_T, 
			class _OUT_T, 
			class _PRINT_OPS, 
			size_t _N, 
			class _DIRECTION_OPS,
			yafft::mult_t _M_T> class UNSWIZZLE_OPS>
    class fft_impl_npow2 : public iyafft<typename SIMD_OPS::VAL_T> {
      
      typedef typename SIMD_OPS::SIMD_T simd_t;
      typedef typename SIMD_OPS::VAL_T val_t;

      typedef std::complex<val_t> cpx_val_t;

      typedef iyafft<val_t> iyafft_t;
      typedef iyafft_pow2<val_t> iyafft_pow2_t;

      typedef boost::shared_ptr<iyafft_t> iyafft_ptr;
      typedef boost::shared_ptr<iyafft_pow2_t> iyafft_pow2_ptr;
      
    public:
      
      fft_impl_npow2(const size_t N) 
	: N_(N), 
	  N_next_pow2t2_(next_pow2(N) << 1),
	  chirp1_(NULL),
	  chirp2_(NULL),
	  chirp3_(NULL),
	  work_(NULL),
	  actual_fft_ptr_() {}

      ~fft_impl_npow2() { if (chirp1_) free(chirp1_); }

      bool init() { 
	
	iyafft_ptr tmp_fft = get_fft<val_t>(N_next_pow2t2_);

	tmp_fft =  get_fft<val_t>(N_next_pow2t2_);
	
	actual_fft_ptr_ = boost::dynamic_pointer_cast<iyafft_pow2_t>(tmp_fft);

	if (actual_fft_ptr_ == NULL || !actual_fft_ptr_->init())
	  return false;

	void * v_p = NULL;
	
	yafft::utils::alloc_aligned(&v_p, 64, 
				    (sizeof(val_t) * N_next_pow2t2_) << 3);

	chirp1_ = static_cast<val_t*>(v_p);
	chirp2_ = chirp1_ + (N_next_pow2t2_ << 1);
	chirp3_ = chirp2_ + (N_next_pow2t2_ << 1);
	work_ = chirp3_ + (N_next_pow2t2_ << 1);
	memset(v_p, 0, (sizeof(val_t) * N_next_pow2t2_) << 3);

	const val_t chirpsig_base = M_PI / ((val_t) N_);
	const val_t scale = val_t(1.0) / N_next_pow2t2_;

	yafft::utils::alloc_aligned(&v_p, 64, 
				    sizeof(size_t) * (N_next_pow2t2_ << 1));

	size_t * chirp2_idxs = static_cast<size_t*>(v_p);
	memset(chirp2_idxs, 0, sizeof(size_t) * (N_next_pow2t2_ << 1));

	{
	  const size_t arrsz = N_next_pow2t2_;
	  for (size_t i = 0; i < arrsz; i += (arrsz >> 2)) {
	    for (size_t j = 0; j < (arrsz >> 3); ++j) {
	      
	      chirp2_idxs[i + j] = (i >> 1) + j;
	      chirp2_idxs[i + j + (arrsz >> 3)] = (i >> 1) + j + (arrsz >> 1);
	    }
	  }
	   
	  generate_twiddleidxs_rec(arrsz, chirp2_idxs, chirp2_idxs + arrsz);
	}

	for (size_t i = 0; i < N_; ++i) {
	    
	    const val_t i_v = (val_t) i;
	    const val_t i_sq = i_v * i_v;
	    const val_t chirpsig = chirpsig_base * i_sq;
	    
	    const size_t r_idx = (i << 1);
	    const size_t i_idx = (i << 1) + 1;
	    
	    chirp1_[r_idx] = cosf(chirpsig);
	    chirp1_[i_idx] = -sinf(chirpsig);
	    
	    chirp3_[r_idx] = chirp1_[r_idx];
	    chirp3_[i_idx] = -chirp1_[i_idx];
	    
	    if (i != 0) {
	      chirp3_[(N_next_pow2t2_ << 1) - r_idx] = chirp3_[r_idx];
	      chirp3_[(N_next_pow2t2_ << 1) - (i_idx - 2)] = chirp3_[i_idx];
	    }
	}

	// this sets up the chirps for the full SSE blocks
	for (size_t i = 0; i < N_next_pow2t2_; ++i) {	    

	    const size_t r_idx = (i << 1);
	    const size_t i_idx = (i << 1) + 1;
	    	
	    chirp2_[r_idx] = chirp1_[chirp2_idxs[i] << 1] * scale;
	    chirp2_[i_idx] = chirp1_[(chirp2_idxs[i] << 1) + 1] * scale;
	}

	SIMD_OPS::stdcpx_ops::swizzle_block(chirp1_, N_next_pow2t2_ << 1);
	SIMD_OPS::stdcpx_ops::swizzle_block(chirp2_, N_next_pow2t2_ << 1);

	actual_fft_ptr_->run_forward_nounswizzle((cpx_val_t*) chirp3_);
	const val_t * const chirp_work = actual_fft_ptr_->get_work();
	memcpy(chirp3_, chirp_work, (sizeof(val_t) * N_next_pow2t2_) << 1);

	free(chirp2_idxs);

	return true; 
      }
 
      // complex to complex
      void run_forward(cpx_val_t * const in,
		       cpx_val_t * const out) {

	/* Ok, time for a bit of explanation.  A chirp-z FFT has the 
	 * following stages:
	 *
	 * create the chirp signal (along with its complex conjugate)
	 * create another buffer with the chirp signal scaled 1/N
	 * FFT the zero padded complex conjugate chirp for use in the 
	 *  convolution
	 * multiply the input signal with the chirp signal
	 * FFT the resulting compound signal
	 * multiply the FFT result with the FFT result from the conjugate
	 *  (aka convolve the two signals)
	 * do the reverse FFT
	 * multiply the result by the scaled chirp	 
	 */
	
	// copy in into the work_ array, and zero out the > N_
	memcpy(work_, in, sizeof(cpx_val_t) * N_);
	memset(work_ + (N_ << 1), 0, sizeof(cpx_val_t) * (N_next_pow2t2_ - N_));

	// multiply the input signal with the chirp signal
	// we also leave it in the actual_fft_ptr_ work array,
	// and skip the unswizzling to another buffer step
	actual_fft_ptr_->run_forward_mul_noout((cpx_val_t*) work_, 
					       chirp1_);

	// there's a lot of extra stuff in here.  I'm doing the 
	// FFT * FFT multiplication inside the unswizzling step
	// of the two work arrays (the actual_fft_ptr_'s work array
	// and the precaculated FFT of the chirp signal,
	// along with the stage 1 additions, to avoid having to do
	// a bunch of redundant loads.  Similary, the multiplication by
	// of the scaled chirp happens in the unswizzling onto the output
	/// buffer.  Hopefully this should give reasonable speed.
	actual_fft_ptr_->run_reverse_mul_noin(chirp3_,
					      chirp2_,
					      (cpx_val_t*) work_);

	// copy the output in work_ into out
	memcpy(out, work_, sizeof(cpx_val_t) * N_);
      }

      // complex to complex
      void run_reverse(cpx_val_t * const in,
		       cpx_val_t * const out) {

	// see run_forward above for notes on what this is doing
	memcpy(work_, in, sizeof(cpx_val_t) * N_);
	memset(work_ + (N_ << 1), 0, sizeof(cpx_val_t) * (N_next_pow2t2_ - N_));

	actual_fft_ptr_->run_reverse_mul_noout((cpx_val_t*) work_, 
					       chirp1_);

	actual_fft_ptr_->run_forward_mul_noin(chirp3_,
					      chirp2_,
					      (cpx_val_t*) work_);

	memcpy(out, work_, sizeof(cpx_val_t) * N_);
      }	

      // real to complex
      void run_forward(val_t * const in,
		       cpx_val_t * const out) {

      }	
	
      // complex to real
      void run_reverse(cpx_val_t * const in,
		       val_t * const out) {

      }	
      
    private:

      const size_t N_;
      const size_t N_next_pow2t2_;
      val_t * chirp1_;
      val_t * chirp2_;
      val_t * chirp3_;
      val_t * work_;
      iyafft_pow2_ptr actual_fft_ptr_;
    };
  } 

  template <typename FP_T, size_t N>
  struct fft_t {};

  template <size_t N>
  struct fft_t<float, N> {
#ifndef _DEBUG
    typedef impl::fft_impl_pow2<N,
				yafft::float_t::sse::ops,
				yafft::float_t::sse::ops::print_noops,
				yafft::float_t::sse::ops::stdcpx_ops,
				yafft::float_t::sse::unswizzle_ops> pow2;

    typedef impl::fft_impl_npow2<yafft::float_t::sse::ops,
				 yafft::float_t::sse::ops::print_noops,
				 yafft::float_t::sse::ops::stdcpx_ops,
				 yafft::float_t::sse::unswizzle_ops> npow2;
#else
    typedef impl::fft_impl_pow2<N,
				yafft::float_t::sse::ops,
				yafft::float_t::sse::ops::print_ops,
				yafft::float_t::sse::ops::stdcpx_ops,
				yafft::float_t::sse::unswizzle_ops> pow2;

    typedef impl::fft_impl_npow2<yafft::float_t::sse::ops,
				 yafft::float_t::sse::ops::print_ops,
				 yafft::float_t::sse::ops::stdcpx_ops,
				 yafft::float_t::sse::unswizzle_ops> npow2;
#endif // _DEBUG 
  };
}

#endif // _YAFFT_IMPL_HPP_
