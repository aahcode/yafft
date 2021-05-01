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

#ifndef _YAFFT_UNSWIZZLE_HPP_
#define _YAFFT_UNSWIZZLE_HPP_
 
#include "yafft_simd.hpp"

namespace yafft {

  /*
   * This is a bit of a cheat - POST also includes the additions that would 
   *  normally be in the first loop
   */
  enum mult_t { NONE, PRE, POST };

  namespace float_t {
    namespace sse {      
	
      typedef yafft::float_t::sse::ops::VAL_T val_t;
      typedef yafft::float_t::sse::ops::SIMD_T simd_t;
      typedef yafft::float_t::sse::ops::VAL_P_PAIR_T val_p_pair_t;

      template <class IN_T, class OUT_T, class PRINT_OPS, 
		size_t N, class DIRECTION_OPS, mult_t M_T>
      struct unswizzle_ops_core_impl;
      
      template <class PRINT_OPS, size_t N, class DIRECTION_OPS>
      struct unswizzle_ops_core_impl <val_p_pair_t, val_t, PRINT_OPS, 
				      N, DIRECTION_OPS, PRE> {
	static inline void
	fn(val_p_pair_t * const _in,
	   const size_t in_offset,
	   val_t * const out,
	   simd_t& bitflips) __attribute__((always_inline)) {

	  val_t * const first = _in->first + in_offset;
	  val_t * const second = _in->second + in_offset;

	  // stage 1 - front quarter
	  simd_t r1, i1;

	  {
	    simd_t r_in = ops::load(first);
	    simd_t i_in = ops::load(first + 4);
	    
	    PRINT_OPS::print_simd("pre r_in: ", r_in);
	    PRINT_OPS::print_simd("pre i_in: ", i_in);
	    
	    simd_t r_mul = ops::load(second);
	    simd_t i_mul = ops::load(second + 4);
	    
	    PRINT_OPS::print_simd("pre r_mul: ", r_mul);
	    PRINT_OPS::print_simd("pre i_mul: ", i_mul);
	    
	    r1 = ops::sub(ops::mul(r_in, r_mul),
			  ops::mul(i_in, i_mul));
	    
	    i1 = DIRECTION_OPS::bitflip(ops::add(ops::mul(r_in, i_mul),
						 ops::mul(i_in, r_mul)),
					bitflips);
	    
	    PRINT_OPS::print_simd("pre r1: ", r1);
	    PRINT_OPS::print_simd("pre i1: ", i1);
	  }

	  simd_t r2, i2;

	  {
	    simd_t r_in = ops::load(first + N);
	    simd_t i_in = ops::load(first + N + 4);
	    
	    PRINT_OPS::print_simd("pre r_in: ", r_in);
	    PRINT_OPS::print_simd("pre i_in: ", i_in);
	    
	    simd_t r_mul = ops::load(second + N);
	    simd_t i_mul = ops::load(second + N + 4);
	    
	    PRINT_OPS::print_simd("pre r_mul: ", r_mul);
	    PRINT_OPS::print_simd("pre i_mul: ", i_mul);
	    
	    r2 = ops::sub(ops::mul(r_in, r_mul),
			  ops::mul(i_in, i_mul));
	    
	    i2 = DIRECTION_OPS::bitflip(ops::add(ops::mul(r_in, i_mul),
						 ops::mul(i_in, r_mul)),
					bitflips);
	    
	    PRINT_OPS::print_simd("pre r2: ", r2);
	    PRINT_OPS::print_simd("pre i2: ", i2);
	  }

	  simd_t x1 = ops::unpacklo(r1, i1);
	  PRINT_OPS::print_simd("pre store 1 low: ", x1);
	  
	  simd_t x2 = ops::unpacklo(r2, i2);
	  PRINT_OPS::print_simd("pre store 2 low: ", x2);
	  
	  ops::store(out, ops::shuf_1_0_1_0(x1, x2));
	  ops::store(out + (N >> 1), ops::shuf_3_2_3_2(x1, x2));
	  
	  simd_t x3 = ops::unpackhi(r1, i1);
	  PRINT_OPS::print_simd("pre store 1 high: ", x3);
	  
	  simd_t x4 = ops::unpackhi(r2, i2);
	  PRINT_OPS::print_simd("pre store 2 high: ", x4);
	  
	  ops::store(out + (N >> 2), ops::shuf_1_0_1_0(x3, x4));
	  ops::store(out + (N >> 1) + (N >> 2), 
		     ops::shuf_3_2_3_2(x3, x4));
	}
      };
      
      template <class PRINT_OPS, size_t N, class DIRECTION_OPS>
      struct unswizzle_ops_core_impl <val_p_pair_t, val_t, PRINT_OPS, 
				      N, DIRECTION_OPS, POST> {
	static inline void
	fn(val_p_pair_t * const _in,
	   const size_t in_offset,
	   val_t * const out,
	   simd_t& bitflips) __attribute__((always_inline)) {

	  val_t * const first = _in->first + in_offset;
	  val_t * const second = _in->second + in_offset;

	  // stage 1 - front quarter
	  simd_t r1, i1;

	  {
	    simd_t r_in = ops::load(first);
	    PRINT_OPS::print_simd("r_in: ", r_in);
	    
	    simd_t i_in = DIRECTION_OPS::bitflip(ops::load(first + 4), 
						 bitflips);
	    PRINT_OPS::print_simd("i_in: ", i_in);
	    
	    simd_t r_mul = ops::load(second);
	    PRINT_OPS::print_simd("r_mul: ", r_mul);

	    simd_t i_mul = ops::load(second + 4);
	    PRINT_OPS::print_simd("i_mul: ", i_mul);

	    r1 = ops::sub(ops::mul(r_in, r_mul),
			  ops::mul(i_in, i_mul));
	    
	    i1 = ops::add(ops::mul(r_in, i_mul),
			  ops::mul(i_in, r_mul));	    

	    PRINT_OPS::print_simd("r1: ", r1);
	    PRINT_OPS::print_simd("i1: ", i1);
	  }
	      
	  simd_t r2, i2;

	  {
	    simd_t r_in = ops::load(first + N);
	    PRINT_OPS::print_simd("r_in: ", r_in);
	    
	    simd_t i_in = DIRECTION_OPS::bitflip(ops::load(first + N + 4), 
						 bitflips);
	    PRINT_OPS::print_simd("i_in: ", i_in);
	    
	    simd_t r_mul = ops::load(second + N);
	    PRINT_OPS::print_simd("r_mul: ", r_mul);

	    simd_t i_mul = ops::load(second + N + 4);
	    PRINT_OPS::print_simd("i_mul: ", i_mul);

	    r2 = ops::sub(ops::mul(r_in, r_mul),
			  ops::mul(i_in, i_mul));
	    
	    i2 = ops::add(ops::mul(r_in, i_mul),
			  ops::mul(i_in, r_mul));	    

	    PRINT_OPS::print_simd("r2: ", r2);
	    PRINT_OPS::print_simd("i2: ", i2);
	  }
	      
	  simd_t x1 = ops::unpacklo(r1, i1);
	  PRINT_OPS::print_simd("store 1 low: ", x1);
	  
	  simd_t x2 = ops::unpacklo(r2, i2);
	  PRINT_OPS::print_simd("store 2 low: ", x2);
	  
	  ops::store(out, ops::shuf_1_0_1_0(x1, x2));
	  ops::store(out + (N >> 1), ops::shuf_3_2_3_2(x1, x2));
	  
	  simd_t x3 = ops::unpackhi(r1, i1);
	  PRINT_OPS::print_simd("store 1 high: ", x3);
	  
	  simd_t x4 = ops::unpackhi(r2, i2);
	  PRINT_OPS::print_simd("store 2 high: ", x4);
	  
	  ops::store(out + (N >> 2), ops::shuf_1_0_1_0(x3, x4));
	  ops::store(out + (N >> 1) + (N >> 2), 
		     ops::shuf_3_2_3_2(x3, x4));
	}
      };
      
      template <class PRINT_OPS, size_t N, class DIRECTION_OPS, mult_t M_T>
      struct unswizzle_ops_core_impl <val_t, val_t, PRINT_OPS, 
				      N, DIRECTION_OPS, M_T> {
	static inline void
	fn(val_t * const _in,
	   const size_t in_offset,
	   val_t * const out,
	   simd_t& bitflips) __attribute__((always_inline)) {
	  
	  val_t * const in = _in + in_offset;
	  
	  // stage 1 - front quarter
	  simd_t r1 = ops::load(in);
	  simd_t i1 = DIRECTION_OPS::bitflip(ops::load(in + 4), bitflips);
	  
	  simd_t r2 = ops::load(in + N);
	  simd_t i2 = DIRECTION_OPS::bitflip(ops::load(in + N + 4), bitflips);
	  
	  simd_t x1 = ops::unpacklo(r1, i1);
	  PRINT_OPS::print_simd("load 1 low: ", x1);
	  
	  simd_t x2 = ops::unpacklo(r2, i2);
	  PRINT_OPS::print_simd("load 2 low: ", x2);
	  
	  ops::store(out, ops::shuf_1_0_1_0(x1, x2));
	  ops::store(out + (N >> 1), ops::shuf_3_2_3_2(x1, x2));
	  
	  simd_t x3 = ops::unpackhi(r1, i1);
	  PRINT_OPS::print_simd("load 1 high: ", x3);
	  
	  simd_t x4 = ops::unpackhi(r2, i2);
	  PRINT_OPS::print_simd("load 2 high: ", x4);
	  
	  ops::store(out + (N >> 2), ops::shuf_1_0_1_0(x3, x4));
	  ops::store(out + (N >> 1) + (N >> 2), 
		     ops::shuf_3_2_3_2(x3, x4));	  
	}
      };
      
      template<class IN_T,
	       class OUT_T, 
	       class PRINT_OPS, 	       
	       class DIRECTION_OPS,
	       size_t I,
	       size_t Offset,
	       size_t full_N,
	       mult_t M_T>
      struct unswizzle_ops_core {

	typedef unswizzle_ops_core<IN_T, OUT_T, PRINT_OPS, DIRECTION_OPS, 
				   I >> 1, Offset << 1, full_N, M_T> uo;

	static inline void
	fn(IN_T * const in,
	   const size_t in_offset,
	   OUT_T ** out) {
	  
	  uo::fn(in, in_offset, out);
	  uo::fn(in, in_offset + Offset, out);
	}
      };
      
      template<class IN_T,
	       class OUT_T,
	       class PRINT_OPS, 
	       class DIRECTION_OPS, 
	       size_t Offset,
	       size_t full_N,
	       mult_t M_T>
      struct unswizzle_ops_core<IN_T,
				OUT_T,
				PRINT_OPS, 
				DIRECTION_OPS, 
				32, 
				Offset, 
				full_N,
				M_T> {

	typedef unswizzle_ops_core_impl<IN_T, OUT_T, PRINT_OPS, full_N, 
					DIRECTION_OPS, M_T> uo;

	static inline void
	fn(IN_T * const in,
	   const size_t in_offset,
	   OUT_T ** out_p) {
	  
	  simd_t bitflips = ops::load_bitflip1();
	  OUT_T * const out = *out_p;
	  
	  { // front
	    uo::fn(in, in_offset, out, bitflips);
	    uo::fn(in, in_offset + (full_N >> 1), out + 4, bitflips);
	    uo::fn(in, in_offset + (full_N >> 2), out + 8, bitflips);
	    uo::fn(in, in_offset + (full_N >> 2) + (full_N >> 1), out + 12,
		   bitflips);
	  }
	  
	  { // back
	    uo::fn(in, in_offset + 8, out + full_N, bitflips);
	    uo::fn(in, in_offset + (full_N >> 1) + 8, out + full_N + 4,
		   bitflips);
	    uo::fn(in, in_offset + (full_N >> 2) + 8, out + full_N + 8,
		   bitflips);
	    uo::fn(in, in_offset + (full_N >> 2) + (full_N >> 1) + 8, 
		   out + full_N + 12,
		   bitflips); 	  
	  }
	  
	  *out_p += 16;
	}
      };
	
      template <class IN_T,
		class OUT_T,
		class PRINT_OPS, 
		size_t N, 
		class DIRECTION_OPS,
		mult_t M_T>
      struct unswizzle_ops {

	static inline void
	fn(IN_T * const in, 
	   OUT_T * const out) {

	  OUT_T * out_tmp = out;

	  unswizzle_ops_core<IN_T,
	    OUT_T,
	    PRINT_OPS,
	    DIRECTION_OPS,
	    N >> 1,
	    16,
	    N,
	    M_T>::fn(in,
		     0,
		     &out_tmp);
	}
      };      

      // specialization for 16      
      template <class IN_T,
		class OUT_T,
		class PRINT_OPS, 
		class DIRECTION_OPS,
		mult_t M_T>
      struct unswizzle_ops<IN_T, OUT_T, PRINT_OPS, 16, DIRECTION_OPS, M_T> {

	typedef unswizzle_ops_core_impl<IN_T, OUT_T, PRINT_OPS, 
					16, DIRECTION_OPS, M_T> uo;

	static inline void
	fn(IN_T * const in,
	   OUT_T * const out) {
	  
	  simd_t bitflips = ops::load_bitflip1();
	  
	  uo::fn(in, 0, out, bitflips);
	  uo::fn(in, 8, out + 16, bitflips);
	}
      };
      
      // specialization for 32
      template <class IN_T,
		class OUT_T,
		class PRINT_OPS, 
		class DIRECTION_OPS,
		mult_t M_T>
      struct unswizzle_ops<IN_T, OUT_T, PRINT_OPS, 32, DIRECTION_OPS, M_T> {

	typedef unswizzle_ops_core_impl<IN_T, OUT_T, PRINT_OPS, 
					32, DIRECTION_OPS, M_T> uo;

	static inline void
	fn(IN_T * const in, 
	   OUT_T * const out) {
	  
	  simd_t bitflips = ops::load_bitflip1();
	  
	  uo::fn(in, 0, out, bitflips);
	  uo::fn(in, 16, out + 4, bitflips);
	  uo::fn(in, 8, out + 32, bitflips);
	  uo::fn(in, 24, out + 36, bitflips);	  
	}
      };
      
      // specialization for 64      
      template <class IN_T,
		class OUT_T,
		class PRINT_OPS, 
		class DIRECTION_OPS,
		mult_t M_T>
      struct unswizzle_ops<IN_T, OUT_T, PRINT_OPS, 64, DIRECTION_OPS, M_T> {

	typedef unswizzle_ops_core_impl<IN_T, OUT_T, PRINT_OPS, 
					64, DIRECTION_OPS, M_T> uo;

	static inline void
	fn(IN_T * const in, 
	   OUT_T * const out) {
	  
	  simd_t bitflips = ops::load_bitflip1();
	  
	  {
	    uo::fn(in, 0, out, bitflips);
	    uo::fn(in, 32, out + 4, bitflips);
	    uo::fn(in, 16, out + 8, bitflips);
	    uo::fn(in, 48, out + 12, bitflips);
	  }
	  
	  {
	    uo::fn(in, 8, out + 64, bitflips);
	    uo::fn(in, 40, out + 68, bitflips);
	    uo::fn(in, 24, out + 72, bitflips);
	    uo::fn(in, 56, out + 76, bitflips);	  
	  }
	}
      };      
    }
  }
}
#endif // _YAFFT_UNSWIZZLE_HPP_
