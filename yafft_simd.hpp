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

#ifndef _YAFFT_SIMD_HPP_
#define _YAFFT_SIMD_HPP_

#include <xmmintrin.h>
#include <string.h>

/*
static inline int
posix_memalign(void **memptr, size_t alignment, size_t size) {
  *memptr = valloc(size);

  if (*memptr == NULL)
    return ENOMEM;

  return 0;
}
*/

namespace yafft {

  namespace utils {
    template <size_t N>
    class N_is_ {
    public:
      enum { POW2 = (N & (N - 1)) == 0 ? 1 : 0 };
    };

    static inline int
    alloc_aligned(void **memptr, size_t alignment, size_t size) {
#if defined __MACH__
      *memptr = valloc(size);

      if (*memptr == NULL)
	return ENOMEM;

      return 0;
#elif defined __LINUX__
      return posix_memalign(memptr, alignment, size);
#else
#error "all allocations are going to be non-aligned - good luck"
      *memptr = malloc(size);
#endif
    }
  }

  namespace float_t {
    namespace sse {      
      static float bitflip1[4] __attribute__((aligned(16))) =
      { -0.0f, -0.0f, -0.0f, -0.0f };
      
      static float bitflip2[4] __attribute__((aligned(16))) =
      { 0.0f, -0.0f, 0.0f, -0.0f };
      
      struct ops {      
	typedef float VAL_T;
	typedef __m128 SIMD_T;
	typedef std::pair<float*, float*> VAL_P_PAIR_T;

	// the idea here is that this is how many VAL_T's are in a complex
	// simd bundle
	enum { STOP = 8 };
	
	static inline SIMD_T 
	load(const VAL_T * f) {
	  return _mm_load_ps(f);
	}
	
	static inline SIMD_T
	load_bitflip1() {
	  return _mm_load_ps(bitflip1);
	}

	static inline SIMD_T
	load_bitflip2() {
	  return _mm_load_ps(bitflip2);
	}

	static inline void 
	store(VAL_T * const f, SIMD_T xmm) {
	  _mm_store_ps(f, xmm);
	}
	
	static inline SIMD_T 
	add(SIMD_T xmm1, SIMD_T xmm2) {
	  return _mm_add_ps(xmm1, xmm2);
	}
	
	static inline SIMD_T 
	sub(SIMD_T xmm1, SIMD_T xmm2) {
	  return _mm_sub_ps(xmm1, xmm2);
	}
	
	static inline SIMD_T
	mul(SIMD_T xmm1, SIMD_T xmm2) {
	  return _mm_mul_ps(xmm1, xmm2);
	}
	
	static inline void 
	cpx_mul(SIMD_T rxmm1, SIMD_T ixmm1,
		SIMD_T rxmm2, SIMD_T ixmm2,
		SIMD_T orxmm, SIMD_T oixmm) {
	  orxmm = _mm_sub_ps(_mm_mul_ps(rxmm1, rxmm2),
			     _mm_mul_ps(ixmm1, ixmm2));
	  
	  oixmm = _mm_add_ps(_mm_mul_ps(rxmm1, ixmm2),
			     _mm_mul_ps(ixmm1, rxmm2));
	}
	
	static inline SIMD_T
	shuf_3_2_3_2(SIMD_T xmm1, SIMD_T xmm2) {
	  return _mm_shuffle_ps(xmm1, xmm2,
				_MM_SHUFFLE(3, 2, 3, 2));
	}
	
	static inline SIMD_T
	shuf_1_0_1_0(SIMD_T xmm1, SIMD_T xmm2) {
	  return _mm_shuffle_ps(xmm1, xmm2,
				_MM_SHUFFLE(1, 0, 1, 0));
	}
	
	static inline SIMD_T
	shuf_3_1_3_1(SIMD_T xmm1, SIMD_T xmm2) {
	  return _mm_shuffle_ps(xmm1, xmm2,
				_MM_SHUFFLE(3, 1, 3, 1));
	}
	
	static inline SIMD_T
	shuf_2_0_2_0(SIMD_T xmm1, SIMD_T xmm2) {
	  return _mm_shuffle_ps(xmm1, xmm2,
				_MM_SHUFFLE(2, 0, 2, 0));
	}
	
	static inline SIMD_T
	shuf_3_1_2_0(SIMD_T xmm1, SIMD_T xmm2) {
	  return _mm_shuffle_ps(xmm1, xmm2,
				_MM_SHUFFLE(3, 1, 2, 0));
	}
	
	static inline SIMD_T
	movehl(SIMD_T xmm1, SIMD_T xmm2) {
	  return _mm_movehl_ps(xmm1, xmm2);
	}
	
	static inline SIMD_T
	movelh(SIMD_T xmm1, SIMD_T xmm2) {
	  return _mm_movelh_ps(xmm1, xmm2);
	}
	
	static inline SIMD_T
	unpacklo(SIMD_T xmm1,SIMD_T xmm2) {
	  return _mm_unpacklo_ps(xmm1, xmm2);
	}
	
	static inline SIMD_T
	unpackhi(SIMD_T xmm1, SIMD_T xmm2) {
	  return _mm_unpackhi_ps(xmm1, xmm2);
	}
	
	static inline void
	prefetch(const void * const p) {
	  _mm_prefetch((char *) p, _MM_HINT_T0);
	}

	static inline SIMD_T
	zero() {
	  return _mm_setzero_ps();	
	}				
	
	struct stdcpx_ops {
	  static inline SIMD_T
	  real(SIMD_T xmm1, SIMD_T xmm2) {
	    return shuf_2_0_2_0(xmm1, xmm2);
	  }
	  
	  static inline SIMD_T
	  imag(SIMD_T xmm1, SIMD_T xmm2) {
	    return shuf_3_1_3_1(xmm1, xmm2);
	  }

	  static inline void
	  swizzle_block(VAL_T * const data_p, const size_t N) {

	    for (size_t i = 0; i < N; i += STOP) {
	      SIMD_T front = load(&data_p[i]);
	      SIMD_T back = load(&data_p[i + (STOP >> 1)]);
	      
	      SIMD_T re = real(front, back);
	      SIMD_T im = imag(front, back);
	      
	      store(&data_p[i], re);
	      store(&data_p[i + (STOP >> 1)], im);	    
	    }
	  }
	};

	struct forward_ops {
	  static inline SIMD_T
	  bitflip(SIMD_T xmm_in, SIMD_T xmm_bits) {
	    return xmm_in;
	  }
	};
	
	struct reverse_ops {
	  static inline SIMD_T
	  bitflip(SIMD_T xmm_in, SIMD_T xmm_bits) {
	    return _mm_xor_ps(xmm_in, xmm_bits);
	  }
	};
	
	struct print_ops {
	  static inline void
	  print_std_cpx(const char * const lbl,
			std::complex<float> * const cpx_arr, 
			const size_t sz) {
	    for (size_t i = 0; i < sz; ++i) {
	      printf("%s - (%+9f, %+9f)\n", 
		     lbl,
		     cpx_arr[i].real(), 
		     cpx_arr[i].imag());
	    }
	  } 
	  
	  static inline void
	  print_val(const char * const lbl,
		    VAL_T * const v) {
	    printf("%s - [%+9f, %+9f, %+9f, %+9f]\n", 
		   lbl, v[0], v[1], v[2], v[3]);
	  }
	  
	  static inline void
	  print_simd(const char * const lbl,
		     SIMD_T s) {
	    VAL_T val_arr[4] __attribute__((aligned(16)));
	    
	    store(&val_arr[0], s);
	    
	    print_val(lbl, &val_arr[0]);
	  }

          static inline void
          print_simd_cpx(const char * const lbl,
                         SIMD_T r,
                         SIMD_T i) {
            VAL_T val_arr[8] __attribute__((aligned(16)));
            
            store(&val_arr[0], r);
            store(&val_arr[4], i);
            
            print_val_cpx(lbl, &val_arr[0], &val_arr[4]);
          }
          
	  static inline void
	  print_val_cpx(const char * const lbl,
			VAL_T * const r, 
			VAL_T * const i) {
	    for (size_t ii = 0; ii < 4; ++ii) {
	      printf("%s - (%+9f, %+9f)\n", 
		     lbl, 
		     r[ii], 
		     i[ii]);
	    }
	  }
	  
          // this is a bit of a hack, as it expects the normal
          // r,r,r,r,i,i,i,i,r,r,r,r,i,..,i swizzled arrays
          static inline void
          print_val_cpx_arr(const char * const lbl,
                            VAL_T * const arr,
                            size_t sz) {
            for (size_t i = 0; i < sz; i += 8) {
              for (size_t j = 0; j < 4; ++j) {
                printf("%s - (%+9f, %+9f)\n",
                       lbl,
                       arr[i+j],
                       arr[i+j+4]);
              }
            }
          }
	  
	  static inline void
	  print_val_cpx_arr2(const char * const lbl,
			     VAL_T * const r_arr,
			     VAL_T * const i_arr,
			     size_t sz) {
	    for (size_t i = 0; i < sz; ++i) {
	      printf("%s - (%+9f, %+9f)\n",
		     lbl,
		     r_arr[i],
		     i_arr[i]);
	    }
	  }
	};
	
	struct print_noops {	  
	  static inline void
	  print_std_cpx(const char * const lbl,
			std::complex<float> * const cpx_arr, 
			const size_t sz) {} 
	
	  static inline void
	  print_val(const char * const lbl,
		    VAL_T * const v) {}
	  
	  static inline void
	  print_simd(const char * const lbl,
		     SIMD_T s) {}
	  
	  static inline void
	  print_cpx(const char * const lbl,
		    std::complex<float> * const cpx_arr, 
		    const size_t sz) {} 
	  
	  static inline void
	  print_val_cpx(const char * const lbl,
			VAL_T * const r, 
			VAL_T * const i) {}
	  
	  static inline void
	  print_simd_cpx(const char * const lbl,
			 SIMD_T r, 
			 SIMD_T i) {}

	  static inline void
	  print_val_cpx_arr(const char * const lbl,
			    VAL_T * const r_arr,
			    size_t sz) {}

	  static inline void
	  print_val_cpx_arr2(const char * const lbl,
			    VAL_T * const r_arr,
			    VAL_T * const i_arr,
			    size_t sz) {}
	};

	template <class PRINT_OPS, size_t N>
	struct unswizzle_ops {
	  static inline void
	  fn(VAL_T * const in, 
	     size_t * const twiddles_idx_arr,
	     VAL_T * const out) {

	    //	    const size_t N_o_4 = N >> 2;
	    const size_t N_o_2 = N >> 1;

	    for (size_t i = 0, ii = 0; i < N_o_2; i += 8, ii += 32) {
	      VAL_T * const in_p_i = in + i;
	      VAL_T * const in_p_i_p_N = in_p_i + N;
	      VAL_T * const in_p_i_p_N_o_2 = in_p_i + N_o_2;
	      VAL_T * const in_p_i_p_N_p_N_o_2 = in_p_i_p_N + N_o_2;

	      VAL_T * const in_p_i_im = in_p_i + 4;
	      VAL_T * const in_p_i_p_N_im = in_p_i_p_N + 4;
	      VAL_T * const in_p_i_p_N_o_2_im = in_p_i_p_N_o_2 + 4;
	      VAL_T * const in_p_i_p_N_p_N_o_2_im = in_p_i_p_N_p_N_o_2 + 4;

	      VAL_T * const out_p_ii = out + ii;
	      
	      SIMD_T xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;

	      xmm1 = load(in_p_i);
	      xmm2 = load(in_p_i_im);

	      PRINT_OPS::print_simd_cpx("load1", xmm1, xmm2);

	      xmm3 = load(in_p_i_p_N);
	      xmm4 = load(in_p_i_p_N_im);

	      PRINT_OPS::print_simd_cpx("load2", xmm3, xmm4);

	      xmm5 = unpacklo(xmm1, xmm2);
	      xmm6 = unpacklo(xmm3, xmm4);

	      store(out_p_ii, movelh(xmm5, xmm6));
	      PRINT_OPS::print_val_cpx_arr("out_p_ii", out_p_ii, 1);
		    
	      store(out_p_ii+8, movehl(xmm6, xmm5));
	      PRINT_OPS::print_val_cpx_arr("out_p_ii+8", out_p_ii+8, 1);

	      xmm5 = unpackhi(xmm1, xmm2);
	      xmm6 = unpackhi(xmm3, xmm4);

	      store(out_p_ii+16, movelh(xmm5, xmm6));
	      PRINT_OPS::print_val_cpx_arr("out_p_ii+16", out_p_ii+16, 1);

	      store(out_p_ii+24, movehl(xmm6, xmm5));
	      PRINT_OPS::print_val_cpx_arr("out_p_ii+24", out_p_ii+24, 1);
	      
	      xmm1 = load(in_p_i_p_N_o_2);
	      xmm2 = load(in_p_i_p_N_o_2_im);
	      
	      PRINT_OPS::print_simd_cpx("load3", xmm1, xmm2);

	      xmm3 = load(in_p_i_p_N_p_N_o_2);
	      xmm4 = load(in_p_i_p_N_p_N_o_2_im);

	      PRINT_OPS::print_simd_cpx("load4", xmm3, xmm4);

	      xmm5 = unpacklo(xmm1, xmm2);
	      xmm6 = unpacklo(xmm3, xmm4);

	      store(out_p_ii+4, movelh(xmm5, xmm6));		    
	      PRINT_OPS::print_val_cpx_arr("out_p_ii+4", out_p_ii+4, 1);

	      store(out_p_ii+12, movehl(xmm6, xmm5));
	      PRINT_OPS::print_val_cpx_arr("out_p_ii+12", out_p_ii+12, 1);

	      xmm5 = unpackhi(xmm1, xmm2);
	      xmm6 = unpackhi(xmm3, xmm4);

	      store(out_p_ii+20, movelh(xmm5, xmm6));
	      PRINT_OPS::print_val_cpx_arr("out_p_ii+20", out_p_ii+20, 1);

	      store(out_p_ii+28, movehl(xmm6, xmm5));
	      PRINT_OPS::print_val_cpx_arr("out_p_ii+28", out_p_ii+28, 1);
	    }
	  }
	};

	template <class PRINT_OPS, size_t N, class DIRECTION_OPS>
	struct unswizzle_ops_stdcpx_loop {
	  static inline void
	  fn(VAL_T * const in,
	     size_t * const twiddles_idx_arr,
	     SIMD_T bitflips,
	     VAL_T * const out) {
	    
	    VAL_T * const in_p_N = in + N;
	    VAL_T * const out2 = out + (N >> 2);

	    for (size_t i = 0; i < (N >> 4); ++i) {
	      const size_t base = twiddles_idx_arr[i];
	      VAL_T * const base_1 = in + base;
	      VAL_T * const base_2 = in_p_N + base;

	      const size_t out_offset = i << 2;

	      SIMD_T xmm01 = load(base_1);
	      PRINT_OPS::print_simd("load_1", xmm01);
	      
	      SIMD_T xmm02 = load(base_2);
	      PRINT_OPS::print_simd("load_2", xmm02);
	      
	      SIMD_T xmm03 = DIRECTION_OPS::bitflip(shuf_1_0_1_0(xmm01, 
								 xmm02),
						    bitflips);

	      PRINT_OPS::print_simd("store_1", xmm03);
	      store(out + out_offset, xmm03);
	      
	      SIMD_T xmm04 = DIRECTION_OPS::bitflip(shuf_3_2_3_2(xmm01, 
								 xmm02),
						    bitflips);

	      PRINT_OPS::print_simd("store_2", xmm04);	      
	      store(out2 + out_offset, xmm04);
	    }
	  }
	};

	template <class PRINT_OPS, size_t N, class DIRECTION_OPS>
	struct unswizzle_ops_stdcpx {
	  static inline void
	  fn(VAL_T * const in, 
	     size_t * const twiddles_idx_arr,
	     VAL_T * const out) {

	    const size_t N_o_2 = N >> 1;
	    const size_t N_o_8 = N >> 3;
	    const size_t N_o_16 = N >> 4;

	    const size_t N_p_N_o_2 = N + N_o_2;
	    const size_t N_o_8_p_N_o_16 = N_o_8 + N_o_16;

	    SIMD_T bitflips = ops::load_bitflip2();

	    unswizzle_ops_stdcpx_loop<PRINT_OPS, 
	      N,
	      DIRECTION_OPS>::fn(in,
				 twiddles_idx_arr,
				 bitflips,
				 out);
	    
	    unswizzle_ops_stdcpx_loop<PRINT_OPS, 
	      N,
	      DIRECTION_OPS>::fn(in,
				 twiddles_idx_arr + N_o_16,
				 bitflips,
				 out + N_o_2);
	    
	    unswizzle_ops_stdcpx_loop<PRINT_OPS, 
	      N,
	      DIRECTION_OPS>::fn(in,
				 twiddles_idx_arr + N_o_8,
				 bitflips,
				 out + N);
	    
	    unswizzle_ops_stdcpx_loop<PRINT_OPS, 
	      N,
	      DIRECTION_OPS>::fn(in,
				 twiddles_idx_arr + N_o_8_p_N_o_16,
				 bitflips,
				 out + N_p_N_o_2);
	    
	  }
	};
      };
    } // namespace sse
  } // namespace float_t
  
  namespace double_t {
    namespace sse {
      struct ops {

	struct stdcpx_ops {

	};

	struct print_ops {

	};

	struct print_noops {

	};

	struct unswizzle_ops {

	};
      };
    } // namespace sse
  } // namespace double_t
} // namespace yafft

#endif // _YAFFT_SIMD_HPP_
