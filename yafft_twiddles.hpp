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

#ifndef _YAFFT_TWIDDLES_HPP_
#define _YAFFT_TWIDDLES_HPP_

namespace yafft {
  namespace impl {

    static inline size_t
    npow_getN_(const size_t N, const size_t acc) {
      if (N > 1)
	return npow_getN_(N>>1, acc+1);
      else
	return acc;  
    }
    
    static inline size_t
    npow_getN(const size_t N) {
      return npow_getN_(N, 0);
    }
    
    template<size_t _N>
    struct npow {
      enum { N = npow<_N>>1>::N + 1L };
      static inline size_t getN() { return (size_t) N; } 
    };
    
    template<>
    struct npow<1> {
      enum { N = 0L };
      static inline size_t getN() { return (size_t) N; } 
    };

    static inline void
    generate_twiddleidxs_rec(const size_t half_N,
			     size_t * const t_arr,
			     size_t * const t2_arr) {
    
      if (half_N == 2) {
	return;
      }

      const size_t half_half_N = half_N >> 1;
      
#ifdef _DEBUG      
      printf("half_N: %lu\n", half_N);
#endif
      
      for (size_t i = 0, j = 0; i < half_N; ++i) {
	const size_t i_mod_2 = i % 2;
	t2_arr[j + (i_mod_2 * half_half_N)] = t_arr[i];
	j += i_mod_2;
      }
      
      memcpy(t_arr, t2_arr, sizeof(size_t)*half_N);
      
#ifdef _DEBUG      
      for (size_t i = 0; i < half_N; ++i) 
	printf("t_arr[%lu]: %lu\n", i, t_arr[i]);
#endif
      
      generate_twiddleidxs_rec(half_half_N, 
			       t_arr, 
			       t2_arr);	

      generate_twiddleidxs_rec(half_half_N, 
			       t_arr + half_half_N, 
			       t2_arr + half_half_N);	
    }

    // this isn't actually used, but it shows how we walk down the 
    // twiddle tree within the actual FFT code
    static inline void
    treeify(const size_t N,
	    const size_t level_size,
	    size_t * const level_base,
	    const size_t offset,
	    size_t * const output_array,
	    size_t * const output_array_idx) {

      if (N == level_size)
	return;

      const size_t next_level_size = level_size << 1;
      const size_t next_offset = offset << 1;
      size_t * const next_level = level_base + level_size;

      {
	const size_t idx = *output_array_idx;
	output_array[idx] = level_base[offset];
	*output_array_idx = idx + 1;
      }

      treeify(N, next_level_size, next_level, next_offset,
	      output_array, output_array_idx);

      {
	const size_t idx = *output_array_idx;
	output_array[idx] = level_base[offset+1];
	*output_array_idx = idx + 1;
      }

      treeify(N, next_level_size, next_level, next_offset + 2,
	      output_array, output_array_idx);
    }
    
    template <typename val_t, size_t N_stop>
    static inline void
    gen_twiddles(const size_t N,
		 val_t * twiddle_array_p) {

      val_t * twiddle_array_base = twiddle_array_p;
      const size_t half_N = N >> 1;

      // build the actual twiddle values
      std::complex<val_t> * const twiddle_val_arr = 
	(std::complex<val_t>*) malloc(sizeof(std::complex<val_t>) * N);
      
      const val_t mval = M_PI * 2.0f / ((float) N);
      
      for (size_t i = 0; i < N; ++i) {
	std::complex<val_t>& t = twiddle_val_arr[i];
	const val_t theta = ((val_t) i) * mval;
	t.real() = cosf(theta);
	t.imag() = -sinf(theta);
      }

      // allocate the scratch array to hold the indices      
      size_t * const t_arr = (size_t*) malloc(sizeof(size_t) * (N << 1));
      memset(t_arr, 0, sizeof(size_t) * (N << 1));

      // the first run is N >> 3
      if (N > (N_stop << 1)) {      
	const size_t i1 = N >> 3;
	
	for (size_t j = 0; j < i1; ++j) 
	  t_arr[j] = j * (half_N / i1);
	
	generate_twiddleidxs_rec(i1,
				 t_arr,
				 t_arr + i1);
	
	// write the float values into the twiddle array
	for (size_t i = 0; i < i1; ++i) {
	  for (size_t j = 0; j < 4; ++j) {
	    
	    const size_t i_p_j = (i * N_stop) + j;

	    std::complex<val_t>& t = twiddle_val_arr[t_arr[i]]; 
	    
#ifdef _DEBUG
	    printf("t_arr[%lu] - 1: (%+9f, %+9f)\n", i+j, t.real(), t.imag());
#endif // _DEBUG

	    twiddle_array_base[i_p_j] = t.real(); 
	    twiddle_array_base[i_p_j + (N_stop >> 1)] = t.imag(); 
	  }
	}
	
	twiddle_array_base += (i1 * N_stop);
      }

      // second run is N >> 1
      {
	memset(t_arr, 0, sizeof(size_t) * (N << 1));
	const size_t i2 = N >> 2;
	
	for (size_t j = 0; j < i2; ++j) 
	  t_arr[j] = j * (half_N / i2);
	
	generate_twiddleidxs_rec(i2,
				 t_arr,
				 t_arr + i2);
	
	// write the float values into the twiddle array
	// note that we're leaving space for the last run's values, as they're
	// interspersed with this run's
	for (size_t i = 0, ii = 0; i < i2; ++i) {

	  ii += ((i > 0) && ((i % 2) == 0)) ? (N_stop << 1) : 0;

	  for (size_t j = 0; j < 2; ++j) {
	    
	    const size_t ii_p_j = ii + ((i % 2) << 1) + j;	    
	    std::complex<val_t>& t = twiddle_val_arr[t_arr[i]]; 
	    
#ifdef _DEBUG	  
	    printf("i: %lu j: %lu t_arr[%lu] - 2: (%+9f, %+9f)\n", 
		   i, j,
		   ii_p_j,
		   t.real(), 
		   t.imag());
#endif // _DEBUG
	    
	    twiddle_array_base[ii_p_j] = t.real(); 
	    twiddle_array_base[ii_p_j + (N_stop >> 1)] = t.imag(); 
	  }
	}
      }

      twiddle_array_base += N_stop;
      
      // third run is N
      {
	memset(t_arr, 0, sizeof(size_t) * (N << 1));
	const size_t i3 = N >> 1;
	
	for (size_t j = 0; j < i3; ++j) 
	  t_arr[j] = j * (half_N / i3);
	
	generate_twiddleidxs_rec(i3,
				 t_arr,
				 t_arr + i3);
	
	// write the float values into the twiddle array
	for (size_t i = 0, ii = 0; i < i3; ++i) {

	  ii += ((i > 0) && ((i % 4) == 0)) ? (N_stop << 1) : 0;
	    
	  const size_t i_p_j = ii + (i % 4);
	  
	  std::complex<val_t>& t = twiddle_val_arr[t_arr[i]]; 
	  
#ifdef _DEBUG	  
	  printf("t_arr[%lu] - 3: (%+9f, %+9f)\n", i_p_j, t.real(), t.imag());
#endif // _DEBUG 
	  
	  twiddle_array_base[i_p_j] = t.real(); 
	  twiddle_array_base[i_p_j + (N_stop >> 1)] = t.imag(); 
	}
      }

      free (t_arr);
    }
  }
}

#endif // _YAFFT_TWIDDLES_HPP_
