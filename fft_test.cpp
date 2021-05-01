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

#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <sys/time.h>

#include <fftw3.h>
#include <iostream>

#include <xmmintrin.h>
#include <complex>

#include "fft_test_defines.hpp"

#include "yafft.hpp"

std::complex<float> *data; 
std::complex<float> *yafft2_input_data;
std::complex<float> *yafft3_input_data;
std::complex<float> *yafft4_input_data;
std::complex<float> *yafft7_input_data;
std::complex<float> *yafft_input_data;
float * yafft2_twiddles;
float * yafft2_work;
fftwf_complex *input_data;

fftwf_complex *output_data;

/*
typedef aahmath::funion_t cpx_m128_t;

static inline void cpx_m128_mult(cpx_m128_t& rin1, cpx_m128_t& iin1,
                                 cpx_m128_t& rin2, cpx_m128_t& iin2,
                                 cpx_m128_t& rout, cpx_m128_t& iout) {
  rout.fvec = _mm_sub_ps(_mm_mul_ps(rin1.fvec, rin2.fvec),
                          _mm_mul_ps(iin1.fvec, iin2.fvec));
  iout.fvec = _mm_add_ps(_mm_mul_ps(rin1.fvec, iin2.fvec),
                          _mm_mul_ps(iin1.fvec, rin2.fvec));
}
*/
void print_input() {
  for (int i = 0; i < DATA_SZ; ++i) 
    printf("(%+9f %+9f)\n",
           data[i].real(), data[i].imag());
}

void print_output() {
  float *f_p = (float*) output_data;
  
  for (int i = 0; i < (DATA_SZ<<1); i += 2)
    printf("(%+f, %+f)\n",  f_p[i], f_p[i+1]);     
}

long gettime() {
  timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec*1000000+tv.tv_usec);
}

void init_storage() {
  const unsigned int data_sz = sizeof(float)*DATA_SZ*2;

  void *v_p;
  yafft::utils::alloc_aligned(&v_p, 16, data_sz);
  data = (std::complex<float>*) v_p;
  ::memset(data, 0, data_sz);

  const float dphasor = 2.0f * M_PI / (float) DATA_SZ;
  float phasor = 0.0f;

  const float rand_mult = 1.0f / ((float) RAND_MAX);

  for (int i = 0; i < DATA_SZ; i++) { 

    const float r = (((((float) rand()) * rand_mult) - 0.5f) * 0.1f);

    data[i].real() = (0.6f*sinf(phasor))+(0.3f*sinf(3.0f*phasor)) + r;
    data[i].imag() = cosf(phasor);

    phasor += dphasor;
  }

  yafft::utils::alloc_aligned(&v_p, 16, data_sz);
  yafft4_input_data = (std::complex<float>*) v_p;
  ::memset(yafft4_input_data, 0, data_sz);

  yafft::utils::alloc_aligned(&v_p, 16, data_sz);
  yafft7_input_data = (std::complex<float>*) v_p;
  ::memset(yafft7_input_data, 0, data_sz);

  yafft::utils::alloc_aligned(&v_p, 16, data_sz);
  yafft_input_data = (std::complex<float>*) v_p;
  ::memset(yafft_input_data, 0, data_sz);

  yafft::utils::alloc_aligned(&v_p, 16, data_sz);
  output_data = (fftwf_complex*) v_p;
  ::memset(output_data, 0, data_sz);

  yafft::utils::alloc_aligned(&v_p, 16, data_sz);
  input_data = (fftwf_complex*) v_p;
  ::memset(input_data, 0, data_sz);
}

void init_data() {
  const size_t data_sz = sizeof(std::complex<float>)*DATA_SZ;
  ::memset(output_data, 0, data_sz);
  
  ::memcpy(input_data, data, data_sz); 
  ::memcpy(yafft4_input_data, data, data_sz);
  ::memcpy(yafft7_input_data, data, data_sz);
  ::memcpy(yafft_input_data, data, data_sz);
}

fftwf_plan init_fftw() {
  
  init_data();
  
  return fftwf_plan_dft_1d(DATA_SZ, 
			   input_data, 
			   output_data,
			   FFTW_FORWARD,
			   FFTW_PLAN_STRAT);
}

fftwf_plan init_backward_fftw() {

  return fftwf_plan_dft_1d(DATA_SZ, 
			   input_data, 
			   output_data,
			   FFTW_BACKWARD,
			   FFTW_PLAN_STRAT);
}

static int run_fftwf_time = 0;

void run_both_fftwf(const fftwf_plan& fplan, 
		    const fftwf_plan& rplan,
		    const bool print) {
  
  init_data();

  long start, end;
  
  start = gettime();
  for (int i = 0; i < INNERLOOP; ++i)
    fftwf_execute(fplan);
  end = gettime();

  if (print) {
    printf("fftwf forward out\n");
    print_output();
  }

  run_fftwf_time += end - start;

  memcpy(input_data, output_data, sizeof(float)*DATA_SZ*2);
  //  memcpy(yafft2_input_data, output_data, sizeof(float)*DATA_SZ*2);

  start = gettime();
  for (int i = 0; i < INNERLOOP; ++i)
    fftwf_execute(rplan);
  end = gettime();

  if (print) {
    printf("fftwf reverse out\n");
    print_output();
  }

  run_fftwf_time += end - start;
}

void run_fftwf(const fftwf_plan& plan, bool print) {

  init_data();

  //  print_input();

  long start = gettime();
  fftwf_execute(plan);
  long end = gettime();

  run_fftwf_time += end - start;

  /*
  if (print) {
    printf("run_fftw: %ld\n", end - start);
    print_output();
  }
  */
  return;
}

void naive_fft_4pt() {

  float r[DATA_SZ], i[DATA_SZ];
  float r_tmp[DATA_SZ], i_tmp[DATA_SZ];

  ::memcpy(r, data, sizeof(float)*DATA_SZ);
  ::memset(i, 0, sizeof(float)*DATA_SZ);
  ::memset(r_tmp, 0, sizeof(float)*DATA_SZ);
  ::memset(i_tmp, 0, sizeof(float)*DATA_SZ);

  const float w_0_r = cosf(0);
  const float w_0_i = -sinf(0);
  const float w_2_r = cosf(M_PI/2.0f);
  const float w_2_i = -sinf(M_PI/2.0f);

  long start = gettime();
  
  const float r0_p_r2 = r[0] + r[2];
  const float i0_p_i2 = i[0] + i[2];

  const float r1_p_r3 = r[1] + r[3];
  const float i1_p_i3 = i[1] + i[3];

  const float r0_m_r2 = r[0] - r[2];
  const float i0_m_i2 = i[0] - i[2];

  const float r1_m_r3 = r[1] - r[3];
  const float i1_m_i3 = i[1] - i[3];

  const float r1_p_r3_w = r1_p_r3 * w_0_r - i1_p_i3 * w_0_i;
  const float i1_p_i3_w = r1_p_r3 * w_0_i + i1_p_i3 * w_0_r;
  const float r1_m_r3_w = r1_m_r3 * w_2_r - i1_m_i3 * w_2_i;
  const float i1_m_i3_w = r1_m_r3 * w_2_i + i1_m_i3 * w_2_r;

  output_data[0][0] = r0_p_r2 + r1_p_r3_w;
  output_data[0][1] = i0_p_i2 + i1_p_i3_w;

  output_data[1][0] = r0_m_r2 + r1_m_r3_w;
  output_data[1][1] = i0_m_i2 + i1_m_i3_w;

  output_data[2][0] = r0_p_r2 - r1_p_r3_w;
  output_data[2][1] = i0_p_i2 - i1_p_i3_w;

  output_data[3][0] = r0_m_r2 - r1_m_r3_w;
  output_data[3][1] = i0_m_i2 - i1_m_i3_w;

  long end = gettime();
  printf("naive_fft_4pt: %ld\n", end - start);
}

//static int slightly_less_naive_time = 0;

void slightly_less_naive() { 
  /*
  const int N = DATA_SZ;
  const int N_sse = N / 4;
  aahmath::funion_t r_data_arr[N_sse] __attribute__((aligned(16)));
  aahmath::funion_t i_data_arr[N_sse] __attribute__((aligned(16)));
  
  aahmath::funion_t r_tmp_arr[N_sse] __attribute__((aligned(16)));
  aahmath::funion_t i_tmp_arr[N_sse] __attribute__((aligned(16)));
  
  // check to see if it's faster to blast using _mm_setzero_ps()
  ::memcpy(r_data_arr, data, sizeof(cpx_m128_t)*N_sse);
  ::memset(i_data_arr, 0, sizeof(cpx_m128_t)*N_sse);
  ::memset(r_tmp_arr, 0, sizeof(cpx_m128_t)*N_sse);
  ::memset(i_tmp_arr, 0, sizeof(cpx_m128_t)*N_sse);
  
  init_data();
  //  print_input();
  //    print_output();
  
  long start = gettime();
  
  aahfft::cpx_fft2<aahfft::cpx_mult,
    aahfft::std_complex_unswizzle2, 
    aahfft::print_data_null>
    (N,
     aahfft::get_twiddle_bases(N),
     (aahmath::funion_t*) r_data_arr, 
      (aahmath::funion_t*) i_data_arr,
     (aahmath::funion_t*) r_tmp_arr, 
     (aahmath::funion_t*) i_tmp_arr,
     (float*) output_data
     );
  
  long end = gettime();

  slightly_less_naive_time += end - start;
  
  //  printf("slightly_less_naive_fft<%d>: %ld\n", N, end - start);  
  
  //  print_output();
  */
}

//static int slightly_less_naive2_time = 0;

void slightly_less_naive2() {
  /*
  const int N = DATA_SZ;
  const int N_sse = N / 4;
  aahmath::funion_t r_data_arr[N_sse] __attribute__((aligned(16)));
  aahmath::funion_t i_data_arr[N_sse] __attribute__((aligned(16)));
  
  aahmath::funion_t r_tmp_arr[N_sse] __attribute__((aligned(16)));
  aahmath::funion_t i_tmp_arr[N_sse] __attribute__((aligned(16)));
  
  // check to see if it's faster to blast using _mm_setzero_ps()
  ::memcpy(r_data_arr, data, sizeof(cpx_m128_t)*N_sse);
  ::memset(i_data_arr, 0, sizeof(cpx_m128_t)*N_sse);
  ::memset(r_tmp_arr, 0, sizeof(cpx_m128_t)*N_sse);
  ::memset(i_tmp_arr, 0, sizeof(cpx_m128_t)*N_sse);
  
  init_data();
  //  print_input();
  //    print_output();
  
  long start = gettime();
  
  aahfft::cpx_fft3<aahfft::cpx_mult,
    aahfft::std_complex_unswizzle2, 
    aahfft::print_data_null>
    (N,
     aahfft::get_twiddle_bases(N),
     (aahmath::funion_t*) r_data_arr, 
     (aahmath::funion_t*) i_data_arr,
     (aahmath::funion_t*) r_tmp_arr, 
     (aahmath::funion_t*) i_tmp_arr,
     (float*) output_data
     );
  
  long end = gettime();

  slightly_less_naive2_time += end - start;
  */  
  //  printf("slightly_less_naive2_fft<%d>: %ld\n", N, end - start);  
  //  print_output();
}

//static int slightly_less_naive3_time = 0;

void print_arr_null(const char * const lbl, const int N, 
		    const float * const r_data, 
		    const float * const i_data) {}

//static int slightly_less_naive4_time = 0;

template<class iter_t, 
	 const iter_t N, 
	 template <class _iter_t, _iter_t _N> class T_fft_impl>
struct yafft_wrapper {

  void 
  fn(typename T_fft_impl<iter_t, N>::t& cpxfft, int& time) {
    
    std::pair<float * const, float * const> data_p =
      cpxfft.get_in_blocks();
    
    init_data();
    
    // check to see if it's faster to blast using _mm_setzero_ps()
    ::memset(data_p.first, 0, sizeof(float)*N*4);

    for (int i = 0; i < DATA_SZ; ++i) {
      data_p.first[i] = data[i].real();
      data_p.first[i+DATA_SZ] = data[i].imag();
    }

    //    ::memcpy(data_p.first, data, sizeof(float)*N);
    
    long start = gettime();
    
    cpxfft.fft((float*) output_data);
    
    long end = gettime();
    
    time += end - start;

    /*
    printf("yafft_wrapper: %ld\n", end - start);
    print_output();
    */
  }
};

template<const int N, 
	 template <const int _N> class T_fft_impl>
struct slightly_less_naive4_wrapper {

  void 
  fn(typename T_fft_impl<N>::t& cpxfft, int& time) {
    
    std::pair<float * const, float * const> data_p =
      cpxfft.get_in_blocks();
    
    init_data();
    
    // check to see if it's faster to blast using _mm_setzero_ps()
    ::memset(data_p.first, 0, sizeof(float)*N*4);

    for (int i = 0; i < DATA_SZ; ++i) {
      data_p.first[i] = data[i].real();
      data_p.first[i+DATA_SZ] = data[i].imag();
    }

    //    ::memcpy(data_p.first, data, sizeof(float)*N);
    //    ::memcpy(data_p.first, data, sizeof(float)*N);
    
    long start = gettime();
    
    cpxfft.fft((float*) output_data);
    
    long end = gettime();
    
    time += end - start;    

    printf("fft_wrapper: %ld\n", end - start);
    print_output();
  }
};

//static int template_less_naive_time = 0;

/*
void template_less_naive() {
  init_data();
  
  yafft::sp::cpx_t<DATA_SZ> cpx_holder;
  memset(cpx_holder.get_block(yafft::sp::cpx_t<DATA_SZ>::rin1), 
	 0,
	 sizeof(float)*DATA_SZ*4);
  
  memset(output_data,
	 0,
	 sizeof(output_data)); */
  /*    
	printf("output\n");
	print_output();
  *//*
  memcpy(cpx_holder.get_block(yafft::sp::cpx_t<DATA_SZ>::rin1),
	 data,
	 sizeof(float)*DATA_SZ);
  
  //  printf("starting cpxfft_aos\n");
  
  long start = gettime();
  
  yafft::sp::cpxfft_aos<DATA_SZ, (DATA_SZ>>2)>(cpx_holder, 
					       (float*)output_data);
  
  long end = gettime();
  
  //  printf("ending cpxfft_aos\n");
  
  template_less_naive_time += end - start;

  printf("template_less_naive<%d>: %ld\n", DATA_SZ, end - start);  
  
  //  print_output();
}
*/

void
gen_yafft2_twiddles(float * const twiddle_array) {
  const float one_o_N = 1.0f / (float) DATA_SZ;
  const float two_pi = M_PI * 2.0f;

  std::complex<float> * twiddles = 
    (std::complex<float>*) malloc(sizeof(float)*DATA_SZ);
  
  for (size_t i = 0; i < (DATA_SZ>>1); ++i) {
    const float theta = two_pi * i * one_o_N;
    twiddles[i].real() = cosf(theta);
    twiddles[i].imag() = -sinf(theta);

    printf("%u: (%+9f, %+9f)\n", i, twiddles[i].real(), twiddles[i].imag());
  }

  size_t * twiddle_idxs = (size_t*) malloc(sizeof(size_t) * 
					   (DATA_SZ>>1) * 3);
  
  size_t twiddle_array_idx = 0;
  size_t twiddle_idx_arr[] = { 0, 16, 8, 24, 4, 20, 12, 28 };
  
  for (size_t i = 0; i < DATA_SZ >> 3; ++i) {
    for (size_t j = 0, jj = 4, jjj = DATA_SZ >> 1; j < 3;
	 jj >>= 1, jjj >>= 1, ++j) {
      for (size_t k = 0, kk = 0; k < (4/jj); ++k, kk += jjj) {
	for (size_t l = 0; l < jj; ++l, ++twiddle_array_idx) {
	  twiddle_idxs[twiddle_array_idx] = (twiddle_idx_arr[i] >> j) + kk; 
	}
      }
    }
  }
  
  for (size_t i = 0; i < ((DATA_SZ>>1) * 3); ++i) {
    printf("idx[%u]: %u\n", i, twiddle_idxs[i]);
  }
  
  for (size_t i = 0, ii = 0; i < ((DATA_SZ>>1) * 6); i+=8) {
    for (size_t j = 0; j < 4; ++j, ++ii) {
      twiddle_array[i+j] = twiddles[twiddle_idxs[ii]].real();
      twiddle_array[i+j+4] = twiddles[twiddle_idxs[ii]].imag();
    }
  }

  for (size_t i = 0; i < ((DATA_SZ>>3) * 24); ++i) {
    printf("idx[%u]: %+f\n", i, twiddle_array[i]);
  }
  
  free(twiddle_idxs);
  free(twiddles);
}


static long yafft7_time = 0;
static long yafft_time = 0;

/*
template <size_t N>
void
run_yafft2(typename yafft2::fft_float<N>::t& yafft2,
	   std::complex<float> * const in,
	   std::complex<float> * const out) {

  long start, end;

  start = gettime();
  //__asm__("yafft2start:");
  yafft2.run(in, out, yafft2::FORWARD);
  //__asm__("yafft2end:");
  end = gettime();

  yafft2_time += end - start;
}

template <size_t N>
void
run_both_yafft2(typename yafft2::fft_float<N>::t& yafft2,
		std::complex<float> * const in,
		std::complex<float> * const out,
		const bool print) {

  //  init_data();

  long start, end;

  //  ::memset(output_data, 0, (sizeof(fftwf_complex)*DATA_SZ));

  start = gettime();
  //__asm__("yafft2fstart:");
  yafft2.run(in, out, yafft2::FORWARD);
  //__asm__("yafft2fend:");
  end = gettime();

  yafft2_time += end - start;

  
  if (print) {
    printf("yafft2 forward out\n");
    for (size_t i = 0; i < N; ++i) {
      std::complex<float>& out_r = out[i];
      printf("(%+9f, %+9f)\n", out_r.real(), out_r.imag());
    }
  }
  

  ::memcpy(in, out, sizeof(std::complex<float>)*N);
  //  ::memset(output_data, 0, (sizeof(fftwf_complex)*DATA_SZ));

  start = gettime();
  __asm__("yafft2rstart:");
  yafft2.run(in, out, yafft2::FORWARD);
  __asm__("yafft2rend:");
  end = gettime();

  yafft2_time += end - start;

  
  if (print) {
    printf("yafft2 reverse out\n");
    for (size_t i = 0; i < N; ++i) {
      std::complex<float>& out_r = out[i];
      printf("(%+9f, %+9f)\n", out_r.real(), out_r.imag());
    }
  } 
  
}
*/

yafft::iyafft_float_ptr yafft_ptr = yafft::get_fft<float>(DATA_SZ);

long run_both_yafft(std::complex<float> * const in,
		     std::complex<float> * const out,
		     const bool print) {

  long forward_time, reverse_time;

#ifdef _DEBUG
  printf("yafft start forward\n");
  ::memset(out, 0, sizeof(std::complex<float>)*DATA_SZ);
#endif

  const long start_forward = gettime();
  for (size_t i = 0; i < INNERLOOP; ++i)
    yafft_ptr->run_forward(in, out);
  const long end_forward = gettime();

  forward_time = end_forward - start_forward;

  if (print) {
    printf("yafft forward out\n");
    for (size_t i = 0; i < DATA_SZ; ++i) 
      printf("(%+9f, %+9f)\n", out[i].real(), out[i].imag());
  }

  ::memcpy(in, out, sizeof(std::complex<float>)*DATA_SZ);

#ifdef _DEBUG
  printf("yafft start reverse\n");
  ::memset(out, 0, sizeof(std::complex<float>)*DATA_SZ);
#endif

  const long start_reverse = gettime();
  for (size_t i = 0; i < INNERLOOP; ++i)
    yafft_ptr->run_reverse(in, out);
  const long end_reverse = gettime();

  if (print) {
    printf("yafft reverse out\n");
    for (size_t i = 0; i < DATA_SZ; ++i) 
      printf("(%+9f, %+9f)\n", out[i].real(), out[i].imag());
  }

  reverse_time = end_reverse - start_reverse;

  return forward_time + reverse_time;
}

int main(int argc, const char **argv) {
  
  //  aahfft::init_twiddles();
  
  //  aahfft::print_twiddles();
  
  //  const std::pair<float*, float*> base = aahfft::get_twiddle_base(64);
  //  std::cout <<  base.first << " " << base.second << std::endl;

  float a = 1.0f;

  for (int i = 0; i < 10000000; i++) 
    a = sqrtf(a);

  printf("a: %+f\n", a);

  //  aahfft::generate_twiddles();
  
  init_storage();
  init_data();
  //  print_input();

  //  gen_yafft2_twiddles(yafft2_twiddles);

  /*
  yafft2::impl::generate_twiddles<yafft::float_t::sse::ops,
    DATA_SZ, 
    yafft::float_t::sse::ops::STOP,
    1>::fn(reinterpret_cast<float*>(yafft2_twiddles));
  */

  //  print_output();

  //  init_data();

  //  naive_fft_1();

  //print_output();
  
  //  init_data();

  //  naive_fft_4pt();

  //  init_data();
  
  //naive_fft_8pt();

  /*
  {
    init_data();
    //    print_output();
    
    naive_fft_16pt((float*) output_data);
    
    print_output();
  }

  int fft1_time = 0;
  int fft2_time = 0;
  int yafft3_time = 0;
  */

  /*
  yafft::cpxf_impl<long, DATA_SZ>::t yafft;
  yafft_wrapper<long, DATA_SZ, yafft::cpxf_impl> yafft_wrapper;

  fft_util::cpxf2_impl<DATA_SZ>::t cpxfft2;
  slightly_less_naive4_wrapper<DATA_SZ, fft_util::cpxf2_impl> fft2_wrapper;

  fft_util::cpxf2_impl2_debug<DATA_SZ>::t cpxfft3;
  slightly_less_naive4_wrapper<DATA_SZ, 
    fft_util::cpxf2_impl2_debug> fft3_wrapper;
  */

  fftwf_plan fplan = init_fftw();
  fftwf_plan rplan = init_backward_fftw();
  printf("completed fftw initialization\n");

  /*
  yafft2::impl::std_complex_fft<yafft::float_t::sse::ops, 
    yafft::float_t::sse::ops::print_noops, 
    yafft::float_t::sse::ops::stdcpx_ops,
    yafft::float_t::sse::ops::unswizzle_ops,
    DATA_SZ, 
    yafft::float_t::sse::ops::STOP> yafft2_fft;
  */

  yafft_ptr->init();

  const long iters = NITERS;
  const bool print = PRINT;

  // prime the code into cache;
  run_both_yafft(yafft_input_data,
		  reinterpret_cast<std::complex<float>*>(output_data),
		  print);
  printf("completed yafft initialization\n");

  for (long i = 0; i < iters; ++i) {
    
    run_both_fftwf(fplan, rplan, print);

    /*    
    __asm__("yafft:");
    yafft_wrapper.fn(yafft, fft1_time);
    */

    yafft_time +=
      run_both_yafft(yafft_input_data,
		      reinterpret_cast<std::complex<float>*>(output_data),
		      print);
  }

  const float scale = 1.0f / ((float) NITERS * INNERLOOP);

  printf("N: %d fftw: %f yafft7: %f yafft8: %f\n",
	 DATA_SZ, 
	 ((float) run_fftwf_time * scale),
	 ((float) yafft7_time * scale),
	 ((float) yafft_time * scale));

  fftwf_destroy_plan(fplan);
  fftwf_destroy_plan(rplan);
  
  //  aahfft::free_twiddles();

  return 0;
}
