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

#include <fftw3.h>
#include <iostream>
#include <sys/time.h>

#include "yafft8.hpp"

#define FFT_SIZE 12347
#define FFT_SIZE_POW2 32768
#define NLOOPS 100

static inline long
gettime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

typedef std::complex<float> cpx_float_t;

void init_input2(cpx_float_t * const input) {

  const float dphasor = 2.0f * M_PI / (float) FFT_SIZE;
  float phasor = 0.0f;

  for (size_t i = 0; i < FFT_SIZE; ++i) {
    input[i].real() = (0.6f*sinf(phasor))+(0.3f*sinf(3.0f*phasor));
    input[i].imag() = 0.0; // cosf(phasor);

    input[i] *= 0.5f;

    phasor += dphasor;
  }

  for (size_t i = FFT_SIZE; i < FFT_SIZE_POW2; ++i) {
    input[i].real() = 0.0f;
    input[i].imag() = 0.0f;
  }
}

void init_output2(cpx_float_t * const output) {
  memset(output, 0, sizeof(float)*FFT_SIZE_POW2);
}


void init_input(fftwf_complex * const input) {

  const float dphasor = 2.0f * M_PI / (float) FFT_SIZE;
  float phasor = 0.0f;

  for (size_t i = 0; i < FFT_SIZE; ++i) {
    input[i][0] = (0.6f*sinf(phasor))+(0.3f*sinf(3.0f*phasor));
    input[i][1] = 0.0; //cosf(phasor);

    input[i][0] *= 0.5f;
    input[i][1] *= 0.5f;

    phasor += dphasor;
  }

  for (size_t i = FFT_SIZE; i < FFT_SIZE_POW2; ++i) {
    input[i][0] = 0.0f;
    input[i][1] = 0.0f;
  }
}

void init_output(fftwf_complex * const output) {
  memset(output, 0, sizeof(fftwf_complex)*FFT_SIZE_POW2);
}

void run_zchirp() {

  cpx_float_t input_data[FFT_SIZE_POW2] __attribute__((aligned(16)));
  cpx_float_t output_data[FFT_SIZE_POW2] __attribute__((aligned(16)));
  
  void * v_p = NULL;

  posix_memalign(&v_p, 64, sizeof(std::complex<float>)*FFT_SIZE_POW2);
  std::complex<float> * const chirp1 = static_cast<std::complex<float>*>(v_p);

  posix_memalign(&v_p, 64, sizeof(std::complex<float>)*FFT_SIZE_POW2);
  std::complex<float> * const chirp2 = static_cast<std::complex<float>*>(v_p);

  posix_memalign(&v_p, 64, sizeof(std::complex<float>)*FFT_SIZE_POW2);
  std::complex<float> * const chirp3 = static_cast<std::complex<float>*>(v_p);

  memset(chirp1, 0, sizeof(std::complex<float>)*FFT_SIZE_POW2);
  memset(chirp2, 0, sizeof(std::complex<float>)*FFT_SIZE_POW2);
  memset(chirp3, 0, sizeof(std::complex<float>)*FFT_SIZE_POW2);
  
  const float chirpsig_base = M_PI / ((float) FFT_SIZE);
  const float scale = 1.0f / FFT_SIZE_POW2;
  
  for (size_t i = 0; i < FFT_SIZE; ++i) {

    const float i_f = (float) i;
    const float n_sq = i_f * i_f;
    const float chirpsig = chirpsig_base * n_sq;

    chirp1[i].real() = chirp2[i].real() = cosf(chirpsig);
    chirp2[i].imag() = sinf(chirpsig);
    chirp1[i].imag() = -chirp2[i].imag();
    
    chirp3[i] = chirp1[i] * scale;

    if (i != 0) {
      chirp2[FFT_SIZE_POW2 - i] = chirp2[i];
    }
  }
 
  for (size_t i = 0; i < FFT_SIZE_POW2; ++i) {
    printf("chirp1[%lu]: (%+9f, %+9f)\n", i, 
           chirp1[i].real(), chirp1[i].imag());
  }

  for (size_t i = 0; i < FFT_SIZE_POW2; ++i) {
    printf("chirp2[%lu]: (%+9f, %+9f)\n", i, 
           chirp2[i].real(), chirp2[i].imag());
  }

  for (size_t i = 0; i < FFT_SIZE_POW2; ++i) {
    printf("chirp3[%lu]: (%+9f, %+9f)\n", i, 
           chirp3[i].real(), chirp3[i].imag());
  }

  boost::shared_ptr<iyafft<float> > fft_ptr = 
    yafft8::get_fft<float>(FFT_SIZE_POW2);

  fft_ptr->init();

  init_input2(input_data);

  std::complex<float> fft_input_data1[FFT_SIZE_POW2];
  std::complex<float> fft_input_data2[FFT_SIZE_POW2];
  std::complex<float> fft_input_data3[FFT_SIZE_POW2];

  memset(fft_input_data1, 0, sizeof(std::complex<float>)*FFT_SIZE_POW2);
  memset(fft_input_data2, 0, sizeof(std::complex<float>)*FFT_SIZE_POW2);
  memset(fft_input_data3, 0, sizeof(std::complex<float>)*FFT_SIZE_POW2);

  fft_ptr->run_forward(chirp2, fft_input_data2);
 
  for (size_t i = 0; i < FFT_SIZE_POW2; ++i) {
    printf("post-fft chirp2[%lu]: (%+9f, %+9f)\n", i, 
           fft_input_data2[i].real(), fft_input_data2[i].imag());
  }

  __asm__ volatile("chirpzs:");
  
  for (size_t i = 0; i < FFT_SIZE; ++i) {
    fft_input_data3[i] = chirp1[i] * input_data[i];
  }
  
  for (size_t i = 0; i < FFT_SIZE_POW2; ++i) {
    printf("fft_input_data3[%lu]: (%+9f, %+9f)\n", i, 
           fft_input_data3[i].real(), fft_input_data3[i].imag());
  }

  fft_ptr->run_forward(fft_input_data3, fft_input_data1);
  
  for (size_t i = 0; i < FFT_SIZE_POW2; ++i) {
    printf("post-fft fft_input_data1[%lu]: (%+9f, %+9f)\n", i, 
           fft_input_data1[i].real(), fft_input_data1[i].imag());
  }

  for (size_t i = 0; i < FFT_SIZE_POW2; ++i) {
    fft_input_data1[i] *= fft_input_data2[i];
  }
  
  for (size_t i = 0; i < FFT_SIZE_POW2; ++i) {
    printf("post-mult fft_input_data1[%lu]: (%+9f, %+9f)\n", i, 
           fft_input_data1[i].real(), fft_input_data1[i].imag());
  }

  fft_ptr->run_reverse(fft_input_data1, output_data);
  
  for (size_t i = 0; i < FFT_SIZE_POW2; ++i) {
    printf("post-fft output_data[%lu]: (%+9f, %+9f)\n", i, 
           output_data[i].real(), output_data[i].imag());
  }

  for (size_t i = 0; i < FFT_SIZE; ++i) {
    printf("multiplying (%+9f, %+9f) with (%+9f, %+9f)\n",
	   output_data[i].real(), output_data[i].imag(),
	   chirp3[i].real(), chirp3[i].imag());

    output_data[i] *= chirp3[i];
  }
  
  __asm__ volatile("chirpze:");
  
  for (size_t i = 0; i < FFT_SIZE_POW2; ++i) {
    printf("zchirp output_data[%lu]: (%+9f, %+9f)\n", i, 
           output_data[i].real(), output_data[i].imag());
  } 
}  

int main(int argc, char ** argv) {
  /*
  {
    const long start = gettime();
    for (size_t i = 0; i < NLOOPS; ++i) 
      run_zchirp();
    const long elapsed = gettime() - start;
    printf("time taken: %ld usec\n", elapsed);
  }
  */
  {
    fftwf_complex input_data[FFT_SIZE_POW2] __attribute__((aligned(16)));
    fftwf_complex output_data[FFT_SIZE_POW2] __attribute__((aligned(16)));
    
    fftwf_plan theplan = fftwf_plan_dft_1d(FFT_SIZE,
					   input_data,
					   output_data,
					   FFTW_FORWARD,
					   FFTW_MEASURE);
    init_input(input_data);
    init_output(output_data);
    /*
    for (size_t i = 0; i < FFT_SIZE_POW2; ++i) {
      printf("fftw input_data[%lu]: (%+9f, %+9f)\n", i, 
	     input_data[i][0], input_data[i][1]);
    } 
    */
    const long start = gettime();
    for (size_t i = 0; i < NLOOPS; ++i)
      fftwf_execute(theplan);
    const long elapsed = gettime() - start;
    printf("time taken: %ld usec\n", elapsed);
    /*
    for (size_t i = 0; i < FFT_SIZE_POW2; ++i) {
      printf("fftw output_data[%lu]: (%+9f, %+9f)\n", i, 
	     output_data[i][0], output_data[i][1]);
    } 
    std::cout << std::endl;
    */
  }

  {
    cpx_float_t input_data[FFT_SIZE_POW2] __attribute__((aligned(16)));
    cpx_float_t output_data[FFT_SIZE_POW2] __attribute__((aligned(16)));
    
    boost::shared_ptr<iyafft<float> > fft_ptr = 
      yafft8::get_fft<float>(FFT_SIZE);

    if (!fft_ptr->init()) {
      printf("unable to properly initialize fft_ptr, exiting\n");
      return -1;
    }

    init_input2(input_data);
    init_output2(output_data);
    /*
    for (size_t i = 0; i < FFT_SIZE_POW2; ++i) {
      printf("yafft8 input_data[%lu]: (%+9f, %+9f)\n", i, 
	     input_data[i].real(), input_data[i].imag());
    } 
    */
    const long start = gettime();
    for (size_t i = 0; i < NLOOPS; ++i)
      fft_ptr->run_forward(input_data, output_data);
    const long elapsed = gettime() - start;
    printf("time taken: %ld usec\n", elapsed);
    /*
    for (size_t i = 0; i < FFT_SIZE_POW2; ++i) {
      printf("yafft8 output_data[%lu]: (%+9f, %+9f)\n", i, 
	     output_data[i].real(), output_data[i].imag());
    } 
    std::cout << std::endl;
    */
    return 0;
  }
}
