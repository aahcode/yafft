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

#include "yafft8.hpp"
#include "fft_test_defines.hpp"

#include <sys/time.h>

namespace yafft8_impl {
  
  boost::shared_ptr<yafft8::iyafft<float> > yafft8_impl;
  
  static long gettime() {
    timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec*1000000+tv.tv_usec);
  }
  
  void init() {
    yafft8_impl = yafft8::get_fft<float>(DATA_SZ);
    yafft8_impl->init();
  }

  long run_both(std::complex<float> * const in,
		std::complex<float> * const out,
		const bool print) {
    
    long start, end, yafft8_time = 0;

#ifdef _DEBUG
    printf("yafft8 start forward\n");
    ::memset(out, 0, (sizeof(std::complex<float>)*DATA_SZ));    
#endif
    
    start = gettime();
    //__asm__("yafft8fstart:");
    for (size_t i = 0; i < INNERLOOP; ++i)
      yafft8_impl->run_forward(in, out);
    //__asm__("yafft8fend:");
    end = gettime();

    yafft8_time += end - start;
    
    if (print) {
      printf("yafft8 forward out\n");
      for (size_t i = 0; i < DATA_SZ; ++i) 
	printf("(%+f, %+f)\n", out[i].real(), out[i].imag());
    }

    ::memcpy(in, out, sizeof(std::complex<float>)*DATA_SZ);

#ifdef _DEBUG
    printf("yafft8 start reverse\n");
    ::memset(out, 0, (sizeof(std::complex<float>)*DATA_SZ));
#endif
        
    start = gettime();
    //    __asm__("yafft8rstart:");
    for (size_t i = 0; i < INNERLOOP; ++i)
      yafft8_impl->run_reverse(in, out);
    //    __asm__("yafft8rend:");
    end = gettime();
    
    yafft8_time += end - start;

    if (print) {
      printf("yafft8 reverse out\n");
      for (size_t i = 0; i < DATA_SZ; ++i) 
	printf("(%+f, %+f)\n", out[i].real(), out[i].imag());
    }
    
    return yafft8_time;
  }
}
