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

#ifndef _YAFFT_WINDOWFN_IMPL_HPP_
#define _YAFFT_WINDOWFN_IMPL_HPP_

namespace yafft {

  template <typename FP_T>
  class hamming_windowfn : public iyafft_windowfn<FP_T> {
  public:

    hamming_windowfn() : window_p_(NULL), windowsz_(0) {}
    virtual ~hamming_windowfn() {}

    virtual FP_T * const get_window(const size_t windowsz) {

    if (windowsz_ == windowsz)
      return window_p_;
    else if (windowsz != 0)
      free(window_p_);

    void * v_p = NULL;
    yafft::utils::alloc_aligned(&v_p, 16, windowsz * sizeof(FP_T));
    window_p_ = (float*) v_p;
    windowsz_ = windowsz;

    const FP_T base = (2.0 * M_PI) / (windowsz - 1);

    for (size_t i = 0; i < windowsz; ++i) {
      window_p_[i] = 0.54 - (0.46 * cos(base * i));
    }

    return window_p_;	 
  }

  private:
    FP_T * window_p_;
    size_t windowsz_;
  };
  
  template <typename FP_T>
  class hann_windowfn : public iyafft_windowfn<FP_T> {
  public:

    hann_windowfn() : window_p_(NULL), windowsz_(0) {}
    virtual ~hann_windowfn() {}

    virtual FP_T * const get_window(const size_t windowsz) {

    if (windowsz_ == windowsz)
      return window_p_;
    else if (windowsz != 0)
      free(window_p_);

    void * v_p = NULL;
    yafft::utils::alloc_aligned((void**) &v_p, 16, windowsz * sizeof(FP_T));
    window_p_ = (float*) v_p;
    windowsz_ = windowsz;

    const FP_T base = (2.0 * M_PI) / (windowsz - 1);

    for (size_t i = 0; i < windowsz; ++i) {
      window_p_[i] = 0.5 * (1.0 - cos(base * i));
    }

    return window_p_;	 
  }

  private:
    FP_T * window_p_;
    size_t windowsz_;
  };
  
  template <typename FP_T>
  class gauss_windowfn : public iyafft_windowfn<FP_T> {
  public:

    gauss_windowfn(FP_T alpha = 0.4) : 
      window_p_(NULL), windowsz_(0),
      alpha_(alpha) {}

    virtual ~gauss_windowfn() {}

    virtual FP_T * const get_window(const size_t windowsz) {

    if (windowsz_ == windowsz)
      return window_p_;
    else if (windowsz != 0)
      free(window_p_);

    void * v_p = NULL;
    yafft::utils::alloc_aligned(&v_p, 16, windowsz * sizeof(FP_T));
    window_p_ = (float*) v_p;
    windowsz_ = windowsz;

    const FP_T N_m_1_o_2 = (FP_T) (windowsz_ - 1) / 2.0; 
    const FP_T base = alpha_ * N_m_1_o_2;

    for (size_t i = 0; i < windowsz; ++i) {
      const FP_T tmp = (((FP_T) i) - N_m_1_o_2) / base;
      window_p_[i] = pow(M_E, -0.5 * tmp * tmp);
    }

    return window_p_;	 
  }
  private:
    FP_T * window_p_;
    size_t windowsz_;
    FP_T alpha_;
  };
  
  template <typename FP_T>
  class blackman_windowfn : public iyafft_windowfn<FP_T> {
  public:

    blackman_windowfn(FP_T alpha = 0.16) : 
      window_p_(NULL), windowsz_(0),
      alpha_(alpha) {}

    virtual ~blackman_windowfn() {}

    virtual FP_T * const get_window(const size_t windowsz) {

    if (windowsz_ == windowsz)
      return window_p_;
    else if (windowsz != 0)
      free(window_p_);

    void * v_p = NULL;
    yafft::utils::alloc_aligned(&v_p, 16, windowsz * sizeof(FP_T));
    window_p_ = (float*) v_p;
    windowsz_ = windowsz;

    const FP_T a0 = (1.0 - alpha_) / 2.0;
    const FP_T a1 = 0.5;
    const FP_T a2 = alpha_ / 2.0;

    const FP_T N_fp = FP_T(windowsz_);

    const FP_T a1_term = (2.0 * M_PI) / (N_fp - 1.0);
    const FP_T a2_term = (4.0 * M_PI) / (N_fp - 1.0);

    for (size_t i = 0; i < windowsz; ++i) {
      const FP_T i_fp = FP_T(i);
      const FP_T term1 = a1 * cos(i_fp * a1_term);
      const FP_T term2 = a2 * cos(i_fp * a2_term);
      window_p_[i] = a0 - term1 + term2;
    }

    return window_p_;	 
  }

  private:
    FP_T * window_p_;
    size_t windowsz_;
    FP_T alpha_;
  };  
}

#endif // _YAFFT_WINDOWFN_IMPL_HPP_
