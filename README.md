A C++ Fast Fourier Transform library for single dimensions, targeted at audio DSP. It currently only supports 32-bit floats, but does handle real-valued and prime size input arrays. Prime size handling is done via the Bluestein algorithm.

The hardware requirements are an SSE-supporting Intel host. It compiles on Linux and Mac OS X using the Automake build system; use GCC 4.2 or higher for best results.

In performance terms it's worse than FFTW (see below for version) Estimate up until ~256 bins, then performs between Estimate and Measure until sizes get larger (262144 or so) at which point it is faster. Intel's FFT library is smoking fast.

The library size is very large, due to the way the code is generated.

-- Notes on FFTW: benched against FFTW 3.1 with SSE enabled
