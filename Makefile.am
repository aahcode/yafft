# hacky but it should work
AM_CFLAGS=
AM_CXXFLAGS=
AM_LDFLAGS = -L. 

combined_cxxflags = -g -Wall -pedantic -msse -funroll-loops -ffast-math

lib_source = yafft.cpp

lib_LTLIBRARIES = libyafft.la libyafft_debug.la

libyafft_la_SOURCES = ${lib_source}
libyafft_la_CFLAGS =
libyafft_la_CPPFLAGS = -O3 ${combined_cxxflags}

libyafft_debug_la_SOURCES = ${lib_source}
libyafft_debug_la_CFLAGS =
libyafft_debug_la_CPPFLAGS = -O0 -D_DEBUG ${combined_cxxflags}

bin_PROGRAMS = fft_test.bin fft_test.bin.debug
bin_source = fft_test.cpp

fft_test_bin_SOURCES = ${bin_source}
fft_test_bin_CFLAGS = 
fft_test_bin_LDADD = -lfftw3f -lm -lyafft
fft_test_bin_CPPFLAGS = -O3 ${combined_cxxflags}

fft_test_bin_debug_SOURCES = ${bin_source}
fft_test_bin_debug_CFLAGS = 
fft_test_bin_debug_CPPFLAGS = -O0 -D_DEBUG ${combined_cxxflags}
fft_test_bin_debug_LDADD = -lfftw3f -lm -lyafft_debug
