#include <emmintrin.h>
#include <stdint.h>

// The Windows x64 capture path owns a tightly packed 640x640 RGBX image.
// SSE2 is available on Windows x64 without an AVX requirement.
__declspec(dllexport) void vision_rgbx640_to_chw(const uint8_t *pixels, float *tensor) {
    const int plane = 640 * 640;
    const __m128i mask = _mm_set1_epi32(255);
    const __m128 scale = _mm_set1_ps(255.0f);
    for (int i = 0; i < plane; i += 4) {
        const __m128i rgbx = _mm_loadu_si128((const __m128i *)(pixels + 4 * i));
        const __m128i red = _mm_and_si128(rgbx, mask);
        const __m128i green = _mm_and_si128(_mm_srli_epi32(rgbx, 8), mask);
        const __m128i blue = _mm_and_si128(_mm_srli_epi32(rgbx, 16), mask);
        _mm_storeu_ps(tensor + i, _mm_div_ps(_mm_cvtepi32_ps(red), scale));
        _mm_storeu_ps(tensor + plane + i, _mm_div_ps(_mm_cvtepi32_ps(green), scale));
        _mm_storeu_ps(tensor + 2 * plane + i, _mm_div_ps(_mm_cvtepi32_ps(blue), scale));
    }
}
