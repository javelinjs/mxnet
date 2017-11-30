/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

/*!
* \file resize_bicubic-inl.h
* \brief
* \author
*/
#ifndef MXNET_OPERATOR_IMAGE_RESIZE_BICUBIC_INL_H
#define MXNET_OPERATOR_IMAGE_RESIZE_BICUBIC_INL_H

#include <vector>

namespace mxnet {
namespace op {

/* 8 bits for result. Filter can have negative areas.
   In one cases the sum of the coefficients will be negative,
   in the other it will be more than 1.0. That is why we need
   two extra bits for overflow and int type. */
#define PRECISION_BITS (32 - 8 - 2)

typedef void* ImagingSectionCookie;

struct filter {
  double (*filter)(double x);
  double support;
};

void normalize_coeffs_8bpc(int outSize, int ksize, double *prekk) {
  int x;
  int *kk;

  // use the same buffer for normalized coefficients
  kk = (int *) prekk;

  for (x = 0; x < outSize * ksize; x++) {
    if (prekk[x] < 0) {
      kk[x] = (int) (-0.5 + prekk[x] * (1 << PRECISION_BITS));
    } else {
      kk[x] = (int) (0.5 + prekk[x] * (1 << PRECISION_BITS));
    }
  }
}

uint8_t _lookups[512] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
  64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
  80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
  96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
  112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
  128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
  144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
  160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
  176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
  192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
  208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
  224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
  240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
};

uint8_t *lookups = &_lookups[128];

static inline uint8_t clip8(int in) {
  return lookups[in >> PRECISION_BITS];
}

int precompute_coeffs(int inSize, float in0, float in1, int outSize,
                      struct filter *filterp, int **boundsp, double **kkp) {
  double support, scale, filterscale;
  double center, ww, ss;
  int xx, x, ksize, xmin, xmax;
  int *bounds;
  double *kk, *k;

  /* prepare for horizontal stretch */
  filterscale = scale = (double) (in1 - in0) / outSize;
  if (filterscale < 1.0) {
    filterscale = 1.0;
  }

  /* determine support size (length of resampling filter) */
  support = filterp->support * filterscale;

  /* maximum number of coeffs */
  ksize = (int) ceil(support) * 2 + 1;

  // check for overflow
  CHECK(outSize <= INT_MAX / (ksize * sizeof(double)));

  /* coefficient buffer */
  /* malloc check ok, overflow checked above */
  kk = static_cast<double *>(malloc(outSize * ksize * sizeof(double)));
  CHECK(kk);

  /* malloc check ok, ksize*sizeof(double) > 2*sizeof(int) */
  bounds = static_cast<int *>(malloc(outSize * 2 * sizeof(int)));
  if (!bounds) {
    free(kk);
    CHECK(bounds);
  }

  for (xx = 0; xx < outSize; xx++) {
    center = in0 + (xx + 0.5) * scale;
    ww = 0.0;
    ss = 1.0 / filterscale;
    // Round the value
    xmin = (int) (center - support + 0.5);
    if (xmin < 0)
      xmin = 0;
    // Round the value
    xmax = (int) (center + support + 0.5);
    if (xmax > inSize)
      xmax = inSize;
    xmax -= xmin;
    k = &kk[xx * ksize];
    for (x = 0; x < xmax; x++) {
      double w = filterp->filter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (x = 0; x < xmax; x++) {
      if (ww != 0.0)
        k[x] /= ww;
    }
    // Remaining values should stay empty if they are used despite of xmax.
    for (; x < ksize; x++) {
      k[x] = 0;
    }
    bounds[xx * 2 + 0] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  *boundsp = bounds;
  *kkp = kk;
  return ksize;
}

template<typename DType>
static void ImagingResampleHorizontal_8bpc(DType *imOut, const DType *imIn,
                                           int oheight, int owidth,
                                           int iheight, int iwidth, int nchannel,
                                           int offset, int ksize,
                                           int *bounds, double *prekk) {
  int ss0, ss1, ss2;
  int xx, yy, x, xmin, xmax;
  int *k, *kk;

  // use the same buffer for normalized coefficients
  kk = (int *) prekk;
  normalize_coeffs_8bpc(owidth, ksize, prekk);

  // TODO: nchannel = 3
  const int ilength = iwidth * nchannel;
  const int olength = owidth * nchannel;

  for (yy = 0; yy < oheight; yy++) {
    for (xx = 0; xx < owidth; xx++) {
      xmin = bounds[xx * 2 + 0];
      xmax = bounds[xx * 2 + 1];
      k = &kk[xx * ksize];
      ss0 = ss1 = ss2 = 1 << (PRECISION_BITS -1);
      for (x = 0; x < xmax; x++) {
        ss0 += ((uint8_t) imIn[(yy + offset)*ilength + (x + xmin)*nchannel + 0]) * k[x];
        ss1 += ((uint8_t) imIn[(yy + offset)*ilength + (x + xmin)*nchannel + 1]) * k[x];
        ss2 += ((uint8_t) imIn[(yy + offset)*ilength + (x + xmin)*nchannel + 2]) * k[x];
      }
      imOut[yy*olength + xx*nchannel + 0] = clip8(ss0);
      imOut[yy*olength + xx*nchannel + 1] = clip8(ss1);
      imOut[yy*olength + xx*nchannel + 2] = clip8(ss2);
    }
  }
}

template<typename DType>
static void ImagingResampleVertical_8bpc(DType *imOut, const DType *imIn,
                                         int oheight, int owidth,
                                         int iheight, int iwidth, int nchannel,
                                         int offset, int ksize,
                                         int *bounds, double *prekk) {
  int ss0, ss1, ss2, ss3;
  int xx, yy, y, ymin, ymax;
  int *k, *kk;

  // use the same buffer for normalized coefficients
  kk = (int *) prekk;
  normalize_coeffs_8bpc(oheight, ksize, prekk);

  const int ilength = iwidth * nchannel;
  const int olength = owidth * nchannel;

  for (yy = 0; yy < oheight; yy++) {
    k = &kk[yy * ksize];
    ymin = bounds[yy * 2 + 0];
    ymax = bounds[yy * 2 + 1];
    for (xx = 0; xx < owidth; xx++) {
      ss0 = ss1 = ss2 = 1 << (PRECISION_BITS -1);
      for (y = 0; y < ymax; y++) {
        ss0 += ((uint8_t) imIn[(y + ymin)*ilength + xx*nchannel + 0]) * k[y];
        ss1 += ((uint8_t) imIn[(y + ymin)*ilength + xx*nchannel + 1]) * k[y];
        ss2 += ((uint8_t) imIn[(y + ymin)*ilength + xx*nchannel + 2]) * k[y];
      }
      imOut[yy*olength + xx*nchannel + 0] = clip8(ss0);
      imOut[yy*olength + xx*nchannel + 1] = clip8(ss1);
      imOut[yy*olength + xx*nchannel + 2] = clip8(ss2);
    }
  }
}

static inline double bicubic_filter(double x) {
  // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
  const double a = -0.5;
  if (x < 0.0)
    x = -x;
  if (x < 1.0)
    return ((a + 2.0) * x - (a + 3.0)) * x*x + 1;
  if (x < 2.0)
    return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}

static struct filter BICUBIC = { bicubic_filter, 2.0 };

template<typename DType>
static void resize_bicubic(DType *output, DType *input,
                           const int oheight, const int owidth,
                           const int iheight, const int iwidth, const int nchannel) {
  struct filter *filterp = &BICUBIC;

  // do resample
  int i, need_horizontal, need_vertical;
  int ybox_first, ybox_last;
  int ksize_horiz, ksize_vert;
  int *bounds_horiz, *bounds_vert;
  double *kk_horiz, *kk_vert;

  need_horizontal = owidth != iwidth;
  need_vertical = oheight != iheight;

  ksize_horiz = precompute_coeffs(iwidth, 0, iwidth, owidth,
                                  filterp, &bounds_horiz, &kk_horiz);
  CHECK(ksize_horiz);

  ksize_vert = precompute_coeffs(iheight, 0, iheight, oheight,
                                 filterp, &bounds_vert, &kk_vert);
  if (!ksize_vert) {
    free(bounds_horiz);
    free(kk_horiz);
    CHECK(ksize_vert);
  }

  // First used row in the source image
  ybox_first = bounds_vert[0];
  // Last used row in the source image
  ybox_last = bounds_vert[oheight*2 - 2] + bounds_vert[oheight*2 - 1];


  DType *imTemp = NULL;
  /* two-pass resize, horizontal pass */
  if (need_horizontal) {
    // Shift bounds for vertical pass
    for (i = 0; i < oheight; i++) {
      bounds_vert[i * 2] -= ybox_first;
    }

    // imTemp = ImagingNewDirty(imIn->mode, xsize, ybox_last - ybox_first);
    imTemp = new DType[(ybox_last - ybox_first) * owidth * nchannel];
    if (imTemp) {
      ImagingResampleHorizontal_8bpc<DType>(imTemp, input,
                                            ybox_last - ybox_first, owidth,
                                            iheight, iwidth, nchannel,
                                            ybox_first, ksize_horiz, bounds_horiz, kk_horiz);
    }
    free(bounds_horiz);
    free(kk_horiz);
    if (!imTemp) {
      free(bounds_vert);
      free(kk_vert);
      CHECK(imTemp);
    }
    input = imTemp;
    std::memcpy(output, imTemp, (ybox_last - ybox_first) * owidth * nchannel);
  } else {
    // Free in any case
    free(bounds_horiz);
    free(kk_horiz);
  }

  /* vertical pass */
  if (need_vertical) {
    /* imIn can be the original image or horizontally resampled one */
    ImagingResampleVertical_8bpc<DType>(output, input, oheight, owidth,
                                        ybox_last - ybox_first, owidth,
                                        nchannel,
                                        0, ksize_vert, bounds_vert, kk_vert);
    /* it's safe to call ImagingDelete with empty value
       if previous step was not performed. */
    delete[] imTemp;
    free(bounds_vert);
    free(kk_vert);
  } else {
    // Free in any case
    free(bounds_vert);
    free(kk_vert);
  }
}

}  // namespace op
}  // namespace mxnet

#endif //  MXNET_OPERATOR_IMAGE_RESIZE_BICUBIC_INL_H
