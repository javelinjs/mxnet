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
* \file image_random-inl.h
* \brief
* \author
*/
#ifndef MXNET_OPERATOR_IMAGE_IMAGE_RANDOM_INL_H_
#define MXNET_OPERATOR_IMAGE_IMAGE_RANDOM_INL_H_

#include <mxnet/base.h>
#include <vector>
#include <algorithm>
#include <utility>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "resize_bicubic-inl.h"

namespace mxnet {
namespace op {

inline bool CheckIsImage(const TBlob &image) {
  CHECK_EQ(image.type_flag_, mshadow::kUint8) << "input type is not an image.";
  CHECK_EQ(image.ndim(), 3) << "input dimension is not 3.";
  CHECK(image.shape_[2] == 1 || image.shape_[2] == 3) << "image channel should be 1 or 3.";
}

static void RandomFlip(const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
}

inline bool ToTensorType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_EQ((*in_attrs)[0], mshadow::kUint8)
    << "`to_tensor` only supports uint8 input";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

inline bool ToTensorShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TShape &shp = (*in_attrs)[0];
  CHECK_EQ(shp.ndim(), 3U) << "`to_tensor` only supports 3 dimensions";
  TShape ret(3);
  ret[0] = shp[2];
  ret[1] = shp[0];
  ret[2] = shp[1];
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return true;
}

static void ToTensor(const nnvm::NodeAttrs &attrs,
                     const OpContext &ctx,
                     const std::vector<TBlob> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &outputs) {
  CHECK_EQ(req[0], kWriteTo)
    << "`to_tensor` does not support inplace";
  CheckIsImage(inputs[0]);

  int length = inputs[0].shape_[0] * inputs[0].shape_[1];
  int channel = inputs[0].shape_[2];

  float* output = outputs[0].dptr<float>();
  uint8_t* input = inputs[0].dptr<uint8_t>();

  for (int l = 0; l < length; ++l) {
    for (int c = 0; c < channel; ++c) {
      output[c*length + l] = static_cast<float>(input[l*channel + c]) / 255.0f;
    }
  }
}

struct NormalizeParam : public dmlc::Parameter<NormalizeParam> {
  nnvm::Tuple<float> mean;
  nnvm::Tuple<float> std;
  DMLC_DECLARE_PARAMETER(NormalizeParam) {
    DMLC_DECLARE_FIELD(mean)
    .describe("Sequence of mean for each channel.");
    DMLC_DECLARE_FIELD(std)
    .describe("Sequence of standard deviations for each channel.");
  }
};


inline bool NormalizeShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  const NormalizeParam &param = nnvm::get<NormalizeParam>(attrs.parsed);
  const auto& dshape = (*in_attrs)[0];
  if (!dshape.ndim()) return false;
  CHECK_EQ(dshape.ndim(), 3)
      << "Input must have 3 dimensions";

  auto nchannels = dshape[0];
  CHECK(param.mean.ndim() == 1 || param.mean.ndim() == nchannels)
      << "mean must have either 1 or " << nchannels << " elements";
  CHECK(param.std.ndim() == 1 || param.std.ndim() == nchannels)
      << "std must have either 1 or " << nchannels << " elements";

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape);
}

static void Normalize(const nnvm::NodeAttrs &attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
  const NormalizeParam &param = nnvm::get<NormalizeParam>(attrs.parsed);

  int nchannels = inputs[0].shape_[0];
  int length = inputs[0].shape_[1] * inputs[0].shape_[2];

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    DType* input = inputs[0].dptr<DType>();
    DType* output = outputs[0].dptr<DType>();

    for (int i = 0; i < nchannels; ++i) {
      DType mean = param.mean[param.mean.ndim() > 1 ? i : 0];
      DType std = param.std[param.std.ndim() > 1 ? i : 0];
      for (int j = 0; j < length; ++j) {
        output[i*length + j] = (input[i*length + j] - mean) / std;
      }
    }
  });
}

inline static int FlipIndex(int idx, const int stride, const int trailing) {
  const int low = idx % trailing;
  int high = idx / trailing;
  const int x = high % stride;
  high /= stride;

  return (high * stride + stride - 1 - x) * trailing + low;
}

template<typename DType>
static void FlipImpl(const int size, DType *src, DType *dst,
                     const int stride, const int trailing) {
  for (int idx = 0; idx < size; ++idx) {
    int new_idx = FlipIndex(idx, stride, trailing);
    if (src == dst) {
      // inplace operation
      if (idx < new_idx) {
        std::swap(dst[new_idx], src[idx]);
      }
    } else {
      dst[new_idx] = src[idx];
    }
  }
}

template<int axis>
static void Flip(const nnvm::NodeAttrs &attrs,
                  const OpContext &ctx,
                  const std::vector<TBlob> &inputs,
                  const std::vector<OpReqType> &req,
                  const std::vector<TBlob> &outputs) {
  CheckIsImage(inputs[0]);
  const TShape& ishape = inputs[0].shape_;
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    FlipImpl(inputs[0].Size(), inputs[0].dptr<DType>(), outputs[0].dptr<DType>(),
             ishape[1-axis], ishape[2] * (axis == 0 ? 1 : ishape[1]));
  });
}

struct ImageCropParam : public dmlc::Parameter<ImageCropParam> {
  int y, x, h, w;
  DMLC_DECLARE_PARAMETER(ImageCropParam) {
    DMLC_DECLARE_FIELD(y)
    .describe("Upper pixel coordinate.");
    DMLC_DECLARE_FIELD(x)
    .describe("Left pixel coordinate.");
    DMLC_DECLARE_FIELD(h)
    .describe("Height of the cropped image.");
    DMLC_DECLARE_FIELD(w)
    .describe("Width of the cropped image.");
  }
};

inline bool ImageCropShape(const nnvm::NodeAttrs& attrs,
                            std::vector<TShape> *in_attrs,
                            std::vector<TShape> *out_attrs) {
  const ImageCropParam &param = nnvm::get<ImageCropParam>(attrs.parsed);
  const auto &ishape = (*in_attrs)[0];
  if (!ishape.ndim()) return false;

  CHECK_EQ(ishape.ndim(), 3) << "Input must have 3 dimensions.";
  CHECK(param.y >= 0 && param.y < ishape[0])
    << "Invalid upper pixel coordinate " << param.y;
  CHECK(param.x >= 0 && param.x < ishape[1])
    << "Invalid left pixel coordinate " << param.x;
  CHECK(param.h > 0 && param.h + param.y <= ishape[0])
    << "Invalid cropped height " << param.h;
  CHECK(param.w > 0 && param.w + param.x <= ishape[1])
    << "Invalid cropped width " << param.w;

  TShape oshape({param.h, param.w, ishape[2]});
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return oshape.Size() != 0;
}

static void ImageCrop(const nnvm::NodeAttrs &attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
  CheckIsImage(inputs[0]);
  const ImageCropParam &param = nnvm::get<ImageCropParam>(attrs.parsed);

  const int nchannels = inputs[0].shape_[2];
  const int length_src = inputs[0].shape_[1] * inputs[0].shape_[2];
  const int length_dst = param.w * nchannels;

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    DType *input = inputs[0].dptr<DType>();
    DType *output = outputs[0].dptr<DType>();

    for (int y = param.y; y < param.y + param.h; ++y) {
      const int j = y - param.y;
      for (int x = param.x; x < param.x + param.w; ++x) {
        const int i = x - param.x;
        for (int c = 0; c < nchannels; ++c) {
          output[j * length_dst + i * nchannels + c]
            = input[y * length_src + x * nchannels + c];
        }
      }
    }
  });
}

struct ImageResizeParam : public dmlc::Parameter<ImageResizeParam> {
  int height, width, interpolation;
  DMLC_DECLARE_PARAMETER(ImageResizeParam) {
    DMLC_DECLARE_FIELD(height)
    .describe("Height of the resized image.");
    DMLC_DECLARE_FIELD(width)
    .describe("Width of the resized image.");
    // TODO: describe
    DMLC_DECLARE_FIELD(interpolation)
    .describe("An optional resampling filter."
              "This can be one of PIL.Image.NEAREST (use nearest neighbour),"
              "PIL.Image.BILINEAR (linear interpolation),"
              "PIL.Image.BICUBIC (cubic spline interpolation),"
              "or PIL.Image.LANCZOS (a high-quality downsampling filter)."
              "If omitted, it is set PIL.Image.BILINEAR.")
    .set_default(2);
  }
};

inline bool ImageResizeShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  const ImageResizeParam &param = nnvm::get<ImageResizeParam>(attrs.parsed);
  const auto &ishape = (*in_attrs)[0];
  if (!ishape.ndim()) return false;

  CHECK_EQ(ishape.ndim(), 3) << "Input must have 3 dimensions.";
  CHECK_GT(param.height, 0) << "Invalid resize height " << param.height;
  CHECK_GT(param.width, 0) << "Invalid resize width " << param.width;

  TShape oshape({param.height, param.width, ishape[2]});
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return oshape.Size() != 0;
}

struct CachedInterpolation {
  int lower;  // Lower source index used in the interpolation
  int upper;  // Upper source index used in the interpolation
  // 1-D linear iterpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  float lerp;
};

inline void compute_interpolation_weights(const int out_size,
                                          const int in_size,
                                          const float scale,
                                          CachedInterpolation* interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (int i = out_size - 1; i >= 0; --i) {
    const float in = i * scale;
    interpolation[i].lower = static_cast<int>(in);
    interpolation[i].upper = std::min(interpolation[i].lower + 1, in_size - 1);
    interpolation[i].lerp = in - interpolation[i].lower;
  }
}

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 */
inline float compute_lerp(const float top_left, const float top_right,
                          const float bottom_left, const float bottom_right,
                          const float x_lerp, const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

template<typename DType>
static void resize_bilinear(DType *output, const DType *input,
                            const int oheight, const int owidth,
                            const int iheight, const int iwidth, const int nchannel) {
  const float height_scale = iheight / static_cast<float>(oheight);
  const float width_scale = iwidth / static_cast<float>(owidth);

  std::vector<CachedInterpolation> ys(oheight + 1);
  std::vector<CachedInterpolation> xs(owidth + 1);

  // Compute the cached interpolation weights on the x and y dimensions.
  compute_interpolation_weights(oheight, iheight, height_scale, ys.data());
  compute_interpolation_weights(owidth, iwidth, width_scale, xs.data());

  // Scale x interpolation weights to avoid a multiplication during iteration.
  for (int i = 0; i < xs.size(); ++i) {
    xs[i].lower *= nchannel;
    xs[i].upper *= nchannel;
  }

  const int in_row_size = iwidth * nchannel;
  const int in_batch_num_values = iheight * in_row_size;
  const int out_row_size = owidth * nchannel;

  for (int y = 0; y < oheight; ++y) {
    const DType *ys_input_lower_ptr = input + ys[y].lower * in_row_size;
    const DType *ys_input_upper_ptr = input + ys[y].upper * in_row_size;
    const float ys_lerp = ys[y].lerp;

    for (int x = 0; x < owidth; ++x) {
      const int xs_lower = xs[x].lower;
      const int xs_upper = xs[x].upper;
      const float xs_lerp = xs[x].lerp;

      // Read channel 0.
      const float top_left0(ys_input_lower_ptr[xs_lower + 0]);
      const float top_right0(ys_input_lower_ptr[xs_upper + 0]);
      const float bottom_left0(ys_input_upper_ptr[xs_lower + 0]);
      const float bottom_right0(ys_input_upper_ptr[xs_upper + 0]);

      // Read channel 1.
      const float top_left1(ys_input_lower_ptr[xs_lower + 1]);
      const float top_right1(ys_input_lower_ptr[xs_upper + 1]);
      const float bottom_left1(ys_input_upper_ptr[xs_lower + 1]);
      const float bottom_right1(ys_input_upper_ptr[xs_upper + 1]);

      // Read channel 2.
      const float top_left2(ys_input_lower_ptr[xs_lower + 2]);
      const float top_right2(ys_input_lower_ptr[xs_upper + 2]);
      const float bottom_left2(ys_input_upper_ptr[xs_lower + 2]);
      const float bottom_right2(ys_input_upper_ptr[xs_upper + 2]);

      // Compute output.
      output[x * nchannel + 0] =
      compute_lerp(top_left0, top_right0, bottom_left0, bottom_right0, xs_lerp, ys_lerp);

      output[x * nchannel + 1] =
      compute_lerp(top_left1, top_right1, bottom_left1, bottom_right1, xs_lerp, ys_lerp);

      output[x * nchannel + 2] =
      compute_lerp(top_left2, top_right2, bottom_left2, bottom_right2, xs_lerp, ys_lerp);
    }

    output += out_row_size;
  }
}

static void ImageResize(const nnvm::NodeAttrs &attrs,
                        const OpContext &ctx,
                        const std::vector<TBlob> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &outputs) {
  CheckIsImage(inputs[0]);
  const ImageResizeParam &param = nnvm::get<ImageResizeParam>(attrs.parsed);

  const int iheight = inputs[0].shape_[0];
  const int iwidth = inputs[0].shape_[1];
  const int nchannel = inputs[0].shape_[2];

  const int oheight = outputs[0].shape_[0];
  const int owidth = outputs[0].shape_[1];

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    if (oheight == iheight && owidth == iwidth) {
      std::memcpy(outputs[0].dptr_, inputs[0].dptr_, iheight * iwidth * nchannel * sizeof(DType));
      return;
    }

    DType *output = outputs[0].dptr<DType>();
    DType *input = inputs[0].dptr<DType>();

    if (param.interpolation == 2) {
      resize_bilinear<DType>(output, input, oheight, owidth, iheight, iwidth, nchannel);
    } else if (param.interpolation == 3) {
      resize_bicubic<DType>(output, input, oheight, owidth, iheight, iwidth, nchannel);
    }
  });
}

struct RandomBrightnessParam : public dmlc::Parameter<RandomBrightnessParam> {
  float max_brightness;
  DMLC_DECLARE_PARAMETER(RandomBrightnessParam) {
    DMLC_DECLARE_FIELD(max_brightness)
    .set_lower_bound(0.0)
    .describe("Max Brightness.");
  }
};

static void RandomBrightness(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  const RandomBrightnessParam &param = nnvm::get<RandomBrightnessParam>(attrs.parsed);

  int length = inputs[0].Size();

  uint8_t* output = outputs[0].dptr<uint8_t>();
  uint8_t* input = inputs[0].dptr<uint8_t>();

  Stream<cpu> *s = ctx.get_stream<cpu>();
  Random<cpu> *prnd = ctx.requested[0].get_random<cpu, float>(s);
  float alpha_b = 1.0 + std::uniform_real_distribution<float>(
      -param.max_brightness, param.max_brightness)(prnd->GetRndEngine());

  for (int l = 0; l < length; ++l) {
    float val = static_cast<float>(input[l]) * alpha_b;
    val = std::min(std::max(val, 0.f), 255.f);
    output[l] = static_cast<uint8_t>(val);
  }
}


struct RandomContrastParam : public dmlc::Parameter<RandomContrastParam> {
  float max_contrast;
  DMLC_DECLARE_PARAMETER(RandomContrastParam) {
    DMLC_DECLARE_FIELD(max_contrast)
    .set_lower_bound(0.0)
    .describe("Max Contrast.");
  }
};


static void RandomContrast(const nnvm::NodeAttrs &attrs,
                           const OpContext &ctx,
                           const std::vector<TBlob> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  static const float coef[] = { 0.299f, 0.587f, 0.114f };
  const RandomContrastParam &param = nnvm::get<RandomContrastParam>(attrs.parsed);

  int length = inputs[0].shape_[0] * inputs[0].shape_[1];
  int nchannels = inputs[0].shape_[2];

  uint8_t* output = outputs[0].dptr<uint8_t>();
  uint8_t* input = inputs[0].dptr<uint8_t>();

  Stream<cpu> *s = ctx.get_stream<cpu>();
  Random<cpu> *prnd = ctx.requested[0].get_random<cpu, real_t>(s);
  float alpha_c = 1.0 + std::uniform_real_distribution<float>(
    -param.max_contrast, param.max_contrast)(prnd->GetRndEngine());

  float sum = 0.f;
  if (nchannels > 1) {
    for (int l = 0; l < length; ++l) {
      for (int c = 0; c < nchannels; ++c) sum += input[l*nchannels + c] * coef[c];
    }
  } else {
    for (int l = 0; l < length; ++l) sum += input[l];
  }
  float gray_mean = sum / static_cast<float>(length);
  float beta = (1 - alpha_c) * gray_mean;

  for (int l = 0; l < length * nchannels; ++l) {
    float val = input[l] * alpha_c + beta;
    val = std::min(std::max(val, 0.f), 255.f);
    output[l] = static_cast<uint8_t>(val);
  }
}

struct RandomSaturationParam : public dmlc::Parameter<RandomSaturationParam> {
  float max_saturation;
  DMLC_DECLARE_PARAMETER(RandomSaturationParam) {
    DMLC_DECLARE_FIELD(max_saturation)
    .set_default(0.0)
    .describe("Max Saturation.");
  }
};

static void RandomSaturation(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  const RandomSaturationParam &param = nnvm::get<RandomSaturationParam>(attrs.parsed);
  static const float coef[] = { 0.299f, 0.587f, 0.114f };

  int length = inputs[0].shape_[0] * inputs[0].shape_[1];
  int nchannels = inputs[0].shape_[2];

  uint8_t* output = outputs[0].dptr<uint8_t>();
  uint8_t* input = inputs[0].dptr<uint8_t>();

  Stream<cpu> *s = ctx.get_stream<cpu>();
  Random<cpu> *prnd = ctx.requested[0].get_random<cpu, real_t>(s);
  float alpha_s = 1.f + std::uniform_real_distribution<float>(
    -param.max_saturation, param.max_saturation)(prnd->GetRndEngine());
  float alpha_o = 1.f - alpha_s;

  if (nchannels == 1) {
    for (int l = 0; l < length * nchannels; ++l) output[l] = input[l];
    return;
  }

  for (int l = 0; l < length; ++l) {
    float gray = 0.f;
    for (int c = 0; c < nchannels; ++c) {
      gray = input[l*nchannels + c] * coef[c];
    }
    gray *= alpha_o;
    for (int c = 0; c < nchannels; ++c) {
      float val = gray + input[l*nchannels + c] * alpha_s;
      val = std::min(std::max(val, 0.f), 255.f);
      output[l*nchannels + c] = static_cast<uint8_t>(val);
    }
  }
}

static void RandomHue(const nnvm::NodeAttrs &attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
}

static void RandomColorJitter(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
}

static void RandomLighting(const nnvm::NodeAttrs &attrs,
                           const OpContext &ctx,
                           const std::vector<TBlob> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &outputs) {
}




}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_IMAGE_RANDOM_INL_H_
