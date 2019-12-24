/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "positionWisePlugin.h"
#include "cublas_v2.h"
#include <cstring>
#include <cudnn.h>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::PositionWise;
using nvinfer1::plugin::PositionWisePluginCreator;

namespace
{
  const char* POSITION_WISE_PLUGIN_VERSION{ "001" };
  const char* POSITION_WISE_PLUGIN_NAME{ "PositionWise_TRT" };
} // namespace

PluginFieldCollection PositionWisePluginCreator::mFC{};
std::vector<PluginField> PositionWisePluginCreator::mPluginAttributes;

PositionWise::PositionWise(const int seql,const int nfeat,DataType dtype,Weights pos) {
  Seql = seql;
  n_Feat = nfeat;
  ctype = dtype;

  CHECK(cudaMalloc(&position, type2size(ctype) * Seql * n_Feat));
  if (dtype == DataType::kHALF) {
    for (int i = 0; i < Seql*n_Feat; i++) {
      ((half*)pos.values)[i] = __float2half(((float*)pos.values)[i]);
    }
  }

  CHECK(cudaMemcpy(position, pos.values, Seql*n_Feat*type2size(ctype), cudaMemcpyHostToDevice));
  
  //cudnn compute
  CHECK(cudnnCreate(&handle));
  CHECK(cudnnCreateTensorDescriptor(&xdes));
  CHECK(cudnnCreateOpTensorDescriptor(&opdes));
  if (ctype == DataType::kFLOAT) {
    CHECK(cudnnSetOpTensorDescriptor(opdes, cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD,
      cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN));
  }
  else {
    CHECK(cudnnSetOpTensorDescriptor(opdes, cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD,
      cudnnDataType_t::CUDNN_DATA_HALF, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN));
  }
}

PositionWise::PositionWise(const PositionWise& other) {
  Seql = other.Seql;
  n_Feat = other.n_Feat;
  ctype = other.ctype;
  mNameSpace = other.mNameSpace;

  CHECK(cudaMalloc(&position, type2size(ctype)*Seql*n_Feat));
  CHECK(cudaMemcpy(position, other.position, Seql*n_Feat*type2size(ctype), cudaMemcpyDeviceToDevice));

  //cudnn compute
  CHECK(cudnnCreate(&handle));
  CHECK(cudnnCreateTensorDescriptor(&xdes));
  CHECK(cudnnCreateOpTensorDescriptor(&opdes));
  if (ctype == DataType::kFLOAT) {
    CHECK(cudnnSetOpTensorDescriptor(opdes, cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD,
      cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN));
  }
  else {
    CHECK(cudnnSetOpTensorDescriptor(opdes, cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD,
      cudnnDataType_t::CUDNN_DATA_HALF, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN));
  }
}

PositionWise::PositionWise(const void* buffer, size_t length) {
  const char* ptr = (const char*)buffer;
  Seql = read<int>(ptr);
  n_Feat = read<int>(ptr);
  ctype = (DataType)read<int>(ptr);

  CHECK(cudaMalloc(&position, type2size(ctype)*Seql*n_Feat));
  CHECK(cudaMemcpy(position, ptr, Seql*n_Feat*type2size(ctype), cudaMemcpyDeviceToDevice));

  //cudnn compute
  CHECK(cudnnCreate(&handle));
  CHECK(cudnnCreateTensorDescriptor(&xdes));
  CHECK(cudnnCreateOpTensorDescriptor(&opdes));
  if (ctype == DataType::kFLOAT) {
    CHECK(cudnnSetOpTensorDescriptor(opdes, cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD,
      cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN));
  }
  else {
    CHECK(cudnnSetOpTensorDescriptor(opdes, cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD,
      cudnnDataType_t::CUDNN_DATA_HALF, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN));
  }
}

IPluginV2DynamicExt* PositionWise::clone()const {
  return (IPluginV2DynamicExt*)new PositionWise(*this);
}

DimsExprs PositionWise::getOutputDimensions(
  int outputIndex, const nvinfer1::DimsExprs* inputs,
  int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
  ASSERT(nbInputs == 1);
  return inputs[0];
}

bool PositionWise::supportsFormatCombination(
  int pos, const nvinfer1::PluginTensorDesc* inOut,
  int nbInputs, int nbOutputs) {
  ASSERT(nbInputs == 1);
  if (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF)
    return true;
  else return false;
}

void PositionWise::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
  const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
  ASSERT(nbInputs == 1);
  //ctype = in[0].desc.type;
}

size_t PositionWise::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
  const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
  return 0;
}

int PositionWise::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
  const nvinfer1::PluginTensorDesc* outputDesc,
  const void* const* inputs, void* const* outputs,
  void* workspace, cudaStream_t stream) {
  int n = inputDesc->dims.d[0];
  int row = inputDesc->dims.d[1];
  ASSERT(row <= Seql);
  if (ctype == DataType::kFLOAT) {
    CHECK(cudnnSetTensor4dDescriptor(xdes, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, n, 1, row, n_Feat));
  }
  else {
    CHECK(cudnnSetTensor4dDescriptor(xdes, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_HALF, n, 1, row, n_Feat));
  }

  CHECK(cudnnSetStream(handle,stream));
  float alpha1 = 1.0, alpha2 = sqrtf(n_Feat), beta = 0.0;
  //x*sqrtf(odim) + postion
  CHECK(cudnnOpTensor(handle, opdes,
    &alpha1, xdes, position,
    &alpha2, xdes, inputs[0],
    &beta, xdes, outputs[0]));
  return 0;
}

DataType PositionWise::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  return ctype;
}

const char* PositionWise::getPluginVersion() const {
  return POSITION_WISE_PLUGIN_VERSION;
}
const char* PositionWise::getPluginType() const {
  return POSITION_WISE_PLUGIN_NAME;
}

int PositionWise::getNbOutputs() const {
  return 1;
}

int PositionWise::initialize() {
  return 0;
}

void PositionWise::terminate() {}

void PositionWise::destroy() {}

void PositionWise::serialize(void* buffer) const {
  char* ptr = (char*)buffer;
  write(ptr, Seql);
  write(ptr, n_Feat);
  write(ptr, (int)ctype);

  CHECK(cudaMemcpy(ptr, position, type2size(ctype)*Seql*n_Feat,cudaMemcpyDeviceToHost));
}

size_t PositionWise::getSerializationSize() const {
  return sizeof(int) * 3 + Seql * n_Feat * type2size(ctype);
}

const char* PositionWise::getPluginNamespace() const {
  return mNameSpace.c_str();
}

void PositionWise::setPluginNamespace(const char* pluginNamespace) {
  mNameSpace = pluginNamespace;
}

PositionWisePluginCreator::PositionWisePluginCreator() {

}
PositionWisePluginCreator::~PositionWisePluginCreator() {

}

const char* PositionWisePluginCreator::getPluginName() const {
  return POSITION_WISE_PLUGIN_NAME;
}

const char* PositionWisePluginCreator::getPluginVersion() const {
  return POSITION_WISE_PLUGIN_VERSION;
}

const PluginFieldCollection* PositionWisePluginCreator::getFieldNames() {
  return &mFC;
}

IPluginV2DynamicExt* PositionWisePluginCreator::createPlugin(
  const char* name, const PluginFieldCollection* fc) {
  int seql = ((int*)fc->fields[0].data)[0];
  int nfeat = ((int*)fc->fields[0].data)[1];
  DataType dtype = (DataType)((int*)fc->fields[0].data)[2];

  Weights pos{ DataType::kFLOAT,fc->fields[1].data, fc->fields[1].length };
  return (IPluginV2DynamicExt*)new PositionWise(seql, nfeat, dtype, pos);
}

IPluginV2DynamicExt* PositionWisePluginCreator::deserializePlugin(
  const char* name, const void* serialData, size_t serialLength) {
  return (IPluginV2DynamicExt*)new PositionWise(serialData, serialLength);
}