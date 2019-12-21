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
#include "layerNormalizationPlugin.h"
#include "cublas_v2.h"
#include <cstring>
#include <cudnn.h>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::LayerNormalizaiton;
using nvinfer1::plugin::LayerNormalizaitonPluginCreator;

namespace
{
const char* LAYER_NORMALIZATION_PLUGIN_VERSION{"001"};
const char* LAYER_NORMALIZATION_PLUGIN_NAME{"LayerNormalization_TRT"};
} // namespace

PluginFieldCollection LayerNormalizaitonPluginCreator::mFC{};
std::vector<PluginField> LayerNormalizaitonPluginCreator::mPluginAttributes;

LayerNormalizaiton::LayerNormalizaiton(const Weights gamma, const Weights beta) {
  ASSERT(gamma.type == DataType::kFLOAT);
  ASSERT(beta.type == DataType::kFLOAT);
  ASSERT(beta.count == gamma.count)
  dim = gamma.count;
  CHECK(cudaMalloc(&gamma_, dim * sizeof(float)));
  CHECK(cudaMalloc(&beta_, dim * sizeof(float)));

  CHECK(cudaMemcpy(gamma_, gamma.values, sizeof(float)*dim, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(beta_, beta.values, sizeof(float)*dim, cudaMemcpyHostToDevice));
}

LayerNormalizaiton::LayerNormalizaiton(const LayerNormalizaiton& other) {
  dim = other.dim;
  CHECK(cudaMalloc(&gamma_, dim * sizeof(float)));
  CHECK(cudaMalloc(&beta_, dim * sizeof(float)));

  CHECK(cudaMemcpy(gamma_, other.gamma_, sizeof(float)*dim, cudaMemcpyDeviceToDevice));
  CHECK(cudaMemcpy(beta_, other.beta_, sizeof(float)*dim, cudaMemcpyDeviceToDevice));
}

LayerNormalizaiton::LayerNormalizaiton(const void* buffer, size_t length) {
  const char* ptr = (const char*)buffer;
  dim = read<int>(ptr);
  ctype = (DataType)read<int>(ptr);
  CHECK(cudaMalloc(&gamma_, dim * sizeof(float)));
  CHECK(cudaMalloc(&beta_, dim * sizeof(float)));

  CHECK(cudaMemcpy(gamma_, ptr, sizeof(float)*dim, cudaMemcpyDeviceToDevice));
  CHECK(cudaMemcpy(beta_, ptr + sizeof(float)*dim, sizeof(float)*dim, cudaMemcpyDeviceToDevice));
}

IPluginV2DynamicExt* LayerNormalizaiton::clone()const {
  return (IPluginV2DynamicExt*)new LayerNormalizaiton(*this);
}

DimsExprs LayerNormalizaiton::getOutputDimensions(
  int outputIndex, const nvinfer1::DimsExprs* inputs,
  int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
  ASSERT(nbInputs == 1);
  DimsExprs output(inputs[0]);
  return output;
}

bool LayerNormalizaiton::supportsFormatCombination(
  int pos, const nvinfer1::PluginTensorDesc* inOut,
  int nbInputs, int nbOutputs) {
  ASSERT(nbInputs == 1);
  if (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF)
    return true;
  else return false;
}

void LayerNormalizaiton::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
  const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
  ASSERT(nbInputs == 1);
  ctype = in[0].desc.type;
}

size_t LayerNormalizaiton::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
  const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
  return 0;
}

int LayerNormalizaiton::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
  const nvinfer1::PluginTensorDesc* outputDesc,
  const void* const* inputs, void* const* outputs,
  void* workspace, cudaStream_t stream) {


  return 0;
}

DataType LayerNormalizaiton::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  return ctype;
}

const char* LayerNormalizaiton::getPluginVersion() const{
  return LAYER_NORMALIZATION_PLUGIN_VERSION;
}
const char* LayerNormalizaiton::getPluginType() const {
  return LAYER_NORMALIZATION_PLUGIN_NAME;
}

int LayerNormalizaiton::getNbOutputs() const {
  return 1;
}

int LayerNormalizaiton::initialize() {
  return 0;
}

void LayerNormalizaiton::terminate() {}

void LayerNormalizaiton::destroy() {}

void LayerNormalizaiton::serialize(void* buffer) const {
  char* ptr = (char*)buffer;
  write<int>(ptr, dim);
  write<int>(ptr, (int)ctype);

  CHECK(cudaMemcpy(ptr, gamma_, sizeof(float)*dim, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(ptr + sizeof(float)*dim, beta_, sizeof(float)*dim, cudaMemcpyDeviceToHost));
}

size_t LayerNormalizaiton::getSerializationSize() const {
  return sizeof(int) * 2 + sizeof(float) * dim * 2;
}

const char* LayerNormalizaiton::getPluginNamespace() const {
  return mNameSpace.c_str();
}

void LayerNormalizaiton::setPluginNamespace(const char* pluginNamespace) {
  mNameSpace = pluginNamespace;
}

LayerNormalizaitonPluginCreator::LayerNormalizaitonPluginCreator() {

}
LayerNormalizaitonPluginCreator::~LayerNormalizaitonPluginCreator() {

}

const char* LayerNormalizaitonPluginCreator::getPluginName() const {
  return LAYER_NORMALIZATION_PLUGIN_NAME;
}

const char* LayerNormalizaitonPluginCreator::getPluginVersion() const {
  return LAYER_NORMALIZATION_PLUGIN_VERSION;
}

const PluginFieldCollection* LayerNormalizaitonPluginCreator::getFieldNames() {
  return &mFC;
}

IPluginV2DynamicExt* LayerNormalizaitonPluginCreator::createPlugin(
  const char* name, const PluginFieldCollection* fc) {
  Weights gamma, beta;
  ASSERT(fc->nbFields == 2);
  gamma.type = DataType::kFLOAT;
  gamma.count = fc->fields[0].length;
  gamma.values = fc->fields[0].data;

  beta.type = DataType::kFLOAT;
  beta.count = fc->fields[1].length;
  beta.values = fc->fields[1].data;

  return (IPluginV2DynamicExt*)new LayerNormalizaiton(gamma, beta);
}

IPluginV2DynamicExt* LayerNormalizaitonPluginCreator::deserializePlugin(
  const char* name, const void* serialData, size_t serialLength) {
  return (IPluginV2DynamicExt*)new LayerNormalizaiton(serialData, serialLength);
}