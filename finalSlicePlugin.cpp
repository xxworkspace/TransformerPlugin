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
#include "finalSlicePlugin.h"
#include "cublas_v2.h"
#include <cstring>
#include <cudnn.h>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::FinalSlice;
using nvinfer1::plugin::FinalSlicePluginCreator;

namespace
{
  const char* FINAL_SLICE_PLUGIN_VERSION{ "001" };
  const char* FINAL_SLICE_PLUGIN_NAME{ "FinalSlice_TRT" };
} // namespace

PluginFieldCollection FinalSlicePluginCreator::mFC{};
std::vector<PluginField> FinalSlicePluginCreator::mPluginAttributes;

FinalSlice::FinalSlice() {}

FinalSlice::FinalSlice(const FinalSlice& other) {}

FinalSlice::FinalSlice(const void* buffer, size_t length) {
  const char* ptr = (const char*)buffer;
  ctype = (DataType)read<int>(ptr);
}

IPluginV2DynamicExt* FinalSlice::clone()const {
  return (IPluginV2DynamicExt*)new FinalSlice(*this);
}

DimsExprs FinalSlice::getOutputDimensions(
  int outputIndex, const nvinfer1::DimsExprs* inputs,
  int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
  ASSERT(nbInputs == 1);
  DimsExprs dims;
  dims.nbDims = 1;
  dims.d[0] = inputs[0].d[1];
  return dims;
}

bool FinalSlice::supportsFormatCombination(
  int pos, const nvinfer1::PluginTensorDesc* inOut,
  int nbInputs, int nbOutputs) {
  ASSERT(nbInputs == 1);
  if (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF)
    return true;
  else return false;
}

void FinalSlice::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
  const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
  ASSERT(nbInputs == 1);
  ctype = in[0].desc.type;
}

size_t FinalSlice::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
  const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
  return 0;
}

int FinalSlice::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
  const nvinfer1::PluginTensorDesc* outputDesc,
  const void* const* inputs, void* const* outputs,
  void* workspace, cudaStream_t stream) {


  return 0;
}

DataType FinalSlice::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  return ctype;
}

const char* FinalSlice::getPluginVersion() const {
  return FINAL_SLICE_PLUGIN_VERSION;
}
const char* FinalSlice::getPluginType() const {
  return FINAL_SLICE_PLUGIN_NAME;
}

int FinalSlice::getNbOutputs() const {
  return 1;
}

int FinalSlice::initialize() {
  return 0;
}

void FinalSlice::terminate() {}

void FinalSlice::destroy() {}

void FinalSlice::serialize(void* buffer) const {
  char* ptr = (char*)buffer;
  write<int>(ptr, (int)ctype);
}

size_t FinalSlice::getSerializationSize() const {
  return sizeof(int);
}

const char* FinalSlice::getPluginNamespace() const {
  return mNameSpace.c_str();
}

void FinalSlice::setPluginNamespace(const char* pluginNamespace) {
  mNameSpace = pluginNamespace;
}

FinalSlicePluginCreator::FinalSlicePluginCreator() {

}
FinalSlicePluginCreator::~FinalSlicePluginCreator() {

}

const char* FinalSlicePluginCreator::getPluginName() const {
  return FINAL_SLICE_PLUGIN_NAME;
}

const char* FinalSlicePluginCreator::getPluginVersion() const {
  return FINAL_SLICE_PLUGIN_VERSION;
}

const PluginFieldCollection* FinalSlicePluginCreator::getFieldNames() {
  return &mFC;
}

IPluginV2DynamicExt* FinalSlicePluginCreator::createPlugin(
  const char* name, const PluginFieldCollection* fc) {
  return (IPluginV2DynamicExt*)new FinalSlice();
}

IPluginV2DynamicExt* FinalSlicePluginCreator::deserializePlugin(
  const char* name, const void* serialData, size_t serialLength) {
  return (IPluginV2DynamicExt*)new FinalSlice(serialData, serialLength);
}