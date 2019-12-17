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
#include "embeddingPlugin.h"
#include "cublas_v2.h"
#include <cstring>
#include <cudnn.h>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::EmbeddingExpr;
using nvinfer1::plugin::Embedding;
using nvinfer1::plugin::EmbeddingPluginCreator;

namespace
{
  const char* EMBEDDING_PLUGIN_VERSION{ "1" };
  const char* EMBEDDING_PLUGIN_NAME{ "Embedding_TRT" };
} // namespace

PluginFieldCollection EmbeddingPluginCreator::mFC{};
std::vector<PluginField> EmbeddingPluginCreator::mPluginAttributes;

Embedding::Embedding(const Weights weight, const DimsHW shape) {
  nvocab = shape.h();
  embed = shape.w();

  CHECK(cudaMalloc(&embedding, weight.count * sizeof(float)));
  CHECK(cudaMemcpy(embedding, weight.values, sizeof(float)*weight.count, cudaMemcpyHostToDevice));
}

Embedding::Embedding(const Embedding& other) {
  nvocab = other.nvocab;
  embed = other.embed;
  ctype = other.ctype;

  CHECK(cudaMalloc(&embedding, embed*nvocab * sizeof(float)));
  CHECK(cudaMemcpy(embedding, other.embedding, sizeof(float)*nvocab*embed, cudaMemcpyHostToDevice));
}

Embedding::Embedding(const void* buffer, size_t length) {
  const char* ptr = (const char*)buffer;
  nvocab = read<int>(ptr);
  embed = read<int>(ptr);
  ctype = (DataType)read<int>(ptr);

  CHECK(cudaMalloc(&embedding, embed*nvocab * sizeof(float)));
  CHECK(cudaMemcpy(embedding, ptr, sizeof(float)*nvocab*embed, cudaMemcpyHostToDevice));
}

IPluginV2DynamicExt* Embedding::clone()const {
  return (IPluginV2DynamicExt*)new Embedding(*this);
}

DimsExprs Embedding::getOutputDimensions(
  int outputIndex, const nvinfer1::DimsExprs* inputs,
  int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
  ASSERT(nbInputs == 1);
  DimsExprs dims;
  dims.nbDims = 2;
  dims.d[0] = inputs[0].d[0];
  dims.d[1] = new EmbeddingExpr(embed);
  return dims;
}

bool Embedding::supportsFormatCombination(
  int pos, const nvinfer1::PluginTensorDesc* inOut,
  int nbInputs, int nbOutputs) {
  ASSERT(nbInputs == 1);
  if (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF)
    return true;
  else return false;
}

void Embedding::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
  const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
  ASSERT(nbInputs == 1);
  ctype = in[0].desc.type;
}

size_t Embedding::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
  const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
  return 0;
}

int Embedding::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
  const nvinfer1::PluginTensorDesc* outputDesc,
  const void* const* inputs, void* const* outputs,
  void* workspace, cudaStream_t stream) {


  return 0;
}

DataType Embedding::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  return ctype;
}

const char* Embedding::getPluginVersion() const {
  return EMBEDDING_PLUGIN_VERSION;
}
const char* Embedding::getPluginType() const {
  return EMBEDDING_PLUGIN_NAME;
}

int Embedding::getNbOutputs() const {
  return 1;
}

int Embedding::initialize() {
  return 0;
}

void Embedding::terminate() {}

void Embedding::destroy() {}

void Embedding::serialize(void* buffer) const {
  char* ptr = (char*)buffer;
  write<int>(ptr, nvocab);
  write<int>(ptr, embed);
  write<int>(ptr, (int)ctype);
  CHECK(cudaMemcpy(buffer, embedding, sizeof(float)*nvocab*embed, cudaMemcpyDeviceToHost));
}

size_t Embedding::getSerializationSize() const {
  return sizeof(int) * 3 + sizeof(float) * nvocab * embed;
}

const char* Embedding::getPluginNamespace() const {
  return mNameSpace.c_str();
}

void Embedding::setPluginNamespace(const char* pluginNamespace) {
  mNameSpace = pluginNamespace;
}

EmbeddingPluginCreator::EmbeddingPluginCreator() {

}
EmbeddingPluginCreator::~EmbeddingPluginCreator() {

}

const char* EmbeddingPluginCreator::getPluginName() const {
  return EMBEDDING_PLUGIN_NAME;
}

const char* EmbeddingPluginCreator::getPluginVersion() const {
  return EMBEDDING_PLUGIN_VERSION;
}

const PluginFieldCollection* EmbeddingPluginCreator::getFieldNames() {
  return &mFC;
}

IPluginV2DynamicExt* EmbeddingPluginCreator::createPlugin(
  const char* name, const PluginFieldCollection* fc) {
  Weights embed;
  DimsHW shape;
  ASSERT(fc->nbFields == 2);
  embed.type = DataType::kFLOAT;
  embed.count = fc[0].fields->length;
  embed.values = fc[0].fields->data;

  shape.d[0] = ((int*)fc[1].fields)[0];
  shape.d[1] = ((int*)fc[1].fields)[1];

  return (IPluginV2DynamicExt*)new Embedding(embed, shape);
}

IPluginV2DynamicExt* EmbeddingPluginCreator::deserializePlugin(
  const char* name, const void* serialData, size_t serialLength) {
  return (IPluginV2DynamicExt*)new Embedding(serialData, serialLength);
}