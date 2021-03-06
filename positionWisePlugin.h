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
#ifndef TRT_POSITION_WISE_PLUGIN_H
#define TRT_POSITION_WISE_PLUGIN_H

#include <vector>
#include <cudnn.h>
#include "plugin.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>

namespace nvinfer1
{
  namespace plugin
  {
    class PositionWise : public IPluginV2DynamicExt
    {
    public:
      PositionWise(const int seql,const int nfeat,const DataType dtype,Weights pos);

      PositionWise(const PositionWise&);

      PositionWise(const void* buffer, size_t length);

      ~PositionWise() override = default;

      // IPluginV2DynamicExt Methods
      nvinfer1::IPluginV2DynamicExt* clone() const override;

      nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;

      bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut,
        int nbInputs, int nbOutputs) override;

      void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;

      size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;

      int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) override;

      // IPluginV2Ext Methods
      nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

      // IPluginV2 Methods
      const char* getPluginVersion() const override;
      const char* getPluginType() const override;
      int getNbOutputs() const override;
      int initialize() override;
      void terminate() override;
      void destroy() override;

      void serialize(void* buffer) const override;
      size_t getSerializationSize() const override;
      const char* getPluginNamespace() const override;
      void setPluginNamespace(const char* pluginNamespace) override;
    private:
      size_t type2size(DataType dtype)const {
        if (dtype == DataType::kFLOAT || dtype == DataType::kINT32)
          return 4;
        else if (dtype == DataType::kHALF)
          return 2;
        else
          return 1;
      }
      DataType ctype{ DataType::kFLOAT };
      std::string mNameSpace{ "" };
      void* position{ NULL };
      int Seql, n_Feat;

      //cudnn compute
      cudnnHandle_t handle;
      cudnnTensorDescriptor_t xdes;
      cudnnOpTensorDescriptor_t opdes;
    };

    class PositionWisePluginCreator : public BaseCreator
    {
    public:
      PositionWisePluginCreator();

      ~PositionWisePluginCreator() override;

      const char* getPluginName() const override;

      const char* getPluginVersion() const override;

      const PluginFieldCollection* getFieldNames() override;

      IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

      IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    private:
      static PluginFieldCollection mFC;
      static std::vector<PluginField> mPluginAttributes;
    };
  } // namespace plugin
} // namespace nvinfer1

#endif // TRT_PRIOR_BOX_PLUGIN_H
