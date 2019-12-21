
#include "logger.h"
#include "NvInfer.h"
#include "finalSlicePlugin.h"
#include "layerNormalizationPlugin.h"
using namespace nvinfer1;

void registryFinalSlicePlugin() {
  IPluginRegistry *registry = getPluginRegistry();
  IPluginCreator *creator = new nvinfer1::plugin::FinalSlicePluginCreator();
  registry->registerCreator(*creator, "");
}

void registryLayerNormalizationPlugin() {
  IPluginRegistry *registry = getPluginRegistry();
  IPluginCreator *creator = new nvinfer1::plugin::LayerNormalizaitonPluginCreator();
  registry->registerCreator(*creator, "");
}

void logTensorInfo(ITensor* tensor, std::string name) {
  Dims dims = tensor->getDimensions();
  std::cout << name << " -> ";
  for (int i = 0; i < dims.nbDims; ++i)
    std::cout << dims.d[i] << " ";
  std::cout << std::endl;
}

template<class T>
T* GetRandomData(int row,int col) {
  T* data = new T[row*col];
  return data;
}

ITensor* AddFinalSlice(
  INetworkDefinition *network,
  ITensor* input){
  auto creator = getPluginRegistry()->getPluginCreator(
    "FinalSlice_TRT", "001", "");

  PluginFieldCollection pfc;
  IPluginV2 *plugin = creator->createPlugin("", &pfc);
  auto bottom = network->addPluginV2(&input, 1, *plugin)->getOutput(0);

  return bottom;
}

ITensor *AddLayerNormalization(
  INetworkDefinition *network,
  ITensor* input) {
  auto creator = getPluginRegistry()->getPluginCreator(
    "LayerNormalization_TRT", "001", "");

  std::vector<PluginField> vpf{
    PluginField{"gamma", (void*)GetRandomData<float>(100,1), PluginFieldType::kFLOAT32, (int32_t)100},
    PluginField{"beta",  (void*)GetRandomData<float>(100,1), PluginFieldType::kFLOAT32, (int32_t)100}
  };

  PluginFieldCollection pfc{ vpf.size(), vpf.data() };
  IPluginV2 *plugin = creator->createPlugin("", &pfc);
  auto bottom = network->addPluginV2(&input, 1, *plugin)->getOutput(0);
  return bottom;
}

int main() {
  Logger logger;
  IBuilder* builder = createInferBuilder(logger.getTRTLogger());
  INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

  registryFinalSlicePlugin();
  registryLayerNormalizationPlugin();

  auto input = network->addInput("input", DataType::kINT32, Dims{1 , -1});
  Weights embed{ DataType::kFLOAT,GetRandomData<float>(1000,100),1000 * 100 };
  auto em = network->addConstant(DimsHW(1000,100),embed)->getOutput(0);
  auto bottom = network->addGather(*em,*input,0)->getOutput(0);
  bottom = AddLayerNormalization(network, bottom);
  bottom = AddFinalSlice(network, bottom);
  //auto dim = network->addInput("dim", DataType::kINT32, Dims{ 1, 100 });
  //auto input = network->addInput("input", DataType::kINT32, Dims{1 , -1});
  //auto input = network->addInput("input", DataType::kFLOAT, DimsCHW(3,-1, -1));
  //logTensorInfo(input,"input");
  //network->addGather();
  //auto bottom = network->addActivation(*input,ActivationType::kSIGMOID)->getOutput(0);
  //auto bottom = AddEmbedding(network, input, dim);
  bottom->setName("output");
  logTensorInfo(bottom, "output");
  network->markOutput(*bottom);

  IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions("input", OptProfileSelector::kMIN, Dims{ 1, 1 });
  profile->setDimensions("input", OptProfileSelector::kOPT, Dims{ 1, 64 });
  profile->setDimensions("input", OptProfileSelector::kMAX, Dims{ 1, 128 });

  IBuilderConfig* config = builder->createBuilderConfig();
  config->addOptimizationProfile(profile);
  config->setMaxWorkspaceSize(1<30);

  builder->setMaxBatchSize(16);
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

  IHostMemory* trtModelStream = engine->serialize();

  return 0;
}