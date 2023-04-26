#include <cassert>
#include <iostream>
#include "onnx_nn.h"

OnnxSession::OnnxSession(std::string device)
    : device_(std::move(device)) {
}

void OnnxSession::load_model(const char *model_path) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "policy_network");
  Ort::SessionOptions session_options;
  if (device_ == "cpu") {
  } else {
    session_options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
  }
  memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  core_ = new Ort::Session(env, model_path, session_options);
  num_inputs_ = core_->GetInputCount();
  num_outputs_ = core_->GetOutputCount();
  for (int i = 0; i < num_inputs_; ++i) {
    auto input_info = core_->GetInputTypeInfo(i);
    auto type_info = input_info.GetTensorTypeAndShapeInfo();
    input_shapes_.emplace_back(std::move(type_info.GetShape()));
    int64_t size = 1;
    for (auto dim : input_shapes_[i]) if (dim != -1) size *= dim;
    input_sizes_.push_back(size);
//      std::cout << "Input " << i << " Dimensions: " << input_shapes_[i] << "\n";
  }
}

void OnnxSession::set_input(std::initializer_list<const char *> input_names) {
  assert(input_names.size() == num_inputs_);
  input_names_ = input_names;
  inputs_.clear();
  for (int i = 0; i < input_names.size(); ++i) {
    inputs_.emplace_back(nullptr);
  }
}

void OnnxSession::set_output(std::initializer_list<const char *> output_names) {
  assert(output_names.size() == num_outputs_);
  output_names_ = output_names;
  outputs_.clear();
  for (int i = 0; i < output_names.size(); ++i) {
    outputs_.emplace_back(nullptr);
  }
}

Ort::Value *OnnxSession::do_inference(Ort::Value *inputs) {
  core_->Run(run_options_, input_names_.data(), inputs, input_names_.size(),
             output_names_.data(), outputs_.data(), outputs_.size());
  return outputs_.data();
}

Ort::Value *OnnxSession::do_inference(std::initializer_list<Ort::Value> inputs) {
  core_->Run(run_options_, input_names_.data(), inputs.begin(), input_names_.size(),
             output_names_.data(), outputs_.data(), outputs_.size());
  return outputs_.data();
}

Ort::Value OnnxSession::create_input_tensor(size_t index, float *p_data) {
  const auto &shape = input_shapes_[index];
  return Ort::Value::CreateTensor<float>(
      memory_info_, p_data, input_sizes_[index], shape.data(), shape.size());
}

Ort::Value *OnnxSession::create_input_tensors(std::initializer_list<float *> data) {
  assert(data.size() == num_inputs_);
  auto data_ptr = data.begin();
  for (int i = 0; i < num_inputs_; ++i, ++data_ptr) {
    inputs_[i] = create_input_tensor(i, *data_ptr);
  }
  return inputs_.data();
}

Mlp::Mlp(const std::string &model_path, const std::string &device)
    : OnnxSession(device) {
  load_model(model_path.c_str());
  set_input({"x"});
  set_output({"y"});
}

Ort::Value *Mlp::infer(float *x) {
  return do_inference(create_input_tensors({x}));
}

Lstm::Lstm(const std::string &model_path, const std::string &device)
    : OnnxSession(device) {
  load_model(model_path.c_str());
  set_input({"x", "h0", "c0"});
  set_output({"y", "hn", "cn"});
}

Ort::Value *Lstm::infer(float *x, float *h0, float *c0) {
  return do_inference(create_input_tensors({x, h0, c0}));
}
