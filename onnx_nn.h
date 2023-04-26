#ifndef ONNX_PARALLEL_ONNX_NN_H_
#define ONNX_PARALLEL_ONNX_NN_H_

#include "onnxruntime_cxx_api.h"

class OnnxSession {
 public:
  explicit OnnxSession(std::string device = "cpu");
  virtual ~OnnxSession() { delete core_; }

  void load_model(const char *model_path);
  inline void set_input(std::initializer_list<const char *> input_names);
  inline void set_output(std::initializer_list<const char *> output_names);
  Ort::Value *do_inference(Ort::Value inputs[]);
  Ort::Value *do_inference(std::initializer_list<Ort::Value> inputs);

  template<typename T>
  inline Ort::Value create_tensor(
      T *p_data, size_t p_data_size, const int64_t *shape, size_t shape_len) {
    return Ort::Value::CreateTensor<T>(
        memory_info_, p_data, p_data_size, shape, shape_len);
  }
  inline Ort::Value create_input_tensor(size_t index, float *p_data);
  Ort::Value *create_input_tensors(std::initializer_list<float *> data);
  Ort::Value *get_output_buffer() { return outputs_.data(); }

 protected:
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<int64_t> input_sizes_;
 private:
  std::string device_;
  Ort::MemoryInfo memory_info_{nullptr};
  Ort::RunOptions run_options_{nullptr};
  Ort::Session *core_{nullptr};
  size_t num_inputs_ = 0, num_outputs_ = 0;
  std::vector<const char *> input_names_, output_names_;
  std::vector<Ort::Value> inputs_, outputs_;
};

class Mlp : private OnnxSession {
 public:
  explicit Mlp(const std::string &model_path,
                const std::string &device = "cpu");
  ~Mlp() override = default;

  Mlp(Mlp &) = delete;
  Mlp(const Mlp &) = delete;
  const std::vector<int64_t> &get_input_sizes() { return input_sizes_; }
  Ort::Value *infer(float *obs);
};

class Lstm : private OnnxSession {
 public:
  explicit Lstm(const std::string &model_path,
                const std::string &device = "cpu");
  ~Lstm() override = default;

  Lstm(Lstm &) = delete;
  Lstm(const Lstm &) = delete;
  const std::vector<int64_t> &get_input_sizes() { return input_sizes_; }
  Ort::Value *infer(float *obs, float *h0, float *c0);
};

#endif //ONNX_PARALLEL_ONNX_NN_H_
