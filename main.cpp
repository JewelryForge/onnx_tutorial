#include <iostream>
#include "onnx_nn.h"

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Usage: onnx_tutorial mlp_model.onnx" << std::endl;
    return 1;
  }

  Mlp mlp(argv[1]);

  float x[10]{0.};
  for (int i = 0; i < 10; ++i) x[i] = 1.;
  auto *y = mlp.infer(x);

  auto *data = y->GetTensorMutableData<float>();
  std::cout << data[0] << std::endl; // -0.0622

  return 0;
}
