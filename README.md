# Onnx Tutorial

1. `onnx_export.py` 把一个pytorch模型导出为一个.onnx模型。

2. 使用c++加载模型并前向推理。编译并运行：

   ```bash
   mkdir build && cd build
   cmake .. & make
   ./onnx_tutorial ../mlp_model.onnx
   ```

   

