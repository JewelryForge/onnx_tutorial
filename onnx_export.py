import torch

torch.manual_seed(2023)

network = torch.nn.Sequential(
    torch.nn.Linear(10, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
)

dummy_input = torch.ones(1, 10)
with torch.inference_mode():
    print(network(dummy_input))  # [[-0.0622]]

torch.onnx.export(
    torch.jit.script(network),
    dummy_input, 'mlp_model.onnx',
    input_names=['x'], output_names=['y'],
    verbose=True,
)
