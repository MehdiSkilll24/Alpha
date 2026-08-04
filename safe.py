import torch
from safetensors.torch import save_file

pt_path = r"C:\Users\mehdi\Desktop\Pythonfiles\Projects\Transformers\Tentative results\R4checkpoint_latest.pt"
output_path = r"C:\Users\mehdi\Desktop\Pythonfiles\Projects\Transformers\Tentative results\TR4_model.safetensors"

# Load PyTorch checkpoint
checkpoint = torch.load(pt_path, map_location="cpu")

# Handle common checkpoint formats
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Convert tensors
state_dict = {
    k: v.contiguous()
    for k, v in state_dict.items()
    if isinstance(v, torch.Tensor)
}

# Save as safetensors
save_file(state_dict, output_path)

print(f"Converted successfully:")
print(output_path)