import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Get the number of available GPUs
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Get the name of the first GPU
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")