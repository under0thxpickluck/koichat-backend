import torch
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print("CUDA is available.")
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPU(s).")
    for i in range(device_count):
        print(f"--- GPU {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"Compute Capability: {props.major}.{props.minor}")
        # このチェックが極めて重要
        if props.major < 12:
            print("WARNING: GPU Compute Capability is less than 12.0. This is not a Blackwell GPU.")
else:
    print("ERROR: CUDA is not available. Check driver and toolkit installation.")