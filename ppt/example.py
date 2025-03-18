import numpy as np
import torch
from ppt.shuffler import PPT
from ppt.metric import ACF_COS

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test with CPU first
    device = torch.device("cpu")
    
    ppt = PPT(channel_num=2, original_time_len=160, patch_len=16, permute_freq=2, device=device)
    input_tensor = torch.randn(2, 2, 16, 10, device=device) # (Batch, Channel, Embedding, Patch Number)
    output = ppt(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input device: {input_tensor.device}")
    print(f"Output device: {output.device}")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nTesting with CUDA:")
        
        ppt = PPT(channel_num=2, original_time_len=160, patch_len=16, permute_freq=2, device=device)
        input_tensor = torch.randn(2, 2, 16, 10, device=device)
        output = ppt(input_tensor)
        
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Input device: {input_tensor.device}")
        print(f"Output device: {output.device}")