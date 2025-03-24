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

    batch_size = 2
    channel_num = 1
    time_len = 10
    patch_len =5
    permute_freq=1
    assert time_len % patch_len == 0, "time_len must be divisible by patch_len"
    patch_num = time_len // patch_len 
    ppt = PPT(channel_num=channel_num, original_time_len=time_len, patch_len=patch_len, permute_freq=permute_freq, device=device)
    input_tensor = torch.arange(batch_size * channel_num * patch_len * patch_num).reshape(batch_size, channel_num, patch_len, patch_num)
    # E.g. Print the first patch. print(input_tensor[0,0,:,0]) -> tensor([0, 2, 4, 6, 8])
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
        
    
    # Test ACF_COS with huge tensor samples (multiprocessing=True)
    print("\nTesting ACF_COS with huge tensor samples:")
    print("On CPU (White noise):")
    device = torch.device("cpu")
    input_tensor = torch.randn(10, 873, 160, device=device)  # batch_size, channel_num, time_len
    acf_cos = ACF_COS(channel_num=873, original_time_len=160, patch_len=16, permute_freq=2, device=device, permute_tensor_size=1000, multiprocessing=True, num_workers=8)
    mean, std = acf_cos(input_tensor)
    print(f"Mean: {mean}, Std: {std}")
    
    print("\nOn CUDA (Step function):")
    device = torch.device("cuda")
    input_tensor = torch.arange(10 * 873 * 160).reshape(10, 873, 160).to(device)  # batch_size, channel_num, time_len
    acf_cos = ACF_COS(channel_num=873, original_time_len=160, patch_len=16, permute_freq=2, device=device, permute_tensor_size=1000, multiprocessing=True, num_workers=8)
    mean, std = acf_cos(input_tensor)
    print(f"Mean: {mean}, Std: {std}")
    
    