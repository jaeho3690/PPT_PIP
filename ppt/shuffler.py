import numpy as np
import torch
import torch.nn as nn
from typing import Union, Dict, List, Tuple, Optional
import random
import os
import time
from einops import rearrange

class PPTShuffle(nn.Module):
    """
    Patch Permutation Transform class for permuting data patches.
    
    Args:
        channel_num (int): Number of channels in the data.
        original_time_len (int): Original time length of the data.
        patch_len (int): Length of each patch
        permute_freq (int): Number of permutations to apply
        permute_strategy (str, optional): Strategy for permutation. Options: "random", "vicinity", "farthest"
        permute_tensor_size (int, optional): Number of precomputed permutation patterns. Defaults to 1000.
        save_path (str, optional): Path to save/load precomputed permutation indices
        device (str, optional): Device to run the model on. Defaults to "cpu".
    """
    
    def __init__(
        self,
        channel_num: int,
        original_time_len: int,
        patch_len: int,
        permute_freq: int,
        permute_strategy: str = "random",
        permute_tensor_size: int = 1000,
        save_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        super().__init__()

            
        self.channel_num = channel_num
        self.original_time_len = original_time_len
        self.patch_len = patch_len
        self.patch_num = int(original_time_len / patch_len)
        self.permute_freq = permute_freq
        self.permute_strategy = permute_strategy
        self.permute_tensor_size = permute_tensor_size
        self.save_path = save_path
        
        self.sample_data = torch.randn(1, self.channel_num, 1, self.patch_num)
    
        # Validate input parameters
        self._validate_inputs()
        
        # Set device automatically if not specified
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize permutation index tensor
        self.permutation_index_tensor = self._build_permutation_index_tensor()
        # Move permutation index tensor to the correct device
        self.permutation_index_tensor = self.permutation_index_tensor.to(self.device)

    def forward(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        assert len(X.shape) == 4, f"Input data must be 4D, got {len(X.shape)}D"
        assert X.shape[1] == self.channel_num, f"Input data must have the same number of channels as sample_data, got {X.shape[1]} channels"
        assert X.shape[3] == self.patch_num, f"Input data must have the same number of patches as sample_data, got {X.shape[3]} patches"
        assert X.device == self.device, f"Input data and sample_data must be on the same device, got input data on {X.device} and PPT on {self.device}"
        
        X = rearrange(X, 'b c e p -> b e c p')
        random_idx = torch.randint(0, self.permute_tensor_size - 1, (1,)).to(self.device)
        shuffled_idx = self.permutation_index_tensor[random_idx] 
        expanded_shuffled_idx = shuffled_idx.expand(X.shape[0], self.channel_num, -1, -1)  # expand idx to match X shape
        # print device type of expanded_shuffled_idx, X, and X_prime
        X_prime = torch.gather(X, 3, expanded_shuffled_idx) # X_prime is the permuted index
        X_shuffled = rearrange(X_prime, 'b e c p -> b c e p')
        return X_shuffled
        
       
    
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if self.patch_len <= 0:
            raise ValueError("patch_len must be positive")
        
        if self.permute_freq <= 0:
            raise ValueError("permute_freq must be positive")
            
        if self.permute_tensor_size <= 0:
            raise ValueError("permute_tensor_size must be positive")
        
        # sample_data should be 4D
        if len(self.sample_data.shape) != 4:
            # Pass in any random sample data to check the shape
            raise ValueError(f"sample_data must be of shape (batch, channel, embedding, patch_length), got {self.sample_data.shape}")
    
        
        if self.original_time_len < self.patch_len:
            raise ValueError(
                f"Data length ({self.original_time_len}) must be greater than or equal to "
                f"patch_len ({self.patch_len})"
            )
        # should be divisible by patch_len
        if self.original_time_len % self.patch_len != 0:
            raise ValueError(
                f"Data length ({self.original_time_len}) must be divisible by patch_len ({self.patch_len})"
            )
        # permute_strategy should be one of the following: "random", "vicinity", "farthest"
        if self.permute_strategy not in ["random", "vicinity", "farthest"]:
            raise ValueError(f"permute_strategy must be one of the following: 'random', 'vicinity', 'farthest'")
    
    def _build_permutation_index_tensor(self) -> torch.Tensor:
        """Build the permutation tensor."""
        # check if save_path exists
        if self.save_path is None:
            # create a new save_path
            self.save_path = "shuffled_index"
            # create the directory if it doesn't exist
            os.makedirs(self.save_path, exist_ok=True)
            # create a new file
            self.save_path = os.path.join(self.save_path, 
                                          f"shuffled_index_{self.permute_strategy}_{self.permute_freq}_{self.patch_len}_{str(self.channel_num).zfill(2)}_{str(self.patch_num).zfill(2)}.pt")
        
        # check if the file exists
        if os.path.exists(self.save_path):
            # load the file
            print(f"Loading permutation tensor from {self.save_path}")
            permutation_index_tensor = torch.load(self.save_path)
        else:
            # create a new file
            permutation_index_tensor = self._create_permutation_index_tensor()
            # Save the tensor
            torch.save(permutation_index_tensor, self.save_path)
            print(f"Saved permutation tensor to {self.save_path}")

        return permutation_index_tensor

    def _create_permutation_index_tensor(self) -> torch.Tensor:
        """Create the permutation tensor if the file does not exist."""
        print(f"Creating permutation tensor at {self.save_path}, with strategy {self.permute_strategy}, frequency {self.permute_freq}, and patch length {self.patch_len}")

        index_data = np.arange(self.patch_num)

        total_shuffled_idx = [] # list of {permute_tensor_size} variation of index_data
        for _ in range(self.permute_tensor_size):
            single_sample_shuffled_idx_list = []
            for _ in range(self.channel_num):
                copy_index_data = index_data.copy()

                for _ in range(self.permute_freq):
                    if self.permute_strategy == "random":
                        idx1, idx2 = np.random.randint(0, self.patch_num, size=2)
                    elif self.permute_strategy == "vicinity":
                        idx1 = np.random.randint(0, len(copy_index_data) - 2)
                        idx2 = idx1 + 1 
                    elif self.permute_strategy == "farthest":
                        idx1, idx2 = random.sample(range(self.patch_num), 2)
                        while abs(idx1 - idx2) < 2:
                            idx1, idx2 = random.sample(range(self.patch_num), 2)
                    else:
                        raise ValueError(f"permute_strategy must be one of the following: 'random', 'vicinity', 'farthest'")
                    
                    copy_index_data[idx1], copy_index_data[idx2] = copy_index_data[idx2], copy_index_data[idx1]

                single_sample_shuffled_idx_list.append(copy_index_data)
            total_shuffled_idx.append(np.stack(single_sample_shuffled_idx_list, axis=0))
        
        return torch.tensor(np.stack(total_shuffled_idx, axis=0))
                

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test with CPU first
    sample_data = np.random.randn(2, 2, 16, 10).astype(np.float32)
    device = torch.device("cpu")
    
    ppt = PPTShuffle(channel_num=2, original_time_len=160, patch_len=16, permute_freq=2, device=device)
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
        
        ppt = PPTShuffle(channel_num=2, original_time_len=160, patch_len=16, permute_freq=2, device=device)
        input_tensor = torch.randn(2, 2, 16, 10, device=device)
        output = ppt(input_tensor)
        
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Input device: {input_tensor.device}")
        print(f"Output device: {output.device}")