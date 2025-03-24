import pytest
import torch
import numpy as np
import sys

from .shuffler import PPT

def test_initialization():
    sample_data = np.random.randn(2, 1, 16, 10).astype(np.float32)  # (Batch, Channel, Embedding, Patch Number)
    ppt = PPT(sample_data, original_time_len=160, patch_len=16, permute_freq=2)
    
    assert ppt.original_time_len == 160
    assert ppt.patch_len == 16
    assert ppt.permute_freq == 2
    assert ppt.P == 10  # Patch 개수
    assert isinstance(ppt.permutation_index_tensor, torch.Tensor)

def test_forward_shape():
    sample_data = np.random.randn(2, 1, 16, 10).astype(np.float32)
    ppt = PPT(sample_data, original_time_len=160, patch_len=16, permute_freq=2, )
    
    input_tensor = torch.randn(2, 1, 16, 10)  # 같은 차원 유지
    input_tensor = input_tensor
    output = ppt(input_tensor)
    
    assert output.shape == input_tensor.shape, "Output shape must match input shape"

# def test_forward_permutation():
#     sample_data = np.random.randn(1, 1, 16, 10).astype(np.float32)
#     ppt = PPT(sample_data, original_time_len=160, patch_len=16, permute_freq=3)
    
#     input_tensor = torch.randn(1, 1, 16, 10)
#     output = ppt(input_tensor)
    
#     assert not torch.equal(input_tensor, output), "Output should be different from input due to permutation"

# def test_invalid_inputs():
#     with pytest.raises(ValueError):
#         PPT(np.random.randn(1, 1, 16, 10), original_time_len=150, patch_len=16, permute_freq=2)  # 원본 길이가 패치 길이로 나누어떨어지지 않음
    
#     with pytest.raises(ValueError):
#         PPT(np.random.randn(1, 1, 16, 10), original_time_len=160, patch_len=-1, permute_freq=2)  # 패치 길이 음수
    
#     with pytest.raises(ValueError):
#         PPT(np.random.randn(1, 1, 16, 10), original_time_len=160, patch_len=16, permute_freq=0)  # permute_freq 0

# def test_permutation_strategy():
#     sample_data = np.random.randn(1, 1, 16, 10).astype(np.float32)
    
#     with pytest.raises(ValueError):
#         PPT(sample_data, original_time_len=160, patch_len=16, permute_freq=2, permute_strategy="invalid")  # 잘못된 전략

if __name__ == "__main__":
    pytest.main()
