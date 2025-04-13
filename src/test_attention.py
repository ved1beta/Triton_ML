import torch
import triton
from attention import attention

def test_attention():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 2
    num_heads = 4
    seq_len = 8
    head_dim = 16
    
    # Create random input tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    
    # Create a simple causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 0
    mask = mask.unsqueeze(0).unsqueeze(0).to(device)
    
    # Run attention
    output = attention(q, k, v, mask=mask)
    
    # Verify output shape
    assert output.shape == (batch_size, num_heads, seq_len, head_dim), \
        f"Expected shape {(batch_size, num_heads, seq_len, head_dim)}, got {output.shape}"
    
    # Verify no NaN values
    assert not torch.isnan(output).any(), "Output contains NaN values"
    
    # Verify output values are reasonable
    assert output.abs().max() < 100, "Output values are too large"
    
    print("All tests passed!")
    print("Output shape:", output.shape)
    print("Output stats:")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")

if __name__ == "__main__":
    test_attention() 