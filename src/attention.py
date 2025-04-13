import torch
import triton
import triton.language as tl
from matmul import matmul_kernel
from softmax import softmax_kernel

def attention(q, k, v, mask=None, scale=None):
    """
    Compute attention scores using the provided query, key, and value matrices.
    
    Args:
        q: Query matrix of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key matrix of shape (batch_size, num_heads, seq_len, head_dim)
        v: Value matrix of shape (batch_size, num_heads, seq_len, head_dim)
        mask: Optional attention mask
        scale: Optional scaling factor (default: 1/sqrt(head_dim))
    
    Returns:
        Attention output of shape (batch_size, num_heads, seq_len, head_dim)
    """
    # Ensure inputs are on the same device
    device = q.device
    k = k.to(device)
    v = v.to(device)
    if mask is not None:
        mask = mask.to(device)
    
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Compute scaling factor if not provided
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    # Reshape for matrix multiplication
    q = q.view(-1, seq_len, head_dim)
    k = k.view(-1, seq_len, head_dim)
    v = v.view(-1, seq_len, head_dim)
    
    # Compute attention scores: Q @ K^T
    scores = torch.empty((batch_size * num_heads, seq_len, seq_len), 
                        device=device, dtype=q.dtype)
    
    # Call matmul kernel for Q @ K^T
    grid = lambda meta: (triton.cdiv(seq_len, meta['BLOCK_SIZE_M']), 
                        triton.cdiv(seq_len, meta['BLOCK_SIZE_N']))
    
    matmul_kernel[grid](q, k.transpose(-2, -1), scores,
                       seq_len, seq_len, head_dim,
                       head_dim, head_dim, seq_len,
                       seq_len, seq_len, seq_len,
                       ACTIVATION="none")  # No activation needed for attention scores
    
    # Scale the scores
    scores = scores * scale
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Apply softmax
    attn_weights = torch.empty_like(scores)
    grid = lambda meta: (triton.cdiv(seq_len, meta['BLOCK_SIZE']),)
    
    softmax_kernel[grid](attn_weights, scores,
                        seq_len, seq_len, seq_len,
                        seq_len, seq_len, seq_len)
    
    # Compute attention output: attn_weights @ V
    output = torch.empty((batch_size * num_heads, seq_len, head_dim),
                        device=device, dtype=q.dtype)
    
    # Call matmul kernel for attn_weights @ V
    grid = lambda meta: (triton.cdiv(seq_len, meta['BLOCK_SIZE_M']), 
                        triton.cdiv(head_dim, meta['BLOCK_SIZE_N']))
    
    matmul_kernel[grid](attn_weights, v, output,
                       seq_len, head_dim, seq_len,
                       seq_len, seq_len, head_dim,
                       head_dim, head_dim, head_dim,
                       ACTIVATION="none")  # No activation needed for final output
    
    # Reshape output back to original shape
    output = output.view(batch_size, num_heads, seq_len, head_dim)
    
    return output 