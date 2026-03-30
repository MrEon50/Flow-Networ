import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from .utils import adjust_num_heads, safe_tensor_to_int

class AdaptiveFlowRouter(nn.Module):
    """
    Core innovation: Pattern-based flow generation
    Uses learned patterns instead of generating full matrices
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 num_flow_patterns: int = 8, base_sparsity: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_patterns = num_flow_patterns
        self.base_sparsity = base_sparsity
        
        # Library of learned flow patterns
        self.flow_patterns = nn.Parameter(
            torch.randn(num_flow_patterns, output_dim, input_dim) * 0.1
        )
        
        # Pattern selector - chooses which patterns to use
        self.pattern_selector = nn.Sequential(
            nn.Linear(input_dim, num_flow_patterns),
            nn.Softmax(dim=-1)
        )
        
        # Flow intensity modulator
        self.flow_intensity = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, _ = x.shape
        
        # Select flow patterns for each token
        pattern_weights = self.pattern_selector(x)  # (B, S, num_patterns)
        
        # Compose flow matrix as combination of patterns
        flow_matrix = torch.einsum('bsp,pij->bsij', pattern_weights, self.flow_patterns)
        
        # Modulate intensity
        intensity = self.flow_intensity(x).unsqueeze(-1)  # (B, S, 1, 1)
        flow_matrix = flow_matrix * intensity
        
        # Apply sparsity through top-k
        flow_matrix = self._apply_sparsity(flow_matrix)
        
        metrics = {
            'pattern_entropy': -(pattern_weights * torch.log(pattern_weights + 1e-8)).sum(-1).mean(),
            'flow_intensity': intensity.mean(),
            'pattern_diversity': torch.std(pattern_weights.mean(dim=(0,1)))
        }
        
        return flow_matrix, metrics
    
    def _apply_sparsity(self, flow_matrix: torch.Tensor) -> torch.Tensor:
        """Efficient batched sparsity application"""
        batch_size, seq_len, out_dim, in_dim = flow_matrix.shape

        # Early return for small matrices
        if out_dim * in_dim <= 64:
            return flow_matrix

        # Flatten for top-k
        flat_flow = flow_matrix.view(batch_size, seq_len, -1)

        # Adaptive sparsity based on matrix size
        base_k = max(1, safe_tensor_to_int(out_dim * in_dim * self.base_sparsity, default=1))
        # Limit k to prevent memory issues with very large matrices
        k = min(base_k, flat_flow.size(-1) // 2)

        # Batched top-k selection - more memory efficient
        _, topk_indices = torch.topk(flat_flow.abs(), k, dim=-1)

        # Create sparse mask efficiently using scatter
        sparse_mask = torch.zeros_like(flat_flow)
        # Use scatter_ for efficient batched assignment
        sparse_mask.scatter_(-1, topk_indices, 1.0)

        return (sparse_mask * flat_flow).view(batch_size, seq_len, out_dim, in_dim)


class ContextAwareFlowRouter(nn.Module):
    """
    Enhanced Flow Router with context awareness for long sequences
    Supports adaptive window adjustment and memory-efficient processing
    """

    def __init__(self, input_dim: int, output_dim: int, num_patterns: int = 16,
                 context_window: int = 1024, max_seq_len: int = 4096):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_patterns = num_patterns
        self.context_window = context_window
        self.max_seq_len = max_seq_len

        # Enhanced flow patterns with context awareness
        self.flow_patterns = nn.Parameter(
            torch.randn(num_patterns, output_dim, input_dim) * 0.1
        )

        # Context memory for long-term dependencies
        self.context_memory = nn.Parameter(torch.randn(context_window, output_dim))

        # Context-aware pattern selector
        # Use fixed dimension to avoid size mismatch
        context_dim = min(input_dim, output_dim)
        self.context_selector = nn.Sequential(
            nn.Linear(input_dim + context_dim, num_patterns * 2),
            nn.GELU(),
            nn.Linear(num_patterns * 2, num_patterns),
            nn.Softmax(dim=-1)
        )

        # Dynamic window adaptor
        self.window_adaptor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Flow intensity with context modulation
        self.flow_intensity = nn.Sequential(
            nn.Linear(input_dim + context_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, context_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, _ = x.shape

        # Extract or use provided context features
        if context_features is None:
            context_features = self._extract_context_features(x)

        # Combine input with context for enhanced pattern selection
        combined_input = torch.cat([x, context_features], dim=-1)
        pattern_weights = self.context_selector(combined_input)

        # Compose flow matrix with context-aware patterns
        flow_matrix = torch.einsum('bsp,pij->bsij', pattern_weights, self.flow_patterns)

        # Context-modulated intensity
        intensity = self.flow_intensity(combined_input).unsqueeze(-1)
        flow_matrix = flow_matrix * intensity

        # Adaptive window size
        window_size = self.window_adaptor(x.mean(dim=1)) * (self.max_seq_len - 256) + 256

        # Apply context-aware sparsity
        flow_matrix = self._apply_context_sparsity(flow_matrix, window_size)

        metrics = {
            'pattern_entropy': -(pattern_weights * torch.log(pattern_weights + 1e-8)).sum(-1).mean(),
            'flow_intensity': intensity.mean(),
            'context_diversity': torch.std(context_features.mean(dim=(0,1))),
            'adaptive_window_size': window_size.mean(),
            'pattern_diversity': torch.std(pattern_weights.mean(dim=(0,1)))
        }

        return flow_matrix, metrics

    def _extract_context_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract context features from input sequence"""
        batch_size, seq_len, input_dim = x.shape

        # Use sliding window approach for long sequences
        if seq_len > self.context_window:
            # Extract features from multiple windows
            window_features = []
            step_size = max(1, seq_len // 4)

            for i in range(0, seq_len - self.context_window + 1, step_size):
                window = x[:, i:i+self.context_window, :]
                window_feat = window.mean(dim=1)  # Simple aggregation
                window_features.append(window_feat)

            if window_features:
                context_features = torch.stack(window_features, dim=1).mean(dim=1)
            else:
                context_features = x.mean(dim=1)
        else:
            context_features = x.mean(dim=1)

        # Project to context memory dimension
        # Ensure compatibility between context_features and context_memory
        if context_features.size(-1) != self.context_memory.size(-1):
            # Add a projection layer if dimensions don't match
            projection = torch.nn.Linear(context_features.size(-1), self.context_memory.size(-1)).to(context_features.device)
            context_features = projection(context_features)

        # Use a simpler approach - just project to the required dimension
        context_dim = min(self.input_dim, self.output_dim)
        if context_features.size(-1) != context_dim:
            projection = torch.nn.Linear(context_features.size(-1), context_dim).to(context_features.device)
            context_features = projection(context_features)

        return context_features.unsqueeze(1).expand(-1, x.size(1), -1)

    def _apply_context_sparsity(self, flow_matrix: torch.Tensor, window_size: torch.Tensor) -> torch.Tensor:
        """Apply context-aware sparsity based on adaptive window size"""
        batch_size, seq_len, out_dim, in_dim = flow_matrix.shape

        # Adaptive sparsity based on window size
        base_sparsity = 0.1
        adaptive_sparsity = base_sparsity * (window_size / self.max_seq_len).mean()

        flat_flow = flow_matrix.view(batch_size, seq_len, -1)
        k = max(1, safe_tensor_to_int(out_dim * in_dim * adaptive_sparsity, default=1))

        # Vectorized top-k selection
        _, topk_indices = torch.topk(flat_flow.abs(), k, dim=-1)
        sparse_mask = torch.zeros_like(flat_flow)
        sparse_mask.scatter_(-1, topk_indices, 1.0)

        return (sparse_mask * flat_flow).view(batch_size, seq_len, out_dim, in_dim)


class FlowMemoryNetwork(nn.Module):
    """
    Flow Memory Network for long-term context and conversational memory
    Implements efficient memory access with flow-based attention
    """

    def __init__(self, d_model: int = 512, memory_size: int = 2048,
                 num_memory_heads: int = 8, memory_dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        self.num_memory_heads = num_memory_heads

        # Memory bank with learnable initialization - use buffer for safe updates
        self.register_buffer('memory_bank', torch.randn(memory_size, d_model) * 0.1)

        # Memory access mechanisms
        self.memory_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(memory_dropout),
            nn.Linear(d_model, d_model)
        )

        self.memory_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(memory_dropout),
            nn.Linear(d_model, d_model)
        )

        # Flow-based memory attention
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_memory_heads,
            dropout=memory_dropout,
            batch_first=True
        )

        # Memory update mechanism
        self.memory_update_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # Memory importance scoring
        self.importance_scorer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, update_memory: bool = True) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, _ = x.shape

        # Encode input for memory access
        encoded_input = self.memory_encoder(x)

        # Expand memory bank for batch processing
        memory_expanded = self.memory_bank.unsqueeze(0).expand(batch_size, -1, -1)

        # Flow-based memory attention
        memory_output, attention_weights = self.memory_attention(
            encoded_input, memory_expanded, memory_expanded
        )

        # Decode memory output
        decoded_output = self.memory_decoder(memory_output)

        # Combine with input
        combined_output = x + decoded_output
        combined_output = self.norm(combined_output)

        # Update memory if requested
        memory_metrics = {}
        if update_memory:
            memory_metrics = self._update_memory(x, attention_weights)

        # Calculate memory usage metrics
        memory_usage = torch.mean(attention_weights.sum(dim=-1))
        memory_diversity = torch.std(attention_weights.mean(dim=(0, 1)))

        metrics = {
            'memory_usage': memory_usage.item(),
            'memory_diversity': memory_diversity.item(),
            'memory_attention_entropy': -(attention_weights * torch.log(attention_weights + 1e-8)).sum(-1).mean().item(),
            **memory_metrics
        }

        return combined_output, metrics

    def _update_memory(self, x: torch.Tensor, attention_weights: torch.Tensor) -> Dict:
        """Update memory bank based on input importance"""
        # Calculate importance scores for input tokens
        importance_scores = self.importance_scorer(x)  # (batch, seq_len, 1)

        # Select most important tokens for memory update
        batch_size, seq_len, _ = x.shape
        top_k = min(seq_len, self.memory_size // 10)  # Update 10% of memory

        # Get top-k important tokens
        _, top_indices = torch.topk(importance_scores.squeeze(-1), top_k, dim=-1)

        # Extract top tokens
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, top_k)
        top_tokens = x[batch_indices, top_indices]  # (batch, top_k, d_model)

        # Update memory bank safely with autograd compatibility
        update_rate = 0.01
        memory_indices = torch.randint(0, self.memory_size, (batch_size, top_k), device=x.device)

        # Safe memory updates without breaking autograd
        with torch.no_grad():
            for b in range(batch_size):
                for k in range(top_k):
                    mem_idx = safe_tensor_to_int(memory_indices[b, k], default=0)
                    mem_idx = min(mem_idx, self.memory_size - 1)  # Bounds check
                    # Use exponential moving average for stable updates
                    self.memory_bank[mem_idx] = (
                        (1 - update_rate) * self.memory_bank[mem_idx] +
                        update_rate * top_tokens[b, k].detach().clone()
                    )

        return {
            'memory_updates': top_k,
            'avg_importance': importance_scores.mean().item(),
            'memory_update_rate': update_rate
        }


class EnhancedFlowLayer(nn.Module):
    """
    Pure Flow Layer with Context Awareness and RoPE - ZERO Attention bottlenecks
    Optimized for infinite sequences and extreme speed.
    """

    def __init__(self, input_dim: int, output_dim: int, num_patterns: int = 16,
                 num_heads: int = 8, dropout: float = 0.1, use_memory: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_memory = use_memory
        
        # We absorb num_heads argument purely for backward compatibility
        # No MultiheadAttention is actually used. Flow handles everything in O(N).

        # Enhanced flow router with context awareness
        self.flow_router = ContextAwareFlowRouter(input_dim, output_dim, num_patterns)

        # Cross-attention replaced with a Linear Memory Mixer
        if use_memory:
            self.memory_mixer = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim)
            )
            # Memory bank for long-term context 
            self.register_buffer('memory_bank', torch.randn(512, output_dim) * 0.1)

        # Feed-forward network (GLU variant for better performance)
        self.ffn_up = nn.Linear(output_dim, output_dim * 4 * 2) # *2 for SwiGLU gating
        self.ffn_down = nn.Linear(output_dim * 4, output_dim)
        self.ffn_dropout = nn.Dropout(dropout)

        # Layer normalization (RMSNorm is faster, using LayerNorm for compatibility)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def apply_rope(self, x: torch.Tensor, base: int = 10000) -> torch.Tensor:
        """Rotary Position Embeddings applied directly to flow vectors"""
        B, S, D = x.shape
        device = x.device
        
        # RoPE needs even dimensions
        if D % 2 != 0:
            return x
            
        positions = torch.arange(0, S, device=device).float().unsqueeze(1)
        dim_indices = torch.arange(0, D, 2, device=device).float()
        inv_freq = 1.0 / (base ** (dim_indices / D))
        
        sinusoid_inp = torch.einsum('i,j->ij', positions.squeeze(1), inv_freq)
        sin_val = torch.sin(sinusoid_inp)
        cos_val = torch.cos(sinusoid_inp)
        
        sin_val = torch.repeat_interleave(sin_val, 2, dim=-1).unsqueeze(0)
        cos_val = torch.repeat_interleave(cos_val, 2, dim=-1).unsqueeze(0)
        
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = -x[..., 1::2]
        x_rotated[..., 1::2] = x[..., 0::2]
        
        return x * cos_val + x_rotated * sin_val

    def forward(self, x: torch.Tensor, memory_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        # Input normalization and RoPE
        x_norm = self.norm1(x)
        x_rope = self.apply_rope(x_norm)

        # Flow transformation with Rotary Awareness
        flow_matrix, flow_metrics = self.flow_router(x_rope)
        flow_output = torch.einsum('bsij,bsj->bsi', flow_matrix, x_norm)
        
        # Residual
        combined_output = x + flow_output if self.input_dim == self.output_dim else flow_output
        combined_output = self.norm2(combined_output)

        # Memory integration (Linear O(N) instead of O(N^2))
        if self.use_memory and hasattr(self, 'memory_mixer'):
            memory_input = combined_output
            
            # Global average pool over memory bank or context
            if memory_context is not None:
                mem_features = memory_context.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1)
            else:
                mem_features = self.memory_bank.mean(dim=0).unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1)
                
            mixer_input = torch.cat([memory_input, mem_features], dim=-1)
            memory_output = self.memory_mixer(mixer_input)
            combined_output = combined_output + memory_output

        # SwiGLU Feed-forward
        ff_out = self.ffn_up(combined_output)
        ff_gate, ff_val = ff_out.chunk(2, dim=-1)
        ffn_output = self.ffn_down(F.silu(ff_gate) * ff_val)
        
        output = combined_output + self.ffn_dropout(ffn_output)

        enhanced_metrics = {
            **flow_metrics,
            'flow_pure_activation': torch.norm(flow_output, dim=-1).mean().item(),
            'layer_output_norm': torch.norm(output, dim=-1).mean().item()
        }

        return output, enhanced_metrics



class FlowLayer(nn.Module):
    """Memory-efficient Flow layer with adaptive processing"""
    
    def __init__(self, input_dim: int, output_dim: int, num_patterns: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Flow router
        self.flow_router = AdaptiveFlowRouter(input_dim, output_dim, num_patterns)
        
        # Bias and normalization
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Generate flow matrix
        flow_matrix, flow_metrics = self.flow_router(x)
        
        # Apply flow transformation
        output = torch.einsum('bsij,bsj->bsi', flow_matrix, x)
        
        # Bias and normalization
        output = output + self.bias
        output = self.layer_norm(output)
        output = F.gelu(output)
        
        return output, flow_metrics


