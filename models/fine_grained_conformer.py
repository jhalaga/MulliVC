"""
Fine-grained Timbre Conformer for capturing fine timbre details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ConvolutionModule(nn.Module):
    """Convolution module for the Conformer."""
    
    def __init__(
        self,
        input_dim: int,
        kernel_size: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        
        # Pointwise convolution 1
        self.pointwise_conv1 = nn.Conv1d(input_dim, input_dim * 2, 1)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            input_dim * 2, 
            input_dim * 2, 
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=input_dim * 2
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(input_dim * 2)
        
        # Activation
        self.activation = nn.SiLU()
        
        # Pointwise convolution 2
        self.pointwise_conv2 = nn.Conv1d(input_dim * 2, input_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolution module.

        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            output: Tensor of shape (batch_size, seq_len, input_dim).
        """
        # Transpose for 1D convolution
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        # Pointwise conv 1
        x = self.pointwise_conv1(x)
        
        # Depthwise conv
        x = self.depthwise_conv(x)
        
        # Batch norm and activation
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Pointwise conv 2
        x = self.pointwise_conv2(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Transpose back
        x = x.transpose(1, 2)  # (batch_size, seq_len, input_dim)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head attention with positional encoding."""
    
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        assert input_dim % num_heads == 0, "input_dim doit être divisible par num_heads"
        
        # Linear projections
        self.q_linear = nn.Linear(input_dim, input_dim)
        self.k_linear = nn.Linear(input_dim, input_dim)
        self.v_linear = nn.Linear(input_dim, input_dim)
        self.out_linear = nn.Linear(input_dim, input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(input_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the attention block.

        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim).
            mask: Optional attention mask.

        Returns:
            output: Tensor of shape (batch_size, seq_len, input_dim).
            attention_weights: Attention weights.
        """
        batch_size, seq_len, _ = x.shape
        
        # Position encoding
        x = self.pos_encoding(x)
        
        # Projections Q, K, V
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply the mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        
        # Reshape and apply the final projection
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.input_dim
        )
        output = self.out_linear(attended)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model).

        Returns:
            x + pe: Tensor with positional encoding.
        """
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class ConformerBlock(nn.Module):
    """Complete Conformer block."""
    
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        ff_expansion_factor: int = 4
    ):
        super().__init__()
        
        self.input_dim = input_dim
        ff_dim = input_dim * ff_expansion_factor
        
        # Feed-forward 1
        self.ff1 = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Convolution module
        self.conv_module = ConvolutionModule(
            input_dim=input_dim,
            kernel_size=conv_kernel_size,
            dropout=dropout
        )
        
        # Feed-forward 2
        self.ff2 = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.layer_norm3 = nn.LayerNorm(input_dim)
        self.layer_norm4 = nn.LayerNorm(input_dim)
        
        # Dropout final
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Conformer block.

        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim).
            mask: Optional attention mask.

        Returns:
            output: Tensor of shape (batch_size, seq_len, input_dim).
            attention_weights: Attention weights.
        """
        # Feed-forward 1 with residual connection
        residual = x
        x = self.layer_norm1(x)
        x = self.ff1(x)
        x = self.dropout(x) + residual
        
        # Multi-head attention with residual connection
        residual = x
        x = self.layer_norm2(x)
        x, attention_weights = self.attention(x, mask)
        x = self.dropout(x) + residual
        
        # Convolution module with residual connection
        residual = x
        x = self.layer_norm3(x)
        x = self.conv_module(x)
        x = self.dropout(x) + residual
        
        # Feed-forward 2 with residual connection
        residual = x
        x = self.layer_norm4(x)
        x = self.ff2(x)
        x = self.dropout(x) + residual
        
        return x, attention_weights


class FineGrainedTimbreConformer(nn.Module):
    """
    Fine-grained Timbre Conformer for capturing fine timbre details.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        output_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.output_dim = output_dim or input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, input_dim)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                input_dim=input_dim,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )
        
        # Attention pooling to obtain a global representation
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=self.output_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(
        self, 
        timbre_features: torch.Tensor,
        content_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Fine-grained Timbre Conformer.

        Args:
            timbre_features: Tensor of shape (batch_size, seq_len, input_dim).
            content_features: Tensor of shape (batch_size, seq_len, input_dim).
            mask: Optional attention mask.

        Returns:
            fine_grained_timbre: Tensor of shape (batch_size, seq_len, output_dim).
            global_timbre: Tensor of shape (batch_size, output_dim).
            attention_weights: Attention weights from all blocks.
        """
        batch_size, seq_len, _ = timbre_features.shape
        
        # Fuse timbre and content features
        combined_features = timbre_features + content_features
        
        # Input projection
        x = self.input_projection(combined_features)
        
        # Apply Conformer blocks
        all_attention_weights = []
        for block in self.conformer_blocks:
            x, attention_weights = block(x, mask)
            all_attention_weights.append(attention_weights)
        
        # Output projection
        fine_grained_timbre = self.output_projection(x)
        
        # Attention pooling for a global representation
        # Use a global query
        global_query = torch.mean(fine_grained_timbre, dim=1, keepdim=True)
        global_timbre, _ = self.attention_pooling(
            global_query, fine_grained_timbre, fine_grained_timbre
        )
        global_timbre = global_timbre.squeeze(1)
        
        return fine_grained_timbre, global_timbre, all_attention_weights
    
    def extract_fine_grained_features(
        self, 
        timbre_features: torch.Tensor,
        content_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified method for extracting fine-grained features.

        Args:
            timbre_features: Tensor of shape (batch_size, seq_len, input_dim).
            content_features: Tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            fine_grained_features: Tensor of shape (batch_size, seq_len, output_dim).
        """
        fine_grained_timbre, _, _ = self.forward(timbre_features, content_features)
        return fine_grained_timbre
