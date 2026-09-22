import torch.nn as nn
import torch
import torch.nn.functional as F
import json

config_path = r"C:\Users\mehdi\Desktop\Pythonfiles\Projects\Transformers\Alpha\config.json"

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()

        theta = 10000

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

        positions = torch.arange(max_seq_len)

        angles = torch.outer(positions, freqs)

        self.register_buffer(
            "cos", 
            angles.cos()
        )

        self.register_buffer(
            "sin", 
            angles.sin()
        )
    
    def rotate_half(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        return torch.stack(
            (-x2,x1),
            dim=-1
        ).flatten(-2)
    
    def forward(self, x):
        
        seq_len = x.shape[-2]
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)

        return x*cos + self.rotate_half(x)*sin

class GQA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.num_kv_heads = num_kv_heads
        self.W_o = nn.Linear(d_model, d_model)
        
        self.d_k = d_model // num_heads
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.V = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.rope = RoPE(self.d_k)

    def forward(self, x, mask=None,  use_rope = True):
        batch_size = x.size(0)
  
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        
        seq_length = x.shape[1]

        Q = Q.view(batch_size, seq_length, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_length, self.num_kv_heads, self.d_k)
        V = V.view(batch_size, seq_length, self.num_kv_heads, self.d_k)

        Q = Q.transpose(1,2)                   
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        if use_rope:
            Q = self.rope(Q)
            K = self.rope(K)

        Q_len = Q.size(2)
        K_len = K.size(2)
        
        output = F.scaled_dot_product_attention(Q, K, V, is_causal= (Q_len == K_len), enable_gqa=True)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_length, self.d_model)
        output = self.W_o(output)

        return output
    
class MLP(nn.Module):
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_kv_heads):
        super().__init__()
        self.self_attn = GQA(d_model, num_heads, num_kv_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):

        normed = self.norm1(x)
        attn_out = self.self_attn(normed)
        x = x + attn_out

        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out

        return x
    
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, vocab_size, d_ff, num_kv_heads, num_decoder_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder = nn.ModuleList([DecoderBlock(d_model, num_heads,
            d_ff, num_kv_heads) for _ in range(num_decoder_layers)])
         
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        x = self.embedding(tokens)

        for block in self.decoder:
            x = block(x)

        logits = self.output_proj(x)
        return logits