import torch.nn as nn
import torch
import torch.nn.functional as F
import math
config_path = r"C:\Users\mehdi\Desktop\Pythonfiles\Projects\Transformers\Alpha\config.json"

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        theta = 10000
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len)
        angles = torch.outer(positions, freqs)

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)
    
    def rotate_half(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        return torch.stack((-x2,x1),dim=-1).flatten(-2)
    
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
        
        #self.attn_dropout_p = dropout
        #self.resid_dropout = nn.Dropout(dropout)

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
        
        output = F.scaled_dot_product_attention(
            Q, K, V,
            #dropout_p=self.attn_dropout_p if self.training else 0.0, 
            is_causal= (Q_len == K_len), enable_gqa=True
        )

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_length, self.d_model)
        output = self.W_o(output)

        return output
    
class MLP(nn.Module):
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        hidden = int(2 * d_ff / 3)
        hidden = (hidden + 63) // 64 * 64   # 4096 -> 2752
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up   = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

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
        self.final_norm = nn.LayerNorm(d_model)
         
        self.output_proj = nn.Linear(d_model, vocab_size)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
        for n, p in self.named_parameters():
            if n.endswith("W_o.weight") or n.endswith("w_down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * len(self.decoder)))
                

    def forward(self, tokens):
        x = self.embedding(tokens)

        for block in self.decoder:
            x = block(x)

        logits = self.output_proj(self.final_norm(x))
        return logits