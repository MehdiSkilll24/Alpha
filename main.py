import torch.nn as nn
import torch

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

class MHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.d_k = d_model // num_heads
        self.rope = RoPE(self.d_k)

    def forward(self, x, kv_input=None, mask=None):
        batch_size = x.size(0)
        
        if kv_input is None:
            kv_input = x
  
        Q = self.Q(x)
        K = self.K(kv_input)
        V = self.V(kv_input)
        
        q_seq_length = x.shape[1]
        kv_seq_length = kv_input.shape[1]

        Q = Q.view(batch_size, q_seq_length, self.num_heads, self.d_k)
        K = K.view(batch_size, kv_seq_length, self.num_heads, self.d_k)
        V = V.view(batch_size, kv_seq_length, self.num_heads, self.d_k)

        Q = Q.transpose(1,2)                   
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        Q = self.rope(Q)
        K = self.rope(K)
        
        scores = Q @ K.transpose(-2, -1) / self.d_k**0.5

        if mask is not None:
            scores += mask
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = attn_weights @ V
        output = output.transpose(1,2).contiguous().view(batch_size, q_seq_length, self.d_model)
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
    

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mha = MHA(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        normed = self.norm1(x)
        attn_out = self.mha(normed)
        x = x + attn_out

        normed2 = self.norm2(x)
        mlp_out = self.mlp(normed2)
        x = x + mlp_out

        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, has_cross_attn=True):
        super().__init__()
        self.has_cross_attn = has_cross_attn
        self.self_attn = MHA(d_model, num_heads)
        if has_cross_attn:
            self.cross_attn = MHA(d_model, num_heads)
            self.norm3 = nn.LayerNorm(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x, encoder_output=None, mask=None):
        if mask is None:
            mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float('-inf'), device=x.device), diagonal=1)

        normed = self.norm1(x)
        attn_out = self.self_attn(normed, mask=mask)
        x = x + attn_out

        if self.has_cross_attn:
            normed2 = self.norm2(x)
            cross_out = self.cross_attn(normed2, encoder_output)
            x = x + cross_out
            normed3 = self.norm3(x)
        else:
            normed3 = self.norm2(x)

        mlp_out = self.mlp(normed3)
        x = x + mlp_out

        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, 
                num_encoder_layers, num_decoder_layers, model_type="encoder-decoder-transformer"):
        super().__init__()

        self.model_type = model_type
        self.d_model = d_model
        self.tgt_vocab_size = tgt_vocab_size
        
        if model_type == "encoder-decoder-transformer":
            self.src_embedding = nn.Embedding(src_vocab_size, d_model)
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
            self.encoder = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_encoder_layers)])
            self.decoder = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, has_cross_attn=True) for _ in range(num_decoder_layers)])

        elif model_type == "decoder-only":
            self.embedding = nn.Embedding(src_vocab_size, d_model)
            self.decoder = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, has_cross_attn=False) for _ in range(num_decoder_layers)])

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src_tokens, tgt_tokens):
        if self.model_type == "encoder-decoder-transformer":
            src = self.src_embedding(src_tokens)
            tgt = self.tgt_embedding(tgt_tokens)

            for block in self.encoder:
                src = block(src)
        
            encoder_output = src
        
            for block in self.decoder:
                tgt = block(tgt, encoder_output)
        elif self.model_type == "decoder-only":
            combined = torch.cat([src_tokens, tgt_tokens], dim=1)
            x = self.embedding(combined)

            src_len = src_tokens.shape[1]
            tgt_len = tgt_tokens.shape[1]
            total_len = src_len + tgt_len

            causal_mask = torch.triu(torch.full((total_len, total_len), float('-inf'), device=x.device), diagonal=1) 
            causal_mask[:src_len, :src_len] = 0

            for block in self.decoder:
                x = block(x, mask=causal_mask)

            tgt = x[:, src_len:, :]
        logits = self.output_proj(tgt)
        return logits