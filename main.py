import torch.nn as nn, torch

VOCAB_SIZE = 128 #placeholder cst
d_model = 512
num_heads = 8
seq_len = 3
d_ff = 2048
class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.d_k = d_model // num_heads

    def forward(self, x, kv_input=None, mask=None):
        batch_size = x.size(0)
        # x.shape = (2, 3, 64)  64 dictates that there will be 64 Q, K and V PER TOKEN 
        if kv_input is None:
            kv_input = x
        Q = self.Q(x) # we do q_1 = XWq = 'W_1'*'e1' + ... + 'W_64'*'e64' W_i being the weight at i ... until we comp q_64, if no kv_input, it means we're doing encoding
        K = self.K(kv_input) # ,, ,, 
        V = self.V(kv_input) # ,, ,, 
        # We do this to ensure all Q,K and V's matrices have access to the input embeddings
        
        # Manual length extraction to avoid mismatch in size from encoder/decoder
        q_seq_length = x.shape[1]
        kv_seq_length = kv_input.shape[1]

        # We then split all this data through different heads so each one goes for a specific context window (varied contexts = better model overall
        Q = Q.view(batch_size, q_seq_length, num_heads, self.d_k)
        K = K.view(batch_size, kv_seq_length, num_heads, self.d_k)
        V = V.view(batch_size, kv_seq_length, num_heads, self.d_k)

        # So we get (batch, heads, seq_len, d_k) last 2 dims are the ones that get multiplied by Pytorch
        # The reason we put seq_len, d_k is because we gotta multiply them indep. of how many heads or batches, what matters is the consistent output per head per batch
        # Also, since we're computing relationships between tokens, there must be seq_len and d_k, so we can compare all tokens with all tokens for every head in every batch
        Q = Q.transpose(1,2)                   
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        
        scores = Q @ K.transpose(-2, -1) / self.d_k**0.5   #we swap d_k with seq_len so we get #seq_len * #seq_len matrix as a result

        if mask is not None:
            scores += mask
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = attn_weights @ V # contiguous vector of shape (batch, heads, seq, d_k)
        output = output.transpose(1,2).contiguous().view(batch_size, q_seq_length, d_model) # transposing makes it breaks contiguousy, so we add .contiguous() and then .view() to reshape it with concatenation
        output = self.W_o(output) # expressing the new shape correctly accross heads

        return output
    
class MLP(nn.Module):
    
    def __init__(self):
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
    def __init__(self):
        super().__init__()
        self.mha = MHA()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP()

    def forward(self, x):
        normed = self.norm1(x)
        attn_out = self.mha(normed)
        x = x + attn_out

        normed2 = self.norm2(x)
        mlp_out = self.mlp(normed2)
        x = x + mlp_out

        return x
    
class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MHA()
        self.cross_attn = MHA()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = MLP()

    def forward(self, x, encoder_output):
        causal_mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float('-inf')), diagonal=1)
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, mask = causal_mask)
        x = x + attn_out

        normed2 = self.norm2(x)
        cross_out = self.cross_attn(normed2, encoder_output)
        x = x + cross_out

        normed3 = self.norm3(x)
        mlp_out = self.mlp(normed3)
        x = x + mlp_out

        return x
    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, 512)
        self.encoder = nn.ModuleList([EncoderBlock() for _ in range(6)])
        self.decoder = nn.ModuleList([DecoderBlock() for _ in range(6)])
        self.output_proj = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, src_tokens, tgt_tokens):
        src = self.embedding(src_tokens)
        tgt = self.embedding(tgt_tokens)

        for block in self.encoder:
            src = block(src)
        
        encoder_output = src
        
        for block in self.decoder:
            tgt = block(tgt, encoder_output)

        logits = self.output_proj(tgt)
        return logits
