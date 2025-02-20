import torch
import torch.nn as nn
import torch.optim as optim
import math

###  75,842,816 parameters

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=200):
        super().__init__()
        
        # Create positional encoding matrix
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('encoding', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].to(x.device)

class TransformerDecoderCaption(nn.Module):
    def __init__(self, vocab_size=49408, d_model=512, num_heads=8, num_layers=6, mlp_ratio=4, dropout=0.1, max_len=50):
        super().__init__()
        self.d_model = d_model
        
        # All components use 512 dimensions
        self.embedding = nn.Embedding(vocab_size, 512)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerDecoderBlockCaption(d_model=512, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(512, vocab_size)

    def forward(self, tgt_seq, image_in):
        # CLIP embeddings are already 512-dimensional
        image_in = image_in.unsqueeze(1)  # [batch_size, 1, 512]
        
        x = self.embedding(tgt_seq) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, image_in)

        return self.fc_out(x)

class TransformerDecoderBlockCaption(nn.Module):
    def __init__(self, d_model=512, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(512)
        self.mask_attn = MultiHeadAttentionCaption(d_model=512, num_heads=num_heads, dropout=dropout, use_causal_mask=True)

        self.norm2 = nn.LayerNorm(512)
        self.self_attn = MultiHeadAttentionCaption(d_model=512, num_heads=num_heads, dropout=dropout)

        self.norm3 = nn.LayerNorm(512)
        self.mlp = nn.Sequential(
            nn.Linear(512, 512 * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512 * mlp_ratio, 512),
            nn.Dropout(dropout)
        )

    def forward(self, x, image_in):
        x = x + self.mask_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.self_attn(self.norm2(x), self.norm2(image_in), self.norm2(image_in))
        x = x + self.mlp(self.norm3(x))
        return x

class MultiHeadAttentionCaption(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1, use_causal_mask=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.use_causal_mask = use_causal_mask

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size, query_len, d_model = query.shape
        _, key_len, _ = key.shape  # Extract key sequence length

        # Project inputs to queries, keys, and values
        Q = self.W_Q(query).view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply causal mask (prevents attending to future tokens)
        if self.use_causal_mask:
            causal_mask = torch.tril(torch.ones(query_len, key_len, device=query.device))
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # Compute attention weights
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        A = torch.matmul(attention, V)

        # Reshape and apply final linear transformation
        A = A.transpose(1, 2).contiguous().view(batch_size, query_len, d_model)
        return self.W_out(A)



# # MODEL CLASSES DEFINITION

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import math


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_length=200):
#         super().__init__()
        
#         # Create positional encoding matrix
#         position = torch.arange(max_seq_length).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_seq_length, d_model)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('encoding', pe.unsqueeze(0))

#     def forward(self, x):
#         return x + self.encoding[:, :x.size(1)].to(x.device)


# class TransformerDecoderCaption(nn.Module):
#     def __init__(self, vocab_size=49408, d_model=768, num_heads=8, num_layers=6, mlp_ratio=4, dropout=0.1, max_len=50):
#         super().__init__()
#         self.d_model = d_model
        
#         # Make sure embedding outputs 768 dimensions
#         self.embedding = nn.Embedding(vocab_size, 768)
#         self.pos_encoding = PositionalEncoding(d_model=768, max_seq_length=max_len)
        
#         self.layers = nn.ModuleList([
#             TransformerDecoderBlockCaption(d_model=768, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout) 
#             for _ in range(num_layers)
#         ])
        
#         self.fc_out = nn.Linear(768, vocab_size)

#     def forward(self, tgt_seq, image_in):
#         image_in = image_in.unsqueeze(1)  # [batch_size, 1, 768]
        
#         # Debug print to check embedding output shape
#         x = self.embedding(tgt_seq)
#         print(f"Embedding output shape: {x.shape}")  # Should be [batch_size, seq_len, 768]
        
#         x = x * math.sqrt(self.d_model)
#         x = self.pos_encoding(x)

#         for layer in self.layers:
#             x = layer(x, image_in)

#         return self.fc_out(x)
    
    
# class TransformerDecoderBlockCaption(nn.Module):
#     def __init__(self, d_model=768, num_heads=8, mlp_ratio=4, dropout=0.1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(768)
#         self.mask_attn = MultiHeadAttentionCaption(d_model=768, num_heads=num_heads, dropout=dropout, use_causal_mask=True)

#         self.norm2 = nn.LayerNorm(768)
#         self.self_attn = MultiHeadAttentionCaption(d_model=768, num_heads=num_heads, dropout=dropout)

#         self.norm3 = nn.LayerNorm(768)
#         self.mlp = nn.Sequential(
#             nn.Linear(768, 768 * mlp_ratio),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(768 * mlp_ratio, 768),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x, image_in):
#         x = x + self.mask_attn(self.norm1(x), self.norm1(x), self.norm1(x))
#         x = x + self.self_attn(self.norm2(x), self.norm2(image_in), self.norm2(image_in))
#         x = x + self.mlp(self.norm3(x))
#         return x



# class MultiHeadAttentionCaption(nn.Module):
#     def __init__(self, d_model=768, num_heads=8, dropout=0.1, use_causal_mask=False):
#         super().__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
#         self.d_k = d_model // num_heads
#         self.num_heads = num_heads
#         self.use_causal_mask = use_causal_mask

#         self.W_Q = nn.Linear(d_model, d_model, bias=False)
#         self.W_K = nn.Linear(d_model, d_model, bias=False)
#         self.W_V = nn.Linear(d_model, d_model, bias=False)
#         self.W_out = nn.Linear(d_model, d_model, bias=False)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, query, key, value, mask=None):
#         batch_size, query_len, d_model = query.shape
#         _, key_len, _ = key.shape  # Extract key sequence length

#         # Project inputs to queries, keys, and values
#         Q = self.W_Q(query).view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)
#         K = self.W_K(key).view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)
#         V = self.W_V(value).view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)

#         # Compute scaled dot-product attention
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

#         # Apply mask if provided
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))

#         # Apply causal mask (prevents attending to future tokens)
#         if self.use_causal_mask:
#             causal_mask = torch.tril(torch.ones(query_len, key_len, device=query.device))
#             scores = scores.masked_fill(causal_mask == 0, float('-inf'))

#         # Compute attention weights
#         attention = torch.softmax(scores, dim=-1)
#         attention = self.dropout(attention)

#         # Apply attention to values
#         A = torch.matmul(attention, V)

#         # Reshape and apply final linear transformation
#         A = A.transpose(1, 2).contiguous().view(batch_size, query_len, d_model)
#         return self.W_out(A)
