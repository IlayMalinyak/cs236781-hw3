import torch
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F
import copy
from .longformer import LongformerSelfAttention, LongformerConfig


# def _pad_mask(x, padding_mask, padding_value):
#     batch_size, num_heads, seq_len, seq_len = x.shape
#     mask = padding_mask.unsqueeze(-1) @ padding_mask.unsqueeze(-2)
#     mask = mask.view(batch_size, 1, seq_len, seq_len).expand_as(x).to(x.device)
#     return x.masked_fill(mask == 0, padding_value)

# def sliding_chunks_matmul_pv(prob: torch.Tensor, v: torch.Tensor, w: int):
#     '''Same as sliding_chunks_matmul_qk but for prob and value tensors. It is expecting the same output
#     format from sliding_chunks_matmul_qk'''
#     bsz, seqlen, num_heads, head_dim = v.size()
#     # print("in pv: bsz,seq_len,heads,dim - ",  bsz, seqlen, num_heads, head_dim)
#     assert seqlen % (w * 2) == 0
#     assert prob.size()[:3] == v.size()[:3]
#     assert prob.size(3) == 2 * w + 1
#     chunks_count = seqlen // w - 1
#     # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
#     chunk_prob = prob.transpose(1, 2).reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)

#     # group bsz and num_heads dimensions into one
#     v = v.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)

#     # pad seqlen with w at the beginning of the sequence and another w at the end
#     padded_v = F.pad(v, (0, 0, w, w), value=-1)

#     # chunk padded_v into chunks of size 3w and an overlap of size w
#     chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
#     chunk_v_stride = padded_v.stride()
#     chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
#     chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)

#     skewed_prob = _skew2(chunk_prob, padding_value=0)

#     context = torch.einsum('bcwd,bcdh->bcwh', (skewed_prob, chunk_v))
#     return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)
    

def _get_invalid_locations_mask(w, device):
    ''' Get values outside the window'''
    diagonals_list = []
    for j in range(-w, 1):
        diagonal_mask = torch.zeros(w, device='cpu', dtype=torch.uint8)
        diagonal_mask[:-j] = 1
        diagonals_list.append(diagonal_mask)
    mask = torch.stack(diagonals_list, dim=-1)
    mask = mask[None, :, None, :]
    ending_mask = mask.flip(dims=(1, 3)).bool().to(device)
    return mask.bool().to(device), ending_mask

def mask_invalid_locations(input_tensor: torch.Tensor, w: int):
    ''' Mask values outside the window'''
    beginning_mask, ending_mask = _get_invalid_locations_mask(w, input_tensor.device)
    seq_len = input_tensor.size(1)
    beginning_input = input_tensor[:, :w, :, :w+1]
    beginning_mask = beginning_mask[:, :seq_len].expand(beginning_input.size())
    beginning_input.masked_fill_(beginning_mask, -9e15)
    ending_input = input_tensor[:, -w:, :, -(w+1):]
    ending_mask = ending_mask[:, -seq_len:].expand(ending_input.size())
    ending_input.masked_fill_(ending_mask, -9e15)



def _skew(x, direction, padding_value):
    '''Convert diagonals into columns (or columns into diagonals depending on `direction`'''
    x_padded = nn.functional.pad(x, direction, value=padding_value)
    x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
    return x_padded


def get_main_diagonals_indices(b, n, w):
    ''' Indices of w main diaognals'''
    diag_indices = torch.arange(-w,w+1)
    row_indices = torch.arange(0,n*n, n+1)
    col_indices = row_indices.view(1,-1,1) + diag_indices
    col_indices = col_indices.repeat(b,1,1)
    return col_indices.flatten(1)[:, w:-w]

def populate_diags(x):
    ''' Populate diagonals from columns'''
    bzs, seq_len, w_ = x.size()
    w = (w_ - 1)//2
    x= x.flatten(1)[:, w:-w].float()
    res = torch.zeros(bzs,seq_len,seq_len, device=x.device).flatten(1)
    idx = get_main_diagonals_indices(bzs,seq_len,w).to(x.device)
    res= res.scatter_(1, idx, x).view(bzs,seq_len,seq_len)
    return res


def sliding_chunks_matmul_qk(q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float):
    '''Matrix multiplicatio of query x key tensors using with a sliding window attention pattern.
    '''
    bsz, num_heads, seqlen, head_dim = q.size()
    # print("shapes ", bsz, seqlen, num_heads, head_dim)
    # assert seqlen % (w * 2) == 0
    assert q.size() == k.size()

    chunks_count = seqlen // w - 1

    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
    q = q.reshape(bsz * num_heads, seqlen, head_dim)
    k = k.reshape(bsz * num_heads, seqlen, head_dim)

    chunk_q = q.unfold(-2, 2*w, w).transpose(-1,-2)
    chunk_k = k.unfold(-2, 2*w, w).transpose(-1,-2)
    # print("num_chunks", chunk_q.shape[1])

    # matrix multipication
    # bcxd: bsz*num_heads x chunks x 2w x head_dim
    # bcyd: bsz*num_heads x chunks x 2w x head_dim
    # bcxy: bsz*num_heads x chunks x 2w x 2w
    chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))  # multiply

    # convert diagonals into columns
    diagonal_chunk_attn = _skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)

    # allocate space for the overall attention matrix where the chunks are compined. The last dimension
    # has (w * 2 + 1) columns. The first (w) columns are the w lower triangles (attention from a word to
    # w previous words). The following column is attention score from each word to itself, then
    # followed by w columns for the upper triangle.
    diagonal_attn = torch.ones((bsz * num_heads, chunks_count + 1, w, w * 2 + 1), device=chunk_attn.device)*(-9e15)

    # copy parts from diagonal_chunk_attn into the compined matrix of attentions
    # - copying the main diagonal and the upper triangle
    diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
    diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
    # - copying the lower triangle
    diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, - (w + 1):-1, w + 1:]
    p = w > 1
    diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, p-w:]
    # separate bsz and num_heads dimensions again
    diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1).transpose(2, 1)
    
    mask_invalid_locations(diagonal_attn, w)
    diagonal_attn = diagonal_attn.transpose(1,2).view(bsz*num_heads, seqlen, 2 * w + 1)
    
    return diagonal_attn


def pad_qk_to_window_size(q,k,one_sided_window_size, padding_mask, paddin_value=0):
    ''' Pad q,k and padding mask to fit window size'''
    seq_len = q.shape[-2]
    w = int(2 * one_sided_window_size)
    padding_len = (w - seq_len % w) % w
    padding_l, padding_r = (padding_len//2, padding_len//2) if w > 2 else (0, 1)
    q = F.pad(q, (0,0,padding_l, padding_r), value=paddin_value)
    k = F.pad(k, (0,0,padding_l, padding_r), value=paddin_value)
    # v = F.pad(v, (0,0,padding_l, padding_r), value=paddin_value)
    if padding_mask is not None:
        padding_mask = F.pad(padding_mask, (padding_l, padding_r), value=0)
    return q,k,padding_mask




def sliding_window_attention(q, k, v, window_size, padding_mask=None):
    '''
    Computes the simple sliding window attention from 'Longformer: The Long-Document Transformer'.
    This implementation is meant for multihead attention on batched tensors. It should work for both single and multi-head attention.
    :param q - the query vectors. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param k - the key vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param v - the value vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param window_size - size of sliding window. Must be an even number.
    :param padding_mask - a mask that indicates padding with 0.  #[Batch, SeqLen]
    :return values - the output values. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :return attention - the attention weights. #[Batch, SeqLen, SeqLen] or [Batch, num_heads, SeqLen, SeqLen]
    '''
    assert window_size%2 == 0, "window size must be an even number"
    seq_len = q.shape[-2]
    embed_dim = q.shape[-1]
    batch_size = q.shape[0] 
    values, attention = None, None


    # TODO:
    #  Compute the sliding window attention.
    # NOTE: We will not test your implementation for efficiency, but you are required to follow these two rules:
    # 1) Implement the function without using for loops.
    # 2) DON'T compute all dot products and then remove the uneccessary comptutations 
    #    (both for tokens that aren't in the window, and for tokens that correspond to padding according to the 'padding mask').
    # Aside from these two rules, you are free to implement the function as you wish. 
    # ====== YOUR CODE: ======
    # Compute the sliding window attention
     # Compute left and right padding based on the window size

    w = window_size //2 

    no_head = False
    if len(q.shape) == 3:
        no_head = True
        q,k,v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
    num_heads = q.shape[1]
    q,k, padding_mask= pad_qk_to_window_size(q,k,w, padding_mask)
    new_seq_len = q.shape[-2]
    scores = sliding_chunks_matmul_qk(q,k,w,padding_value=-9e15).view(batch_size, num_heads, new_seq_len, 2 * w + 1) #[batch_size,num_heads,seq_len, 2w+1]

    if padding_mask is not None:
        padding_mask = torch.logical_not(padding_mask.unsqueeze(dim=1).unsqueeze(dim=-1))
        padding_mask = padding_mask.type_as(q).masked_fill(padding_mask, -9e15)
        ones = padding_mask.new_ones(size=padding_mask.size())  # tensor of ones
        d_mask = sliding_chunks_matmul_qk(ones, padding_mask, w, padding_value=-9e15).view(batch_size, 1, new_seq_len, 2 * w + 1)
        scores += d_mask
    attention =  torch.nn.functional.softmax(scores/math.sqrt(embed_dim), dim=-1).view(batch_size*num_heads, new_seq_len, 2 * w + 1) 
    attention = populate_diags(attention).view(batch_size, num_heads, new_seq_len, new_seq_len) # [batch_size, num_heads, seq_len, seq_len]
    if new_seq_len != seq_len:
        pad = new_seq_len - seq_len
        padding_l, padding_r = (pad//2, pad//2) if pad > 1 else (0, 1)
        attention = attention[:,:,padding_l:-padding_r,padding_l:-padding_r]
    values = torch.matmul(attention, v)
    if no_head:
        values = values.squeeze(1)
        attention = attention.squeeze(1)        
    # print('vals', values[0])
    return values, attention


def attention_vanilla(q, k, v, window_size, padding_mask=None):
    print("vanilla!", padding_mask)
    


    # Compute the attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32))
    n = scores.shape[-2]
    for i in range(3,n):
        scores[:,:, torch.arange(n-i), torch.arange(i,n)] = -9e15
        scores[:,:, torch.arange(i,n), torch.arange(n-i)] = -9e15
    
    # Apply padding mask if provided
    if padding_mask is not None:
        print("masking")
        scores = _pad_mask(scores, padding_mask, padding_value=-9e15)

   
        

    
    # Apply softmax to obtain attention weights
    attention_weights = torch.softmax(scores, dim=-1)
    # print("scores", scores[0][0])

    # Compute the output values
    values = torch.matmul(attention_weights, v)
    print()
    
    return values, attention_weights
    
    

class MultiHeadAttention(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_heads, window_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        # print("input dim", input_dim)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        
        # Stack all weight matrices 1...h together for efficiency
        # "bias=False" is optional, but for the projection we learned, there is no teoretical justification to use bias
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation of the paper if you would like....
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, padding_mask, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, 3*Dims]
        
        q, k, v = qkv.chunk(3, dim=-1) #[Batch, Head, SeqLen, Dims]
        # print(q.shape)
        
        # Determine value outputs
        # TODO:
        # call the sliding window attention function you implemented
        # ====== YOUR CODE: ======
        # print('kq_before attention', q[0,0,:10,0], k[0,0,:10,0])
        values, attention = sliding_window_attention(q,k,v,self.window_size, padding_mask)
        # ========================

        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim) #concatination of all heads
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o
        
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000): 
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model) 
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, window_size, dropout=0.1):
        '''
        :param embed_dim: the dimensionality of the input and output
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param num_heads: the number of heads in the multi-head attention
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, embed_dim, num_heads, window_size)
        self.feed_forward = PositionWiseFeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size
        
    def forward(self, x, padding_mask):
        '''
        :param x: the input to the layer of shape [Batch, SeqLen, Dims]
        :param padding_mask: the padding mask of shape [Batch, SeqLen]
        :return: the output of the layer of shape [Batch, SeqLen, Dims]
        '''
        # TODO:
        #   To implement the encoder layer, do the following:
        #   1) Apply attention to the input x, and then apply dropout.
        #   2) Add a residual connection from the original input and normalize.
        #   3) Apply a feed-forward layer to the output of step 2, and then apply dropout again.
        #   4) Add a second residual connection and normalize again.
        # ====== YOUR CODE: ======
        x_ = x
        x = self.dropout(self.self_attn(x, padding_mask))
        
        x = self.norm1(x + x_)

        x__ = x
        x = self.dropout(self.feed_forward(x))

        x = self.norm2(x + x__)
                                                               
        return x
    
    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length, window_size, dropout=0.1):
        '''
        :param vocab_size: the size of the vocabulary
        :param embed_dim: the dimensionality of the embeddings and the model
        :param num_heads: the number of heads in the multi-head attention
        :param num_layers: the number of layers in the encoder
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param max_seq_length: the maximum length of a sequence
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability

        '''
        super().__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, hidden_dim, num_heads, window_size, dropout) for _ in range(num_layers)])

        self.classification_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the logits  [Batch]
        '''
        output = None

        # TODO:
        #  Implement the forward pass of the encoder.
        #  1) Apply the embedding layer to the input.
        #  2) Apply positional encoding to the output of step 1.
        #  3) Apply a dropout layer to the output of the positional encoding.
        #  4) Apply the specified number of encoder layers.
        #  5) Apply the classification MLP to the output vector corresponding to the special token [CLS] 
        #     (always the first token) to receive the logits.
        # ====== YOUR CODE: ======
        embedded = self.encoder_embedding(sentence)
        encoded = self.dropout(self.positional_encoding(embedded))

        for layer in self.encoder_layers:
            encoded = layer(encoded, padding_mask)

        cls_token = encoded[:, 0, :]
        output = self.classification_mlp(cls_token).squeeze(-1)     
        return output
    
    def predict(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the binary predictions  [Batch]
        '''
        logits = self.forward(sentence, padding_mask)
        preds = torch.round(torch.sigmoid(logits))
        return preds

    
    