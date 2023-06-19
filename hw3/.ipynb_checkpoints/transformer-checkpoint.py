import torch
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F

from .longformer import LongformerSelfAttention, LongformerConfig


# def _mask(x, padding_mask, mask_value) -> Tensor:
#     """
#     Masks the input tensor according to 'padding_mask'.
#     :param x - Tensor of shape (batch_size, head_size, seq_size, seq_size)
#     param padding_mask - A padding mask of shape (batch_size, seq_size)
#     :return - A masked tensor of shape (batch_size, head_size, seq_size, seq_size)
#     """
#     if padding_mask is None:
#         return x
#     batch_size, head_size, seq_size,d = x.shape
#     mask = padding_mask.unsqueeze(-1) @ padding_mask.unsqueeze(-2)
#     print
#     mask = mask.view(batch_size, 1, seq_size, d).expand_as(x)
#     return x.masked_fill(mask == 0, mask_value)



# def sliding_chunks_matmul_pv(prob: torch.Tensor, v: torch.Tensor, w: int):
#     '''Same as sliding_chunks_matmul_qk but for prob and value tensors. It is expecting the same output
#     format from sliding_chunks_matmul_qk'''
#     bsz, num_heads, seqlen, head_dim = v.size()
#     # assert seqlen % (w * 2) == 0
#     # assert prob.size()[] == v.size()[:3]
#     # assert prob.size(3) == 2 * w + 1
#     chunks_count = seqlen // w - 1
#     # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
#     # chunk_prob = prob.transpose(1, 2).reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)
#     chunk_prob = prob.reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)
#     # print("chunk", chunk_prob)

#     # group bsz and num_heads dimensions into one
#     v = v.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)

#     # pad seqlen with w at the beginning of the sequence and another w at the end
#     padded_v = nn.functional.pad(v, (0, 0, w, w), value=-1)

#     # chunk padded_v into chunks of size 3w and an overlap of size w
#     chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
#     chunk_v_stride = padded_v.stride()
#     chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
#     chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)

#     skewed_prob = _skew2(chunk_prob, padding_value=0)
#     # print('skewed', skewed_prob.shape, skewed_prob)

#     context = torch.einsum('bcwd,bcdh->bcwh', (skewed_prob, chunk_v))
#     # print("diag" , skewed_prob.shape, chunk_prob.shape)
#     diag_prob = skewed_prob.reshape(bsz*num_heads, seqlen, 2 * w + 2)
#     # diag_prob = diag_prob.reshape(bsz*num_heads, seqlen, seqlen)
#     print("diag", diag_prob)
#     full_prob = populate_diags(diag_prob, bsz*num_heads, seqlen, w)

#     return context.view(bsz, num_heads, seqlen, head_dim), full_prob.view(bsz, num_heads, seqlen, seqlen)

# def _skew2(x, padding_value):
#     '''shift every row 1 step to right converting columns into diagonals'''
#     # X = B x C x M x L
#     B, C, M, L = x.size()
#     x = nn.functional.pad(x, (0, M + 1), value=padding_value)  # B x C x M x (L+M+1)
#     x = x.view(B, C, -1)  # B x C x ML+MM+M
#     x = x[:, :, :-M]  # B x C x ML+MM
#     x = x.view(B, C, M, M + L)  # B x C, M x L+M
#     x = x[:, :, :, :-1]
#     return x


# def _chunk(x, w):
#     '''convert into overlapping chunkings. Chunk size = 2w, overlap size = w'''

#     # non-overlapping chunks of size = 2w
#     x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

#     # use `as_strided` to make the chunks overlap with an overlap size = w
#     chunk_size = list(x.size())
#     chunk_size[1] = chunk_size[1] * 2 - 1

#     chunk_stride = list(x.stride())
#     chunk_stride[1] = chunk_stride[1] // 2
#     return x.as_strided(size=chunk_size, stride=chunk_stride)


def _unfold(x: Tensor, window_size: int, pad_value: float) -> Tensor:
    """
    Folds input tensor 'window_size' times on the 'seq_size' dimension.
    Makes sure the number of folds is the same as seq_size, by padding accordingly.
    :param x - Tensor of shape (batch_size, head_size, seq_size, hidden_size)
    :return - an unfolded tensor on the seq_size dimension. Tensor of
    shape (batch_size, head_size, seq_size, window_size, hidden_size)
    """
    x = F.pad(x, (0, 0, window_size // 2, window_size // 2), value=pad_value)
    return x.unfold(2, window_size, 1).swapaxes(-1, -2)


def _sk(x: Tensor, pad_value: float) -> Tensor:
    """
    Skews input tensor and creates a matching banded square matrix.
    :param x - Tensor of shape (batch_size, head_size, seq_size, window_size)
    :return - an skewed tensor on the seq_size rows. Tensor of
    shape (batch_size, head_size, seq_size, seq_size)
    """
    batch_size, head_size, seq_size, window_size = x.size()
    mask = torch.ones(seq_size, seq_size, device=x.device, dtype=torch.bool)
    mask = mask.triu(-window_size // 2 + 1).tril(window_size // 2)
    mask = mask.view(1, 1, seq_size, seq_size).expand(batch_size, head_size, seq_size, seq_size)
    skewed = torch.full_like(mask, fill_value=pad_value, dtype=x.dtype, device=x.device)
    skewed[mask] = x[x.isfinite()]
    return skewed


def _mask(x, padding_mask, mask_value) -> Tensor:
    """
    Masks the input tensor according to 'padding_mask'.
    :param x - Tensor of shape (batch_size, head_size, seq_size, seq_size)
    param padding_mask - A padding mask of shape (batch_size, seq_size)
    :return - A masked tensor of shape (batch_size, head_size, seq_size, seq_size)
    """
    if padding_mask is None:
        return x
    batch_size, head_size, seq_size, _ = x.shape
    mask = padding_mask.unsqueeze(-1) @ padding_mask.unsqueeze(-2)
    mask = mask.view(batch_size, 1, seq_size, seq_size).expand_as(x)
    return x.masked_fill(mask == 0, mask_value)


def sliding_window_attention2(q: Tensor, k: Tensor, v: Tensor, window_size: int, padding_mask: Tensor = None) -> tuple:
    """
    Computes the simple sliding window attention from 'Longformer: The Long-Document Transformer'.
    This implementation is meant for multihead attention on batched tensors. It should work for both single and multi-head attention.
    :param q - the query vectors. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param k - the key vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param v - the value vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param window_size - size of sliding window. Must be an even number.
    :param padding_mask - a mask that indicates padding with 0.  #[Batch, SeqLen]
    :return values - the output values. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :return attention - the attention weights. #[Batch, SeqLen, SeqLen] or [Batch, num_heads, SeqLen, SeqLen]
    """
    assert window_size % 2 == 0, "window size must be an even number"

    window_size += 1  # The actual window size

    v_shape = v.size()

    batch_size = q.shape[0]
    seq_size = q.shape[-2]
    hidden_size = q.shape[-1]

    q = q.view(batch_size, -1, seq_size, hidden_size)  # (batch_size, head_size, seq_size, hidden_size)
    k = k.view(batch_size, -1, seq_size, hidden_size)  # (batch_size, head_size, seq_size, hidden_size)
    v = v.view(batch_size, -1, seq_size, hidden_size)  # (batch_size, head_size, seq_size, hidden_size)

    head_size = q.shape[1]

    k = _unfold(k, window_size, math.nan)  # (batch_size, head_size, seq_size, window_size, hidden_size)

    q = q.view(batch_size, head_size, seq_size, 1, hidden_size)

    b = (q @ k.transpose(-1, -2)).squeeze(3) / math.sqrt(hidden_size)

    b = _sk(b, -9e15)  # (batch_size, head_size, seq_size, seq_size)

    b = _mask(b, padding_mask, -9e15)  # (batch_size, head_size, seq_size, seq_size)

    a = torch.softmax(b, dim=-1)  # (batch_size, head_size, seq_size, seq_size)

    values = a @ v  # (batch_size, head_size, seq_size, hidden_size)

    a = a.view(*v_shape[:-2], seq_size, seq_size)
    values = values.view(*v_shape[:-2], seq_size, hidden_size)
    print("attn2", a[0][0][:,:9])

    return values, a

def _skew(x, direction, padding_value):
    '''Convert diagonals into columns (or columns into diagonals depending on `direction`'''
    x_padded = nn.functional.pad(x, direction, value=padding_value)
    x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
    return x_padded


def get_main_diagonals_indices(b, n, w):
    diag_indices = torch.arange(-w,w+1)
    row_indices = torch.arange(0,n*n, n+1)
    col_indices = row_indices.view(1,-1,1) + diag_indices
    col_indices = col_indices.repeat(b,1,1)
    return col_indices.flatten(1)[:, w:-w]

def populate_diags(a,bzs,seq_len,w):
    a= a.flatten(1)[:, w:-w].float()
    res = torch.zeros(bzs,seq_len,seq_len).flatten(1)
    idx = get_main_diagonals_indices(bzs,seq_len,w)
    res= res.scatter_(1, idx, a).view(bzs,seq_len,seq_len)
    return res


def sliding_chunks_matmul_qk(q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float):
    '''Matrix multiplicatio of query x key tensors using with a sliding window attention pattern.
    This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
    with an overlap of size w'''
    bsz, num_heads, seqlen, head_dim = q.size()
    print("shapes ", bsz, seqlen, num_heads, head_dim)
    # assert seqlen % (w * 2) == 0
    assert q.size() == k.size()

    chunks_count = seqlen // w - 1

    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
    q = q.reshape(bsz * num_heads, seqlen, head_dim)
    k = k.reshape(bsz * num_heads, seqlen, head_dim)

    chunk_q = q.unfold(-2, 2*w, w).transpose(-1,-2)
    chunk_k = k.unfold(-2, 2*w, w).transpose(-1,-2)

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
    diagonal_attn = torch.ones((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))*(-math.inf)

    # copy parts from diagonal_chunk_attn into the compined matrix of attentions
    # - copying the main diagonal and the upper triangle
    diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
    diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
    # - copying the lower triangle
    diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, - (w + 1):-1, w + 1:]

    p = w > 1
    diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, p-w:]
    print("diag" , diagonal_attn.shape, "\n", diagonal_attn[0])

    diagonal_attn = diagonal_attn.view(bsz*num_heads, seqlen, 2 * w + 1)

    
    return diagonal_attn

def _pad_mask(padding_mask, num_heads, w, padding_value):
    batch_size, seq_len= padding_mask.size()
    padding_mask = torch.logical_not(padding_mask)
        
    padding_mask = padding_mask.unsqueeze(dim=1).unsqueeze(dim=-1)
    # cast to float/half then replace 1's with -inf
    float_mask = padding_mask.masked_fill(padding_mask, 1)
    float_mask = float_mask.repeat(num_heads, 1, 1, 1)
    ones = float_mask.new_ones(size=float_mask.size()) 
    d_mask = sliding_chunks_matmul_qk(ones, float_mask, w, padding_value=1)
    diag_mask = d_mask.reshape(batch_size*num_heads, seq_len, 2 * w + 1)
    diag_mask = torch.where(diag_mask==0, 0.0, padding_value)
    print(diag_mask[0])
    return diag_mask  


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
    print('initial' , q.shape)
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
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
    num_heads = q.shape[1]
    scores = sliding_chunks_matmul_qk(q,k,w,padding_value=0)

    if padding_mask is not None:
        d_mask = _pad_mask(padding_mask, num_heads, w, padding_value=-1e30)
        # scores = _mask(scores.view(batch_size, num_heads, seq_len,-1), padding_mask, -1e15)
        scores += d_mask
    scores =  torch.nn.functional.softmax(scores/math.sqrt(embed_dim), dim=-1)
    attention = populate_diags(scores, batch_size*num_heads, seq_len, w).view(batch_size, num_heads, seq_len, seq_len)

    print(attention.shape)
    values = torch.matmul(attention, v)

    # scores = torch.nan_to_num(scores, 0)
    print("attn", attention[0][0].sum(1))
    if no_head:
        values = values.squeeze(1)
        attention = attention.squeeze(1)
    return values, attention


def attention_vanilla(q, k, v, window_size, padding_mask=None):
    print("vanilla!", padding_mask)
    


    # Compute the attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32))
    n = scores.shape[-2]
    for i in range(3,n):
        scores[:,:, torch.arange(n-i), torch.arange(i,n)] = -math.inf
        scores[:,:, torch.arange(i,n), torch.arange(n-i)] = -math.inf
    
    # Apply padding mask if provided
    if padding_mask is not None:
        print("masking")
        scores =_mask(scores, padding_mask, -math.inf)
        # scores[:,:,5:,:] = -math.inf
        # scores[:,:,:,5:] = -math.inf

   
        

    
    # Apply softmax to obtain attention weights
    attention_weights = torch.softmax(scores, dim=-1)
    print("scores", scores[0][0])

    # Compute the output values
    values = torch.matmul(attention_weights, v)
    print()
    
    return values, attention_weights
    
    

class MultiHeadAttention(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_heads, window_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
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
        print(q.shape)
        
        # Determine value outputs
        # TODO:
        # call the sliding window attention function you implemented
        # ====== YOUR CODE: ======
        values, attention = sliding_window_attention(q,k,v,self.window_size, padding_mask)
        # print("hi", padding_mask)
        # values, attention = attention_vanilla(q,k,v, self.window_size, padding_mask)
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
        self.cfg = LongformerConfig(attention_window=[window_size//2], attention_dilation=[1], hidden_size=hidden_dim, num_attention_heads=num_heads)
        self.longformer_atn = LongformerSelfAttention(self.cfg, 0)
        
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
        # to_pad = self.window_size - (x.shape[1] % self.window_size)
        # x = F.pad(x, (0,0,0,to_pad,0,0))
        # if padding_mask is not None:
        #     padding_mask = F.pad(padding_mask, (0,to_pad,0,0))
        # # print(x.shape, padding_mask.shape)
        # x_attn = self.dropout(self.longformer_atn(x, attention_mask=padding_mask)[0])
        
        x_attn = self.dropout(self.self_attn(x, padding_mask))
        # print("attn - ", attn[0][0])
        x = self.norm1(x_attn+x)
        x_feed = self.dropout(self.feed_forward(x))
        x = self.norm2(x_feed+x)
                                                       
        # ========================
        
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
        super(Encoder, self).__init__()
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
        raise NotImplementedError()
        
        # ========================
        
        
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

    
    