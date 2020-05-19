import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def src_mask(src):
    # srcは[batch_size, src_len]
    # パディング箇所が0、それ以外が1のTensorを生成
    pad = 1
    src_mask = (src != pad).unsqueeze(-2)  # [batch_size, 1, src_len]
    return src_mask

class Embedder(nn.Module):
    "idで示されている単語をベクトルに変換"
    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors, freeze=True)
        # freeze=Trueでbackprop時に更新されない

    def forward(self, x):
        x_vec = self.embeddings(x)

        return x_vec

class PositionalEncoder(nn.Module):
    '''入力された単語の位置を示すベクトルを付加'''
    
    def __init__(self, d_model=300, max_seq_len=256):
        super(PositionalEncoder, self).__init__()

        self.d_model = d_model # 単語ベクトル次元数

        # 単語の順番(pos)と埋め込みベクトルの次元の位置(i)によって一意に定まる値の表をpeとして作成
        pe = torch.zeros(max_seq_len, d_model)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1))/d_model)))

        # peの先頭にMini-batch次元となる次元を追加
        self.pe = pe.unsqueeze(0)

        # 勾配を計算しないようにする
        self.pe.requires_grad = False

    def forward(self, x):

        # 入力xとpositional encodingを足し算
        # xがpeよりも小さいので、大きくする
        ret = math.sqrt(self.d_model)*x + self.pe
        return ret

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate Scaled Dot-Product Attention
        output = scaled_dot_product_attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        # concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        concat = output.transpose(1, 2).reshape(bs, -1, self.d_model)
        output = self.out(concat)

        return output

def scaled_dot_product_attention(q, k, v, d_k, mask=None, dropout=None):
    # Scaled Dot-Product Attention
    # 多次元tensor同士のmatmulは後ろ2つの次元の行列同士の内積になる
    # そのため内積ができるようにkの後ろの次元を転置させる
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask==0, -1e9)
        
    normalized_attention_weight = F.softmax(scores, dim=-1) # normalized attention weight

    if dropout is not None:
        normalized_attention_weight = dropout(normalized_attention_weight)

    output = torch.matmul(normalized_attention_weight, v)

    # return output, normalized_attention_weight.to('cpu').detach().numpy()
    return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # Layer Normalization
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        # Multi-Head Attention
        self.attn = MultiHeadAttention(heads, d_model)
        # Position-wise Feed-Forward Network
        self.ff = PositionwiseFeedForward(d_model)
        # Dropout
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # Layer Normalization -> Multi-Head Attention -> Residual Dropout
        x_normalized_1 = self.norm_1(x)
        output = self.attn(x_normalized_1, x_normalized_1, x_normalized_1, mask)
        x2 = x + self.dropout_1(output)
        # Layer Normalization -> Position-wise Feed-Forward Network -> Residual Dropout
        x_normalized_2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normalized_2))
        
        return output

class ClassificationHead(nn.Module):
    '''Transformer_Blockの出力を使用し、最後にクラス分類させる'''

    def __init__(self, d_model=300, output_dim=2):
        super().__init__()

        # 全結合層
        self.linear = nn.Linear(d_model, output_dim)  # output_dimはポジ・ネガの2つ

        # 重み初期化処理
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        x0 = x[:, 0, :]  # 各ミニバッチの各文の先頭の単語の特徴量（300次元）を取り出す
        out = self.linear(x0)

        return out

class TransformerClassification(nn.Module):
    def __init__(self, text_embedding_vectors, d_model=300, max_seq_len=256, heads=2, N=2, output_dim=2):
        super().__init__()
        self.N = N
        self.embed = Embedder(text_embedding_vectors)
        self.pe = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.layers = nn.ModuleList([EncoderLayer(heads=heads, d_model=d_model) for i in range(N)])
        self.outlayer = ClassificationHead(d_model=d_model, output_dim=output_dim)
        # self.normalized_attention_weights = []

    def forward(self, x, mask):
        x = self.embed(x)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
            # self.normalized_attention_weights.append(weight)
        x = self.outlayer(x)

        return x