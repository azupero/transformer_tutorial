import torch
import torchtext
import MeCab
import string
import re
import neologdn
from torchtext.vocab import Vectors

# MeCab + NEologdによるtokenizer
def tokenizer_mecab(text):
    tagger = MeCab.Tagger('-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd') # -Owakatiで分かち書きのみ出力
    text = tagger.parse(text)
    text = text.strip().split()
    return text

# 前処理
def preprocessing_text(text):
    # 英語の小文字化(表記揺れの抑制)
    text = text.lower()
    # URLの除去(neologdnの後にやるとうまくいかないかも(URL直後に文章が続くとそれも除去される)))
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', '', text)
    # neologdnを用いて文字表現の正規化(全角・半角の統一と重ね表現の除去)
    text = neologdn.normalize(text)
    # 数字を全て0に置換(解析タスク上、数字を重要視しない場合は語彙数増加を抑制するために任意の数字に統一したり除去することもある)
    text = re.sub(r'[0-9 ０-９]+', '0', text)
    # 半角記号の除去
    text = re.sub(r'[!-/:-@[-`{-~]', "", text)
    # カンマ、ピリオド以外の記号をスペースに置換
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

    return text

def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_mecab(text)
    
    return ret

def get_dataloader(max_length=256, batch_size=8):
    PATH = '/content/drive/My Drive/Colab Notebooks/NLP/transformer_negaposi/data/'
    # Field
    TEXT = torchtext.data.Field(sequential=True, 
                                tokenize=tokenizer_with_preprocessing, 
                                use_vocab=True, 
                                lower=True, 
                                include_lengths=True, 
                                batch_first=True, 
                                fix_length=max_length, 
                                init_token="<cls>", 
                                eos_token="<eos>"
                                )
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False, dtype=torch.float32)
    # Dataset
    train_ds, val_ds = torchtext.data.TabularDataset.splits(path=PATH, 
                                                            train='train.tsv', 
                                                            validation='test.tsv', 
                                                            format='tsv', 
                                                            fields=[('Text', TEXT), ('Label', LABEL)]
                                                            )
    # embedding
    FASTTEXT = '/content/drive/My Drive/Colab Notebooks/NLP/nlp_tutorial/model.vec'
    fastText_vectors = Vectors(name=FASTTEXT)
    # vocab
    TEXT.build_vocab(train_ds, vectors=fastText_vectors, min_freq=5)
    # Iterator
    train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True)

    val_dl = torchtext.data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)

    return train_dl, val_dl, TEXT