import sentencepiece as spm
import random
import torch

from collections import namedtuple
from typing import Dict, List

Batch = namedtuple("Batch", ["src", "tgt", "batch_size"])
Example = namedtuple("Example", ["src", "tgt"])

EOS_TOKEN = "<eos>"
BOS_TOKEN = "<bos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

class Vocab(object):
    """
        Integer to String / String to Integer helper
    """
    def __init__(self, words: List[str], specials: List[str]):
        self.itos = specials + words
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
    
    def __len__(self):
        return len(self.itos)


class Field(object):
    """
        Field class helper for source and target example
    """
    def __init__(self, bos : bool, eos: bool, pad: bool, unk: bool):
        self.bos_token = BOS_TOKEN if bos else None
        self.eos_token = EOS_TOKEN if eos else None
        self.unk_token = UNK_TOKEN if unk else None
        self.pad_token = PAD_TOKEN if pad else None

        self.vocab = None

    def load_vocab(self, words: List[str], specials: List[str]):
        self.vocab = Vocab(words, specials)

    def process(self, batch, device):
        max_len = max(len(x) for x in batch)

        padded, length = [], []

        for x in batch:
            bos = [self.bos_token] if self.bos_token else []
            eos = [self.eos_token] if self.eos_token else []
            pad = [self.pad_token] * (max_len - len(x))

            padded.append((bos + x + eos + pad))
            length.append(len(x) + len(bos) + len(eos))

        padded = torch.tensor([self.encode(ex) for ex in padded])
        return padded.long().to(device)

    def encode(self, tokens):
        ids = []
        for tok in tokens:
            if tok in self.vocab.stoi:
                ids.append(self.vocab.stoi[tok])
            else:
                ids.append(self.unk_id)
        return ids

    def decode(self, ids):
        tokens = []
        for tok in ids:
            tok = self.vocab.itos[tok]
            if tok == self.eos_token:
                break
            if tok == self.bos_token:
                continue
            tokens.append(tok)

        return " ".join(tokens).replace("##", "").replace("\u2581", "")
    
    @property
    def special(self):
        return [tok for tok in [self.unk_token, self.pad_token, self.bos_token, self.eos_token] if tok is not None]

    @property
    def pad_id(self):
        return self.vocab.stoi[self.pad_token]
    
    @property
    def eos_id(self):
        return self.vocab.stoi[self.eos_token]

    @property
    def bos_id(self):
        return self.vocab.stoi[self.bos_token]
    
    @property
    def unk_id(self):
        return self.vocab.stoi[self.unk_token]


class DataLoader(object):
    """
        Load raw string as tensor
    """
    def __init__(self, 
                src_path: str,
                tgt_path: str,
                batch_size: int,
                device: torch.device,
                train: bool,
                fields: Dict[str, Field], opt):
        self.batch_size = batch_size
        self.train = train
        self.device = device
        self.fields = fields
        self.sort_key = lambda ex: (len(ex.src), len(ex.tgt))

        examples = []
        for src_line, tgt_line in zip(read_file(src_path, "src", opt), read_file(tgt_path, "tgt", opt)):
            examples.append(Example(src_line, tgt_line))

        examples, self.seed = self.sort(examples)

        self.num_examples = len(examples)
        self.batches = list(batch(examples, self.batch_size))

    def __iter__(self):
        if self.train:
            random.shuffle(self.batches)

        for minibatch in self.batches:
            src = self.fields["src"].process([x.src for x in minibatch], self.device)
            tgt = self.fields["tgt"].process([x.tgt for x in minibatch], self.device)

            src = src[:, :500]
            yield Batch(src=src, tgt=tgt, batch_size=len(minibatch))

    def sort(self, examples):
        seed = sorted(range(len(examples)), key=lambda idx: self.sort_key(examples[idx]))
        return sorted(examples, key=self.sort_key), seed


def read_file(path, info, opt):
    sp_user = spm.SentencePieceProcessor()

    if info == 'src':
        sp_user.load(opt.vocab_source_model)
    else:
        sp_user.load(opt.vocab_target_model)
    with open(path, encoding="utf-8") as f:
        for line in f:
            l_info = sp_user.encode_as_pieces(line.strip())
            yield l_info

def batch(data, batch_size):
    data_len = len(data)

    for i in range(0, data_len, batch_size):
        yield data[i:i+batch_size]