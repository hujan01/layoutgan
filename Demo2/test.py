 
from torch import nn
import torch
from torch.nn import functional as F
 
 
class TextNet(nn.Module):
    def __init__(self, vocab_size, seq_len,embedding_len, num_classes=2):
        super(TextNet, self).__init__()
        self.seq_len=seq_len
        self.vocab_size = vocab_size
        self.embedding_len = embedding_len
        self.word_embeddings = nn.Embedding(vocab_size, embedding_len)
 
    def forward(self, x):
        x = self.word_embeddings(x)
        return x
 
 
if __name__ == '__main__':
    model = TextNet(vocab_size=5000, seq_len=600,embedding_len=2)
    x=[[1,2,2,4]]
    input = torch.autograd.Variable(torch.LongTensor(x))
    o = model(input)
    print(o)
    print(o.size())
 
    x = [[1, 3, 2, 4]]
    input = torch.autograd.Variable(torch.LongTensor(x))
    o = model(input)
    print(o)
    print(o.size())
