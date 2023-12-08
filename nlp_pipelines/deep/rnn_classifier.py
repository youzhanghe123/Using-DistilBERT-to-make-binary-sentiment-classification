import torch
import torch.nn as nn

from torchtext.vocab import GloVe
from transformers import BertModel, BertTokenizer

bert_pretrain_pth = "/Users/tracy/Desktop/bert-base-uncased" #"../pretrained/bert-base-uncased"

class RNNBinarySequenceClassifier(nn.Module):
    def __init__(self, 
                 vocab_size=33000,
                 embedding_size=256, 
                 hidden_size=256, 
                 output_size=1, 
                 num_layers=1, 
                 embedding_dropout=0.,
                 output_dropout=0.,
                 rnn_dropout=.0,
                 rnn_base_cell="vanilla", 
                 embedding_type="vanilla",
                 learnable=False,
                 bidirectional=False,
                 vocab=None,):
        super(RNNBinarySequenceClassifier, self).__init__()

        self.embedding_layer = WordEmbedding(vocab_size, 
                                       embedding_size, 
                                       learnable=learnable, 
                                       vocab=vocab,
                                       method=embedding_type)
        rnn_cell_map = {
            'gru': nn.GRU,
            'lstm': nn.LSTM,
            'vanilla': nn.RNN
        }
        RNN = rnn_cell_map[rnn_base_cell]
        self.rnn = RNN(embedding_size, hidden_size, num_layers, batch_first=True,
                       bidirectional=bidirectional, dropout=rnn_dropout if num_layers > 1 else 0.)
        self.rnn_type = rnn_base_cell
        
        self.is_lstm = rnn_base_cell=="lstm"

        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

        self.output_layer = nn.Linear(hidden_size, output_size) if not bidirectional else nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        """ 
          Args:
            ids: [bs, T] T is max_seq_len in the batch ; lengths: tuple of all lengths
            output: [bs, 1]
        """ 

        ids, lengths = x
        x_embedded = self.embedding_layer(ids)   # [bs, T, embedding_size]
        x_embedded = self.embedding_dropout(x_embedded)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(x_embedded, lengths, batch_first=True)
        _, rnn_hidden = self.rnn(packed_embedded) 

        if self.is_lstm:
            rnn_hidden = rnn_hidden[0] 
        # rnn_hidden: (n_layer*n_direction, bs, hidden_size)
        # We use the last rnn_hidden state for classification
        if self.rnn.bidirectional:
            rnn_hidden = torch.cat([rnn_hidden[-1], rnn_hidden[-2]], dim=-1)
        else:
            rnn_hidden = rnn_hidden[-1]
        
        output = self.output_layer(self.output_dropout(rnn_hidden))
        return output

    def predict(self, x):
        
        logit = nn.Sigmoid()(self(x))
        pred = (logit > .5).int()

        return pred
    
    def init_hidden(self,):
        hidden = (None, None) if isinstance(self.rnn, nn.LSTM) else None
        return hidden

class WordEmbedding(nn.Module):
    def __init__(self, 
                 vocab_size,
                 embedding_size,
                 learnable=False,
                 vocab=None,
                 method="") -> None:
        super().__init__()

        if method in ["vanilla"]:
            print("Random Initialize")
            self.embedding = nn.Embedding(vocab_size, embedding_size)
            learnable = True

        elif method in ["glove"]:
            assert embedding_size < 300, "GLoVE has maximum embedding size of 300"
            assert vocab is not None, "Please provide your own vocab dictionary"

            embedding_matrix = get_glove_embedding_matrix(vocab, dim=embedding_size)
            print("Initialize by GLoVE word embedding")
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        elif method in ["bert"]:
            print("Use BERT representation [fixed]")
            self.embedding = BertModel.from_pretrained(bert_pretrain_pth)
            self.embedding.eval() # Always in eval mode
            for param in self.embedding.parameters():
                param.requires_grad = False
            print("BERT layers are freezed")
        else:
            raise ValueError("Embedding Method must be in [vanilla, glove, bert]")
        
        if method != "bert" and (not learnable):
            print("Fix the Embedding Layer")
            self.embedding.weight.requires_grad = False

        self.embedding_type = method

    def forward(self, x):
        """x: [bs, T]"""
        
        if not isinstance(self.embedding, BertModel):
            return self.embedding(x)

        else:
            return get_bert_representation(self.embedding, x)
    

def get_glove_embedding_matrix(vocab, dim=128):
    glove = GloVe(name='6B', dim=300)

    embedding_matrix = torch.zeros(len(vocab), dim)
    for i, word in enumerate(vocab.values()):
        if word in glove.stoi:
            embedding_matrix[i] = glove[word][:dim]
        else:
            embedding_matrix[i] = torch.randn(dim)
    
    return embedding_matrix

@torch.no_grad()
def get_bert_representation(model = BertModel,
                        input_ids = torch.Tensor,):
        
    outputs = model(input_ids=input_ids)
    last_hidden_states = outputs.last_hidden_state

    return last_hidden_states