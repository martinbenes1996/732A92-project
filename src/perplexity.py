
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
import sys

sys.path.append('src')
import config
import dataset

class NextWord(nn.Module):
    """"""
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """"""
        super().__init__()
        # embedding layer
        self._embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._embeddings.requires_grad = False
        # lstm layer
        self._lstm = nn.LSTM(embedding_dim, hidden_dim)
        # linear layer
        self._out = nn.Linear(hidden_dim, vocab_size)
        self._m = nn.Sigmoid()
    def forward(self, word, state):
        """"""
        # embedding layer
        embeds = self._embeddings(word)
        # lstm layer
        lstm_out,state = self._lstm(embeds.view(1,1,-1), state)
        # linear layer
        vector = self._out(lstm_out)\
            .view(1,-1)
        # activation of output
        scores = self._m(vector)
        return scores,state
    def get_word(self, vector):
        """"""
        word = torch.argmax(vector)
        return word

    @classmethod
    def train(cls, *args, **kw):
        """"""
        # load data
        x_tr = dataset._read_train_words(*args, **kw)\
            .reset_index(drop = True)\
            .text
        # construct vocabulary
        vocab = {word for sentence in x_tr for word in sentence}
        vocab_size = len(vocab)
        # words to vocabulary
        map_vocab = {word:i for i,word in enumerate(vocab)}
        x_tr = x_tr\
            .apply(lambda sentence: list(map_vocab[word] for word in sentence))

        # model
        embedding_dim = 300
        model = cls(embedding_dim, 50, vocab_size)\
            .to(config.device)
        print(model)
        # optimizer and loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # train
        num_epochs = 50
        state = None
        for epoch in range(num_epochs):
            for j,sentence in enumerate(x_tr):
                # detach
                if state is not None:
                    state = [s.detach() for s in state]
                # forward propagation
                score = torch.FloatTensor(len(sentence)-1,vocab_size)\
                    .to(config.device)
                for i in range(len(sentence)-1):
                    prev_word = torch.LongTensor([sentence[i]])\
                        .to(config.device)
                    score[i,:],state = model(prev_word,state)
                # loss
                label = torch.LongTensor(sentence[1:])\
                    .to(config.device)
                loss = criterion(score, label)
                # detach
                score.detach()
                label.detach()
                # backward propagation
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 100)
                optimizer.step()
                # log
                if j % 100 == 0:
                    print("Sentence", j, "/", x_tr.shape[0])
            # log
            print("Epoch", epoch, "/", num_epochs, "- loss %.5f" % (loss,))
        # return
        return model,state
    
if __name__ == '__main__':
    NextWord.train()
    