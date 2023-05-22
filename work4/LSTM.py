import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import jieba
from tqdm import tqdm

class Dictionary(object):
    #初始化，建立两个映射
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

class Corpus(object):
    #将文本向量化
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        # step 1
        with open(path, 'r', encoding="utf-8") as f:
            tokens = 0
            for line in f.readlines():
                words = jieba.lcut(line) + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # step 2
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                words = jieba.lcut(line) + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        # step 3
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        ids = ids.view(batch_size, -1)
        return ids

class LSTMmodel(nn.Module):
    #调用nn模块，训练网络
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMmodel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)#初始化词嵌入层，输入映射表长度（单词总数），词嵌入空间维数（每个单词的特征数）
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.linear(out)

        return out, (h, c)

def train(num_epochs, num_layers, batch_size, hidden_size, seq_length, model, cost, optimizer ):
    for epoch in range(num_epochs):
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))

        for i in tqdm(range(0, ids.size(1) - seq_length, seq_length)):
            inputs = ids[:, i:i + seq_length].to(device)
            targets = ids[:, (i + 1):(i + 1) + seq_length].to(device)

            states = [state.detach() for state in states]
            outputs, states = model(inputs, states)
            loss = cost(outputs, targets.reshape(-1))

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        torch.save(model.state_dict(),f'./model/model{epoch}.pkl')

def test(num_samples, state, model, words):

    # words = jieba.lcut(input_text)
    input_indices = [corpus.dictionary.word2idx[word] for word in words]
    _input = torch.LongTensor(input_indices).unsqueeze(1).to(device)
    # _input = torch.LongTensor(input_indices).view(-1, 1).to(device)

    article = str()  # 输出字符串

    for i in range(num_samples):
        output, state = model(_input, state)

        prob = output[-1].exp()
        word_id = torch.multinomial(prob, num_samples=1).item()
        _input = torch.cat((_input[1:], torch.LongTensor([word_id]).view(1, 1).to(device)), dim=0)
        # _input.fill_(word_id)

        word = corpus.dictionary.idx2word[word_id]
        word = '\n' if word == '<eos>' else word
        article += word

    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(article)

    print(article)

if __name__ == '__main__':

    embed_size = 128#词的特征数
    hidden_size = 1024
    num_layers = 1
    num_epochs = 10
    batch_size = 50
    seq_length = 30#获取序列的长度
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    corpus = Corpus()#语料库实例化
    ids = corpus.get_data('./nlp_corpus/天龙八部.txt', batch_size)
    vocab_size = len(corpus.dictionary)

    model = LSTMmodel(vocab_size, embed_size, hidden_size, num_layers).to(device)#lstm实例化
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(num_epochs, num_layers, batch_size, hidden_size, seq_length, model, cost, optimizer)

    num_samples = 500#生成文本的长度
    lstm = LSTMmodel(vocab_size, embed_size, hidden_size, num_layers).to(device)
    lstm.load_state_dict(torch.load(f'./model/model{num_epochs-1}.pkl'))

    #测试文本
    with open('test.txt', 'r', encoding='utf-8') as f:
        input_text = f.read()
    # input_text = '段誉和'
    words = jieba.lcut(input_text)

    state = (torch.zeros(num_layers, len(words), hidden_size).to(device),
             torch.zeros(num_layers, len(words), hidden_size).to(device))

    test(num_samples,state, lstm, words)


