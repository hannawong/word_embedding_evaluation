import numpy as np
import pandas as pd
import os
import time
import gc
import random
from keras.preprocessing import text,sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors,word2vec,Word2Vec
import pickle,json

wv_from_bin=pickle.load(open("GloVe_50.pkl",'rb'))  ###GLOVE
wv_from_scratch = Word2Vec.load('word2vec.model')  ##word2vec from scratch
wordVectors = np.load("/home/wzh/wzh/glove/wordVectors.npy")  ##word2vec delta training
tokens = json.load(open("/home/wzh/wzh/glove/tokens.json"))  ##word2vec delta training

cate2id = json.load(open("label.json","rb"))
NUM_LABLES = len(cate2id)
NUM_MODELS = 1
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
MAX_LEN = 40
BATCH_SIZE=32
EPOCH=20
MODE = "delta_word2vec" ###GLOVE, word2vec, delta_word2vec
DATASET = "embed_eval_data_sample_general.csv"

def cate_id(x):
    return cate2id[x]

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

def build_matrix(word_index):
    #embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 50))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            if MODE == "GLOVE":
                embedding_matrix[i] = wv_from_bin.word_vec(word)## embedding_index[word]
            if MODE == "word2vec":
                embedding_matrix[i] = wv_from_scratch[word]
            if MODE == "delta_word2vec":
                embedding_matrix[i] = wordVectors[tokens[word], :]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words

dataset = pd.read_csv(DATASET)
data_X = dataset["sku_name"].map(lambda x:str(x))
data_y = dataset["label"].map(lambda x: cate_id(x))
X_train,X_test, y_train, y_test =train_test_split(data_X,data_y,test_size=0.3, random_state=0)
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_LEN)
print(X_train.shape,X_test.shape)

max_features = len(tokenizer.word_index) + 1
print(max_features)
glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index)
print('n unknown words (glove): ', len(unknown_words_glove))
print(y_train,y_test)
X_train_torch = torch.tensor(X_train, dtype=torch.long).cuda()
X_test_torch = torch.tensor(X_test, dtype=torch.long).cuda()
y_train_torch = torch.tensor(np.array(list(y_train)),dtype=torch.int64).cuda()
y_test_torch = torch.tensor(np.array(list(y_test)),dtype=torch.int64).cuda()
train_dataset = data.TensorDataset(X_train_torch, y_train_torch)
test_dataset = data.TensorDataset(X_test_torch,y_test_torch)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_model(model, train, test, loss_fn, output_dim, lr=0.001,
                batch_size=BATCH_SIZE, n_epochs=EPOCH,
                enable_checkpoint_ensemble=True):
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    
    for epoch in range(n_epochs):    
        scheduler.step()
        model.train()
        avg_loss = 0.
        for data in train_loader:
            x_batch = data[:-1]
            y_batch = data[-1]
            y_pred = model(*x_batch)          
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)


        model.eval()
        eval_accuracy = []
        for data in test_loader:
            x_batch = data[:-1]
            y_batch = data[-1].detach().cpu().numpy()
            y_pred = model(*x_batch).detach().cpu().numpy()
            y_pred_num = [np.argmax(line) for line in y_pred]
            eval_accuracy.append(accuracy_score(y_batch,y_pred_num))

        print('Epoch {}/{} \t loss={:.4f}'.format(
              epoch + 1, n_epochs, avg_loss))
        print('accuracy on eval dataset is:',np.mean(np.array(eval_accuracy)))


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
    
class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
    
        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
    
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)
        
    def forward(self, x):
        h_embedding = self.embedding(x)  ###(batchsize, max_len, embedding_size) = (32,40,50)
        h_embedding = self.embedding_dropout(h_embedding) #####(batchsize, max_len, embedding_size) = (32,40,50)
        
        h_lstm1, _ = self.lstm1(h_embedding) ##(batchsize, max_len, hidden_size * 2) = (32,40,256), 因为是双向
        h_lstm2, _ = self.lstm2(h_lstm1) ##(batchsize, max_len, hidden_size * 2) = (32,40,256),
        print(h_lstm2.shape)
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1) ###(32, 256)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1) ##(32,256)
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        aux_result = self.linear_aux_out(hidden)
        softmax_func=nn.Softmax()
        out=softmax_func(aux_result)
        
        return out


def loss_fn(pred,target):
    crossentropyloss=nn.CrossEntropyLoss()
    crossentropyloss_output=crossentropyloss(pred,target)
    return crossentropyloss_output

for model_idx in range(NUM_MODELS):
    print('Model ', model_idx)
    seed_everything(1234 + model_idx)
    model = NeuralNet(glove_matrix, NUM_LABLES)
    model.cuda()
    print(model)

    train_model(model, train_dataset, test_dataset, output_dim=NUM_LABLES, 
                             loss_fn=loss_fn)

