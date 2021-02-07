# %%
import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json

from word2vec import *
from sgd import *
import pickle
import gensim.downloader as api
wv_from_bin = api.load("glove-wiki-gigaword-50")
pickle.dump(wv_from_bin,open("GloVe_50.pkl","wb"))

# Check Python Version
import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5
def initialize():
    # Reset the random seed to make sure that everyone gets the same results
    random.seed(314)
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    print(tokens)
    json.dump(tokens,open("tokens.json","w"))

    nWords = len(tokens)
    print(nWords,"word")


    # We are going to train 10-dimensional vectors for this assignment
    dimVectors = 50
    EPOCH=100

    # Context size
    C = 5
    # Reset the random seed to make sure that everyone gets the same results
    random.seed(31415)
    np.random.seed(9265)
    in_glove=0
    wordVectors=np.zeros((2*nWords,dimVectors))

    for i in range(0,nWords):
        if list(tokens.keys())[i] in wv_from_bin.vocab.keys():
            wordVectors[i]=np.array(wv_from_bin.word_vec(list(tokens.keys())[i]))
            in_glove+=1
        else:
            wordVectors[i]=(np.random.rand(1, dimVectors) - 0.5) /dimVectors

    for i in range(nWords,2*nWords):
        if list(tokens.keys())[i-nWords] in wv_from_bin.vocab.keys():
            wordVectors[i]=np.array(wv_from_bin.word_vec(list(tokens.keys())[i-nWords]))

    print(wordVectors)
    print(in_glove, " in GloVe")

    wordVectors = sgd(
        lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
            negSamplingLossAndGradient),
        wordVectors, 0.3, EPOCH, None, True, PRINT_EVERY=1)
    # Note that normalization is not called here. This is not a bug,
    # normalizing during training loses the notion of length.

    print("sanity check: cost at convergence should be around or below 10")

    # concatenate the input and output word vectors
    wordVectors = np.concatenate(
        (wordVectors[:nWords,:], wordVectors[nWords:,:]),
        axis=0)
    print(wordVectors.shape)
    # %%
    np.save("wordVectors",wordVectors)
    
initialize()
wordVectors = np.load("wordVectors.npy")
tokens = json.load(open("tokens.json"))
visualizeWords = ["summer","spring","second","first","womens","women","men","mens","blue","pink",\
                  "wine","vinegar","grey","green","silica","carbon","jade","flower","mobile","zinc","aluminum",\
                  "snake","pet","cat","cats","yellow","chinese","african","bohemian","mom","daughter","girl",\
                  "autumn","winter","korean","sister","clock"]
visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])

for i in range(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i],
        bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('word_vectors_.png')
# %%
