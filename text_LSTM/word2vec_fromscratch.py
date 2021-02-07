########################################从零开始训练word2vec ################################### 

from gensim.models import KeyedVectors,word2vec,Word2Vec
import jieba
import multiprocessing
 
if __name__ == '__main__':
    sentences = list(word2vec.LineSentence('sku_names.txt'))
    model2 = Word2Vec(size=50,min_count=1)
    model2.build_vocab(sentences)
    model2.train(sentences, total_examples=model2.corpus_count, epochs=100)
    print(model2)
    model2.save('word2vec.model')
 