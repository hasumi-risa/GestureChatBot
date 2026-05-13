import os
import json
import pandas as pd
import numpy as np
from glob import glob

import corenlp
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

courpus_path = "./data/TED_corpus.npy"
nlp_base_dir = './src/NLP/'
corenlp_dir = nlp_base_dir + "stanford-corenlp-full-2013-06-20/"
properties_file = nlp_base_dir + "user.properties"
save_word_weight = './data/tfidf_TED_weight.npy'

parser = corenlp.StanfordCoreNLP(
    corenlp_path=corenlp_dir,
    properties=properties_file)

def tokenizer(text):
    tokens = []
    p = json.loads(parser.parse(text))
    for s in p['sentences']:
        for w in s['words']:
            tokens.append(w[0])
    return tokens

print('Loading subtitles...')
corpus = np.load(courpus_path, allow_pickle=True)

# tf-idf
print('Calclating TS-IDF...')
# max_df: 閾値の割合以上の文書に出現する言葉は無視
vectorizer = TfidfVectorizer(tokenizer=tokenizer, max_df=0.95)
X = vectorizer.fit_transform(corpus)
index = X.toarray().argsort(axis=1)[:,::-1]
feature_names = np.array(vectorizer.get_feature_names())
feature_words = feature_names[index]
print('Done!')

del corpus

xt = X.toarray().T
word2weight = {}
for i in range(len(feature_names)):
    word2weight[feature_names[i]] = np.max(xt[i])
np.save(save_word_weight, word2weight)

print()

# fig, ax = plt.subplots(figsize=(4,12))
# sns.set(font_scale=1.5)
# sns.heatmap(data=vectors.reshape([vectors.shape[0], 1]), cmap="RdBu_r", annot=True)
# ax.set_xticks([])
# ax.set_yticks(np.arange(len(tokens))+0.5)
# ax.set_yticklabels(tokens, rotation=0, fontsize=24)
# fig.tight_layout()
# plt.show()

print()