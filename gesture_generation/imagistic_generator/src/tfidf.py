import os
import token
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

def tfidf(corpus, input_text, tokenizer):
    corpus.insert(0, input_text)

    # tf-idf
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(corpus)
    new_features = tfidf_vectorizer.transform([input_text])

    df = pd.DataFrame(data=new_features.toarray(),
                      columns=tfidf_vectorizer.get_feature_names())
    
    tokens = tokenizer(input_text)
    vectors = np.zeros([len(tokens)])
    for i in range(len(tokens)):
        if str.lower(tokens[i]) in df.keys():
            vectors[i] = df[str.lower(tokens[i])]

    return tokens, vectors


if __name__ == '__main__':
    
    input_text = "all I have to do is build a platform and all these people are going to put their stuff on top and I sit back and roll it in ?"
    csv_path = "./data/annotation_results_integrated_20210528.xlsx"
    
    df = pd.read_excel(csv_path)

    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    stemmer = nltk.stem.PorterStemmer()

    corpus = list(df['Text'].dropna())

    tokens, vectors = tfidf(corpus, input_text, tokenizer=nltk.word_tokenize)

    print(tokens)
    print(vectors)

    fig, ax = plt.subplots()
    sns.heatmap(data=vectors.reshape([vectors.shape[0], 1]), cmap="RdBu_r", annot=True)
    ax.set_yticks(np.arange(len(tokens)), rotation=90)
    ax.yaxis.set_ticklabels(tokens)
    # plt.colorbar()
    plt.savefig("tmp.png")
