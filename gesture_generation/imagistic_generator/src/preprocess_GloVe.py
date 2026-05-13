import numpy as np
from tqdm import tqdm

def load_emb_file(emb_path):
    vects = []
    idx = 0
    word2idx = dict()
    idx2word = dict()
    with open(emb_path, 'r', encoding="utf-8_sig") as f:
        for l in tqdm(f):
            line = l.split()
            word = line[0]
            w_vec = np.array(line[1:]).astype(np.float)

            vects.append(w_vec)
            word2idx[word] = idx
            idx2word[idx] = word
            idx += 1
    
    return np.array(vects), word2idx, idx2word


    bert_option = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_option)
    bert_model = BertModel.from_pretrained(bert_option)

    tokens = tokenizer.tokenize(input_text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids_tensor = torch.tensor(ids)
    ids_tensor = ids_tensor.reshape(1, -1)
    embed_vectors, _, = bert_model(ids_tensor)

    t = nltk.pos_tag(text)

if __name__ == '__main__':

    glove_file = '../../OtherMethods/Co-Speech_Gesture_Generation_my/data/glove.6B.300d.txt'
    save_path = './data/glove.npy'

    print('Loading Glove...')
    emb_vec, word2idx, idx2word = load_emb_file(glove_file)

    glove_data = {'emb_vec':emb_vec, 'word2idx':word2idx, 'idx2word':idx2word}
    np.save(save_path, glove_data)
