import numpy as np

def match_words(word_list, bert_tokens, search_range=10):
    previous_idx = 0
    matching_list = []
    for i in range(len(word_list)):
        found = False
        for j in range(previous_idx, previous_idx+search_range):
            if j >= len(bert_tokens):
                matching_list.append(-1)
                found = True
                break
            if word_list[i] == bert_tokens[j]:
                matching_list.append(j)
                previous_idx = j
                found = True
                break
        if not found:
            matching_list.append(-1)
            previous_idx += 1
    return matching_list

def reverseMatchingList(matching_list, bert_tokens):
    reversed_list = np.array([-1]*len(bert_tokens))
    for i in range(len(matching_list)):
        if matching_list[i] != -1:
            reversed_list[matching_list[i]] = i
    return reversed_list


def getWordTime(word_time_list, bert_tokens):
    word_list = [str.lower(w[0]) for w in word_time_list]
    bert_tokens = [str.lower(w) for w in bert_tokens]
    matching_list = reverseMatchingList(match_words(word_list, bert_tokens), bert_tokens)
    word_time = []
    for i in range(len(matching_list)):
        if matching_list[i] == -1:
            num = len(word_time)-1
            if num == -1:
                word_time.append([bert_tokens[i], 0, 0])
            else:
                word_time.append([bert_tokens[i], word_time[num][2], word_time[num][2]])
            continue
        word_time.append(word_time_list[matching_list[i]])
    return word_time




if __name__ == "__main__":
    word_list = ['our', 'story', 'therefore', 'meets', 'two', 'dimensions', 'of', 'time', 'a', 'long', 'arc', 'of', 'time', 'that', 'is', 'our', 'lifespan', 'and']
    bert_tokens = ['our', 'story', ',', 'therefore', ',', 'needs', 'two', 'dime', '##nti', '##ons', 'of', 'time', ':', 'a', 'long', 'arc', 'of', 'time', 'that', 'is', 'our', 'lifespan', ',', 'and']

    matching_list = match_words(word_list, bert_tokens)

    for i in range(len(matching_list)):
        if matching_list[i] == -1:
            continue
        print(word_list[i], bert_tokens[matching_list[i]])