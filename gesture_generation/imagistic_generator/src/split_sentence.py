import os
import json
import pprint
import corenlp
from copy import copy

""" The function that extract a tree block surronded by () """
def extractBlock(tree):
    first_parenth = tree.find('(')
    if first_parenth == -1:
        return -1, -1
    tree_ = tree[first_parenth:]
    cnt = 0
    end_parenth = -1
    for i,c in enumerate(tree_):
        if c == '(':
            cnt += 1
        elif c == ')':
            cnt -= 1
        if cnt == 0:
            end_parenth = i
            break
    if end_parenth == -1:
        print("The number of parentheses is wrong")
        sys.exit(-1)
    tree__ = tree_[:end_parenth+1]
    # return tree__
    first_idx = first_parenth
    end_idx = end_parenth+first_parenth+1
    return first_idx, end_idx

""" The function that extract words from tree """
def block2words(tree):
    word_list = []
    while(1):
        end_parenth = tree.find(')')
        if end_parenth == -1:
            break
        if end_parenth == 0 or tree[end_parenth-1] == ')':
            tree = tree[end_parenth+1:]
            continue
        word = tree[tree[:end_parenth].rfind(' ')+1:end_parenth]
        word_list.append(word)
        tree = tree[end_parenth:]
    return word_list

"""Split the parsed tree to words"""
def splitTree(tree, max_word_num):
    words_list = []
    origin_tree = copy(tree)
    progress = []
    while(1):
        first_idx, end_idx = extractBlock(tree)
        if first_idx == -1 and end_idx == -1:
            # end_idx = origin_tree.find(new_tree)
            end_idx = origin_tree[sum(progress[:-1]):].find(new_tree)
            if end_idx == -1:
                break
            # tree = origin_tree[end_idx + len(new_tree) + 1:]
            tree = origin_tree[sum(progress[:-1]) + end_idx + len(new_tree) + 1:]
            progress.append(len(new_tree) + 1)
            if sum(progress) >= len(origin_tree):
                break
            continue
        new_tree = tree[first_idx: end_idx]

        word_list = block2words(new_tree)
        if len(word_list) <= max_word_num:
            words_list.append(word_list)
            tree = tree[end_idx:]
            progress.append(end_idx)
        else:
            tree = new_tree[1:]
            progress.append(1)

    return words_list

"""Split text to words"""
def splitSentence(text, max_word_num, parser=None):
    words_list = []

    if parser:
        result_json = json.loads(parser.parse(text))
        # pprint.pprint(result_json)
        for sentence in result_json['sentences']:
            tree = sentence['parsetree']
            wl = splitTree(tree, max_word_num)
            for w in wl:
                words_list.append(w)
    else:
        word_list = text.split()
        for i in range(0, len(word_list), max_word_num):
            words_list.append(word_list[i:i+max_word_num])
            
    return words_list

if __name__ == '__main__':

    base_dir = './src/NLP/'
    corenlp_dir = base_dir + "stanford-corenlp-full-2013-06-20/"
    properties_file = base_dir + "user.properties"
    max_word_num = 4
    text = "a doughnut or a half-moon shape with a large, central hole."
    # text = "The quick brown fox jumps over the lazy dog"
    # text = "they keep spinning with the same axis, indefinitely. Hubble kind of rotates around them, and so it can orient itself"
    # text = "and the radiation of flowering plants, or angiosperms, onto land."


    # generate parser
    parser = corenlp.StanfordCoreNLP(
        corenlp_path=corenlp_dir,
        properties=properties_file) # propertiesを設定

    words_list = splitSentence(text, max_word_num, parser)

    print("Text:  ", text)
    print("Splitted Text:  ", words_list)

