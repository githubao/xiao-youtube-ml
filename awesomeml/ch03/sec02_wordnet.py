#!/usr/bin/env python
# encoding: utf-8

"""
@description: 

@author: BaoQiang
@time: 2017/7/17 19:57
"""

from nltk.corpus import gutenberg,wordnet
from nltk.tokenize import word_tokenize,sent_tokenize
import nltk

# nltk.download_gui()

def run2():
    # syns = wordnet.synsets('program')
    # print(syns)
    # print(syns[0].lemmas()[0].name())
    # print(syns[0].definition())
    # print(syns[0].examples())

    # synonyms = []
    # antonyms = []
    #
    # for syn in wordnet.synsets('good'):
    #     for l in syn.lemmas():
    #         # print('l: ',l)
    #         synonyms.append(l.name())
    #         if l.antonyms():
    #             antonyms.append(l.antonyms()[0].name())
    #
    # print(set(synonyms))
    # print(set(antonyms))

    w1 = wordnet.synset('ship.n.01')
    w2 = wordnet.synset('boat.n.01')

    print(w1.wup_similarity(w2))

def run():
    sample = gutenberg.raw('bible-kjv.txt')
    print(sent_tokenize(sample)[:5])


def main():
    # run()
    run2()

if __name__ == '__main__':
    main()
