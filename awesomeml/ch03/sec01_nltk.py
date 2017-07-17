#!/usr/bin/env python
# encoding: utf-8

"""
@description: 

@author: BaoQiang
@time: 2017/7/17 17:07
"""

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
import nltk
from nltk.stem import WordNetLemmatizer

"""
nltk.download()
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
"""

# nltk.download()

def run7():
    lemmatizer = WordNetLemmatizer()
    for word in ['cats','better','python','best','ran']:
        print(lemmatizer.lemmatize(word))

    print(lemmatizer.lemmatize('better',pos='a'))

def run6():
    train_text = state_union.raw('2005-GWBush.txt')
    sample_text = state_union.raw('2006-GWBush.txt')

    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
    tokenized = custom_sent_tokenizer.tokenize(sample_text)

    def process_content():
        for i in tokenized[:3]:
            try:
                words = nltk.word_tokenize(i)
                tagged = nltk.pos_tag(words)

                named_entity = nltk.ne_chunk(tagged)
                named_entity.draw()

            except Exception as e:
                print(e)

    process_content()

def run5():
    train_text = state_union.raw('2005-GWBush.txt')
    sample_text = state_union.raw('2006-GWBush.txt')

    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
    tokenized = custom_sent_tokenizer.tokenize(sample_text)

    def process_content():
        for i in tokenized[:3]:
            try:
                words = nltk.word_tokenize(i)
                tagged = nltk.pos_tag(words)

                chunked_gram = """Chunk: {<.*>+}
                }<VB.?|IN|DT|TO>+{"""
                chunked_parser = nltk.RegexpParser(chunked_gram)
                chunked = chunked_parser.parse(tagged)

                # print(tagged)
                # print(chunked)
                chunked.draw()
            except Exception as e:
                print(e)

    process_content()

def run4():
    train_text = state_union.raw('2005-GWBush.txt')
    sample_text = state_union.raw('2006-GWBush.txt')

    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
    tokenized = custom_sent_tokenizer.tokenize(sample_text)

    def process_content():
        for i in tokenized:
            try:
                words = nltk.word_tokenize(i)
                tagged = nltk.pos_tag(words)

                chunked_gram = """Chunk: {<RB.?>*<VB.?>*<NNP><NN>*}"""
                chunked_parser = nltk.RegexpParser(chunked_gram)
                chunked = chunked_parser.parse(tagged)

                # print(tagged)
                # print(chunked)
                chunked.draw()
            except Exception as e:
                print(e)

    process_content()


def run3():
    ps = PorterStemmer()
    example_words = ['python', 'pythoner', 'pythoning', 'pythoned', 'pythonly', ]
    for w in example_words:
        print(ps.stem(w))

    new_text = 'It is very important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once.'
    for w in word_tokenize(new_text):
        print(ps.stem(w))


def run2():
    example_text = 'This is a example showing off stop word filtration.'

    stop_words = set(stopwords.words('english'))
    # for word in stop_words:
    #     print(word)

    filterd_sentences = [word for word in word_tokenize(example_text) if word not in stop_words]
    for item in filterd_sentences:
        print(item)


def run():
    example_text = 'Hello Mr. Smith, how are you doing today? The weather is great and Python is awesome. The sky is pinkish-blue. You should not eat cardboard.'
    # example_text = '我饿了，我想吃东西。'

    for word in word_tokenize(example_text):
        print(word)

    for sent in sent_tokenize(example_text):
        print(sent)


def main():
    # run()
    # run2()
    # run3()
    # run4()
    # run5()
    # run6()
    run7()


if __name__ == '__main__':
    main()

'''
POS tag list:
CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
'''
