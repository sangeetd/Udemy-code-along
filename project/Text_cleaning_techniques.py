import numpy as np
import nltk as nl
from nltk.corpus import stopwords
import string
import re

input = """In this paper, we will talk about the basic steps of text preprocessing. 
These steps are needed for transferring text from human language to 456 
machine-readable format for further processing. We will also discuss text preprocessing tools.123"""

# print ('actual input : \n', input)

def stringToLower(input):
    #converting string to either all lower or upper case
    toAllLower = input.lower()
    # print('str to all lower : \n', toAllLower)
    return toAllLower

def numberRemoved(input):
    #remove number from str
    numbRemoved = re.sub(r'\d+', '', input)
    # print('number removed : \n', numbRemoved)
    return numbRemoved

def punctuationRemoved(input):
    #remove punctuation marks
    punctuationRemoved = input.translate(str.maketrans('','', string.punctuation))
    # print ('punctuation removed : ', punctuationRemoved)
    return punctuationRemoved

def toTokenizedWithoutStopword(input):
    #tokenization
    tokenized = nl.tokenize.word_tokenize(input)
    stopWords = set(stopwords.words('english'))
    # print ('type of tokenized output : \n', type(tokenized))
    # print ('tokenized output : \n', tokenized)

    stop_words_in_str = [i for i in tokenized if i in stopWords]
    tokenized_without_stopword = [i for i in tokenized if i not in stopWords]

    # print('stopword output : \n', stop_words_in_str)
    # print('tokenized without stopword output : \n', tokenized_without_stopword)

    return tokenized_without_stopword


def toLemmatized(input):
    #lemmatization
    lemmatized_tokens = []
    lemmatizer = nl.stem.WordNetLemmatizer()
    for word in input:
        lemmatized_tokens.append(lemmatizer.lemmatize(word))

    # print ('lemmatized tokens output : \n', lemmatized_tokens)
    return lemmatized_tokens

#
# output = toLemmatized(
#     toTokenizedWithoutStopword(
#         punctuationRemoved(
#             numberRemoved(stringToLower(input)
#                           )
#         )
#     )
# )
#
# print ('output : \n', output)


