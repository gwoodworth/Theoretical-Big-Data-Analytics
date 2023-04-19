#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:23:36 2019

@author: glenwoodworth
"""


import re
import numpy as np
       
def stopWordDict() :
    ''' use this if there are problems with the stop_words module '''
    stopWordList = [] # from http://www.ranks.nl/stopwords
    path = '/Users/glenwoodworth/Documents/Spring 2019/M462/Data/stopWord.txt'
    with open(path, "rU") as f:
        for word in f:
            stopWordList.append(word.strip('\n'))
    stopWordSet = set(stopWordList)
    print(stopWordSet)
    return stopWordSet            
stopWordSet=stopWordDict()
'''
from stop_words import get_stop_words
stop_words = get_stop_words('en')
stopWordSet = set(stop_words) | set(' ') 
'''
print(len(stopWordSet))        


    
from collections import namedtuple
identifier = namedtuple('label','index author')
authors = ['JAY', 'MADISON','HAMILTON' ]


from collections import Counter

path = '/Users/glenwoodworth/Documents/Spring 2019/M462/Data/owners.txt'
paperDict = {}
with open(path, "rU") as f: 
    for string in f:
        string = string.replace('\n', ' ')
        print(string)
        key, value = string.split(',')
        paperDict[int(key)] = value.replace(' ','')


path = '/Users/glenwoodworth/Documents/Spring 2019/M462/Data/FedPapersClean.txt'
#path = '/home/brian/Algorithms/MultnomialNB/Data/pg1404.txt'
sentenceDict = {}
nSentences = 0
sentence = ''
String = ''
with open(path, "rU") as f: 
    for string in f:
        string = string.replace('\n',' ')
        String = String+string
        
############################### 
positionDict = {}
opening  = 'To the People of the State of New York'
counter = 0
for m in re.finditer(opening, String):
    counter+= 1
    positionDict[counter] = [m.end()]

close  = 'PUBLIUS'
counter = 0
for m in re.finditer(close, String):
    counter+= 1
    positionDict[counter].append(m.start())

wordDict = {}

paperCount = 0          
for paperNumber in positionDict:
    b,e = positionDict[paperNumber]
    author = paperDict[paperNumber]
    label = identifier(paperNumber,author)
    paper = String[b+1:e-1]

    for char in '.,?!/;:-"()':
        paper = paper.replace(char,'')
    paper = paper.lower().split(' ')
    for sw in stopWordSet:
        paper = [w for w in paper if w != sw]
        
    #freqDict = Counter(paper)
        
    wordDict[label] = Counter(paper)
    print(label.index,label.author,len(wordDict[label]))
        

table = dict.fromkeys(set(paperDict.values()),0)
for label in wordDict:
    table[label.author] += 1
print(table)    

############ 


############################### 



disputedList = [18,19,20,49,50,51,52,53,54,55,56,57,58,62,63]
trainLabels = []
for label in wordDict:
    number, author = label
    if number not in disputedList:
        trainLabels.append(label)
print(len(trainLabels))    

####################### 

usedDict = {}
for label in trainLabels:
    print(label.author,len(wordDict[label]))
    words = list(wordDict[label].keys())
    for word in words:
        value = usedDict.get(word)
        if value is None:
            usedDict[word] = set([label.author])
        else:
            usedDict[word] = value | set([label.author])

commonList = [word for word in usedDict if len(usedDict[word] ) == 3] 
len(commonList)


#################################### shortening wordList to only common
for label in wordDict:
    D = wordDict[label]
    newDict = {}
    for word in D:
        if word in commonList:
            newDict[word] = D[word]
    wordDict[label] = newDict        
    print(label,len(wordDict[label]))   
    
################################################3
    
logPriors = dict.fromkeys(authors,0)
freqDistnDict = dict.fromkeys(authors)
for label in trainLabels:
    number,author = label
    D = wordDict[label]
    distn = freqDistnDict.get(author)
    if distn is None:
        distn = D
    else:
        for word in D:
            value = distn.get(word)
            if value is not None:
                distn[word] += D[word]
            else:
                distn[word] = D[word]
    freqDistnDict[author] = distn
    logPriors[author] += 1
    
nR = len(trainLabels)
logProbDict = dict.fromkeys(authors,{})
distnDict = dict.fromkeys(authors)
for author in authors:
    authorDict = {}
    logPriors[author] = np.log(logPriors[author]/nR)
    nWords = sum([freqDistnDict[author][word] for word in commonList])
    print(nWords)

    for word in commonList:
        relFreq = freqDistnDict[author][word]/nWords
        authorDict[word] = np.log(relFreq)

    distnDict[author] = [logPriors[author], authorDict]


    
    
    
    
    
    
    
    
nGroups = len(authors)
confusionMatrix = np.zeros(shape = (nGroups,nGroups))
skip = [18,19,20,49,50,51,52,53,54,55,56,57,58,62,63]

for label in wordDict:
    testNumber,testAuthor = label
    
    if testNumber not in skip:
        xi = wordDict[label]
        postProb = dict.fromkeys(authors,0)
        for author in authors:
            logPrior, logProbDict = distnDict[author]
            postProb[author] = logPrior + sum([xi[word] * logProbDict[word] for word in xi])
        postProbList = list(postProb.values())
        postProbAuthors = list(postProb.keys())
        maxIndex = np.argmax(postProbList)
        prediction = postProbAuthors[maxIndex]
        print(testAuthor,prediction)
        i = list(authors).index(testAuthor)
        j = list(authors).index(prediction)
        confusionMatrix[i,j] += 1
        
print(confusionMatrix)
print('acc = ',sum(np.diag(confusionMatrix))/sum(sum(confusionMatrix)))
        
        
    
    
