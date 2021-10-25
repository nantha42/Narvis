from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "cornell"
corpus = os.path.join("../dataset/", corpus_name)

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

printLines(os.path.join(corpus, "movie_lines.txt"))


def loadLines(fileName,fields):
    lines = {}
    with open(fileName, 'r', encoding = 'iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines

def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations

def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation["lines"])-1):
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()

            if inputLine and targetLine:
                qa_pairs.append([inputLine,targetLine])
    return qa_pairs


datafile = os.path.join(corpus,"formatted_movies_lines.txt")

delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

lines = {}
conversations = []
MOVIE_LINES_FIELDS = ["lineID","characterID","movieID","character","text"]
MOVIE_CONVERSATION_FIELDS = ["character1ID","character2ID", "movieID","utteranceIDs"]

print("\nProcessing corpus...")
lines = loadLines(os.path.join(corpus,"movie_lines.txt"),MOVIE_LINES_FIELDS)
print("\nLoading Conversations...")
conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                  lines, MOVIE_CONVERSATION_FIELDS)

print("\nWriting newly formatted files...")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)

print("Sample : ")
printLines(datafile)


PAD_token = 0
SOS_token = 1
EOS_token = 2

class Voc:
    def __init__(self,name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k,v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print(f'keep_words {len(keep_words)} / {len(self.word2index)} = {len(keep_words)/len(self.word2index):.4f}')

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)


MAX_LENGTH = 10

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
            if unicodedata.category(c) != 'Mn'
        )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])",r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    s = re.sub(r"\s+",r" ",s).strip()
    return s

def readVocs(datafile, corpus_name):
    print("Reading lines...")
    line = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc,pairs

def filterPair(p):
    return len( p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH
