import time
import pandas as pd
from collections import defaultdict
from wordUtils import normalizeSentence
from tokenizers.normalizers import NFD, StripAccents, Strip, Lowercase, BertNormalizer
from math import log

def getPositiveNegativeDFs(filename, sep='\t'):
    combined = pd.read_csv(filename, delimiter="\t", header=0)

    positive = combined[combined["Sentiment"] == "Positive"]
    positive = positive.reset_index(drop=True)
    negative = combined[combined["Sentiment"] == "Negative"]
    negative = negative.reset_index(drop=True)
    return combined, positive, negative


def cleanSentences(listOfSentences,normalizerSequence):
    cleanedSentences=[]
    for sentence in listOfSentences:
        normalizedSentence = normalizeSentence(sentence, normalizerSequence)
        normalizedSentence = ''.join(character for character in normalizedSentence if (character.isalnum() or character.isspace()))
        cleanedSentences.append(normalizedSentence)
    return cleanedSentences



def getWordStats(listOfSentences):
    wordCounts = defaultdict(int)
    probWord = defaultdict(float)
    numWords=0
    for sentence in listOfSentences:
        words=sentence.split(' ')
        for word in words:
            wordCounts[word]+=1
            probWord[word]+=1
            numWords+=1
    for word in probWord:
        probWord[word] = probWord[word]/numWords
    return wordCounts,probWord,numWords


def getLogRatiosOfWords(posProbDict,negProbDict,defaultLogBound=10):
    posWords = [key for key, vals in posProbDict.items() if key not in negProbDict]
    negWords = [key for key, vals in negProbDict.items() if key not in posProbDict]
    bothWords = [key for key, vals in posProbDict.items() if key in negProbDict]

    wordProbRatios = defaultdict(float)
    for word in posWords:
        wordProbRatios[word] = defaultLogBound
    for word in negWords:
        wordProbRatios[word] = -defaultLogBound

    for word in bothWords:
        try:
            wordProbRatios[word] = log(probWordPos[word]/probWordNeg[word])
        except:
            print("word=",word,"posProbDict[word]=",posProbDict[word],"negProbDict[word]=",negProbDict[word])
        wordProbRatios[word] = max(min(wordProbRatios[word],defaultLogBound),-defaultLogBound)
    return wordProbRatios




if __name__ == "__main__":
    startTime = time.time()

    # combined = pd.read_csv("datasets/combDev.csv",delimiter="\t", header=0)
    # positive = combined[combined["Sentiment"] == "Positive"]
    # positive = positive.reset_index(drop=True)
    # negative = combined[combined["Sentiment"] == "Negative"]
    # negative = negative.reset_index(drop=True)
    combined, positive, negative = getPositiveNegativeDFs("datasets/combDev.csv", sep='\t')
    posSentences = positive['Text'].tolist()
    negSentences = negative['Text'].tolist()

    normalizerSequence = [NFD(), StripAccents(), Strip(), Lowercase()]
    posCleanedSentences = cleanSentences(posSentences, normalizerSequence)
    negCleanedSentences = cleanSentences(negSentences, normalizerSequence)

    wordCountPos, probWordPos, numWordsPos = getWordStats(posCleanedSentences)
    wordCountNeg, probWordNeg, numWordsNeg = getWordStats(negCleanedSentences)
    logRatiosOfWords = getLogRatiosOfWords(probWordPos, probWordNeg, defaultLogBound=5)
    absLogRatiosOfWords = {key:abs(val) for key,val in logRatiosOfWords.items()}
    sortedAbsLogRatiosOfWords = dict(sorted(absLogRatiosOfWords.items(), key=lambda item: item[1]))


    timeTaken = time.time() - startTime
    print("Time Taken:", timeTaken)
