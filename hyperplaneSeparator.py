import time
from wordSentimentStatsInDataset import getLogRatioOfWordsFromDataset
from tokenizers.normalizers import NFD, StripAccents, Strip, Lowercase, BertNormalizer
from wordUtils import getSpacyVector


def getClassFromLogRatio(value,threshold=0.5):
    if abs(value)>threshold:
        return 1
    else:
        return -1


if __name__ == "__main__":
    startTime = time.time()

    filename = "datasets/combDev.csv"
    normalizerSequence = [NFD(), StripAccents(), Strip(), Lowercase()]
    sentenceColName = 'Text'
    sep = '\t'

    sortedLogRatiosOfWords, sortedAbsLogRatiosOfWords = \
        getLogRatioOfWordsFromDataset(filename, normalizerSequence, sentenceColName=sentenceColName, sep=sep)

    xVecsDict = {key:getSpacyVector(key) for key,val in sortedAbsLogRatiosOfWords.items()}
    yValsDict = {key: getClassFromLogRatio(val,threshold=0.5) for key, val in sortedAbsLogRatiosOfWords.items()}
    # print(xVecsDict)
    print(yValsDict)

    timeTaken = time.time() - startTime
    print("Time Taken:", timeTaken)