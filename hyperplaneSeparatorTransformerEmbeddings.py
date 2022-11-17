import time, numpy as np
from wordSentimentStatsInDataset import getLogRatioOfWordsFromDataset
from tokenizers.normalizers import NFD, StripAccents, Strip, Lowercase, BertNormalizer
from wordUtils import getTokenTransformerEmbedding
from sklearn import svm
from transformers import DistilBertTokenizerFast, DistilBertModel


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

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    xVecsDict = {key:getTokenTransformerEmbedding(key, model, tokenizer) for key,val in sortedAbsLogRatiosOfWords.items()}
    yValsDict = {key: getClassFromLogRatio(val,threshold=0.5) for key, val in sortedAbsLogRatiosOfWords.items()}
    xyDict = {key: (val,xVecsDict[key]) for key, val in yValsDict.items()}
    listOfXs = list(xVecsDict.values())
    listOfYs = list(yValsDict.values())
    nplistOfXs = np.array(listOfXs)
    nplistOfYs = np.array(listOfYs)


    # Load an SVM model
    # fit the model
    classifier = svm.SVC(kernel='linear', C=1)
    classifier.fit(listOfXs, listOfYs)






    # print(xVecsDict)
    print(yValsDict)

    timeTaken = time.time() - startTime
    print("Time Taken:", timeTaken)