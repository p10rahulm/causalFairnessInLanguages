import time, numpy as np, pandas as pd
from wordSentimentStatsInDataset import getCleanedWordsFromDataset,computeLogRatios
from tokenizers.normalizers import NFD, StripAccents, Strip, Lowercase, BertNormalizer
from wordUtils import getTokenTransformerEmbedding
from sklearn import svm
from wordUtils import getSpacyVector
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def getClassFromLogRatio(value,threshold=0.5):
    if abs(value)>threshold:
        return 1
    else:
        return -1

def getMetrics(accuracyTableSckitLinear):
    accuracy = (accuracyTableSckitLinear[0,0]+accuracyTableSckitLinear[1,1])/np.sum(accuracyTableSckitLinear)
    truePositive = (accuracyTableSckitLinear[0, 0]) / (accuracyTableSckitLinear[0, 0] + accuracyTableSckitLinear[0, 1])
    trueNegative = (accuracyTableSckitLinear[1, 1]) / (accuracyTableSckitLinear[1, 0] + accuracyTableSckitLinear[1, 1])
    falsePositive = (accuracyTableSckitLinear[0, 1]) / (accuracyTableSckitLinear[0, 0] + accuracyTableSckitLinear[0, 1])
    falseNegative = (accuracyTableSckitLinear[1, 0]) / (accuracyTableSckitLinear[1, 0] + accuracyTableSckitLinear[1, 1])
    return accuracy,truePositive,trueNegative,falsePositive,falseNegative


def getBertEmbeddings(words):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    xVecsDict = {word: getTokenTransformerEmbedding(word, model, tokenizer)[0][1].detach().numpy() for word in words}
    return xVecsDict


def getSpacyEmbeddings(words):
    xVecsDict = {word: getSpacyVector(word) for word in words}
    return xVecsDict

def runNuSVC(listOfXs, listOfYs,nuValue=0.03):
    nusvc = NuSVC(nu=nuValue)
    nusvc.fit(listOfXs, listOfYs)
    nusvcPred = nusvc.predict(nplistOfXs)
    accuracyTableNuSVC = pd.crosstab(nplistOfYs, nusvcPred).to_numpy()
    accuracy, truePositive, trueNegative, falsePositive, falseNegative = getMetrics(accuracyTableNuSVC)
    return accuracy, truePositive, trueNegative, falsePositive, falseNegative


def runLinearSVC(listOfXs, listOfYs,random_state=0, tol=1e-5, max_iter=100000):
    sckitLinear = make_pipeline(StandardScaler(), LinearSVC(random_state=random_state, tol=tol, max_iter=max_iter))
    sckitLinear.fit(listOfXs, listOfYs)
    coefficientsOfHyperplane = sckitLinear.named_steps.linearsvc.coef_[0]
    predssckitLinear = sckitLinear.predict(nplistOfXs)
    accuracyTableSckitLinear = pd.crosstab(nplistOfYs, predssckitLinear).to_numpy()
    accuracy, truePositive, trueNegative, falsePositive, falseNegative = getMetrics(accuracyTableSckitLinear)
    return accuracy, truePositive, trueNegative, falsePositive, falseNegative

def separateWordsInSentence():
    return 0



def printMetrics(embeddingName,modelName,accuracy, truePositive, trueNegative, falsePositive, falseNegative):
    print("-" * 50)
    if(embeddingName=='transformer' and modelName == 'nuSVC'):
        print("NuSVC on Transformer embeddings 768D")
    elif(embeddingName=='transformer' and modelName == 'linearSVC'):
        print("LinearSVC on Transformer Embeddings 768D")
    elif(embeddingName=='spacy' and modelName == 'nuSVC'):
        print("NuSVC on Spacy embeddings 300D")
    else:
        print("LinearSVC on Spacy embeddings 300D")

    print("accuracy: %f" % accuracy)
    print("truePositive: %f" % truePositive)
    print("trueNegative: %f" % trueNegative)
    print("falsePositive: %f" % falsePositive)
    print("falseNegative: %f" % falseNegative)
    print("-" * 50)


if __name__ == "__main__":
    startTime = time.time()

    # filename = "datasets/combDev.csv"
    filename = "datasets/combTrain.csv"
    normalizerSequence = [NFD(), StripAccents(), Strip(), Lowercase()]
    sentenceColName = 'Text'
    sep = '\t'
    # First create the true dataset.
    posCleanedSentences, negCleanedSentences = \
        getCleanedWordsFromDataset(filename, normalizerSequence, sentenceColName=sentenceColName, sep=sep)
    sortedLogRatiosOfWords, sortedAbsLogRatiosOfWords = computeLogRatios(posCleanedSentences, negCleanedSentences)

    words = list(sortedLogRatiosOfWords.keys())
    # methods = ['spacy','transformer']
    methods = ['spacy']
    for method in methods:
        if method == 'transformer':
            xVecsDict = getBertEmbeddings(words)
        else:
            xVecsDict = getSpacyEmbeddings(words)

        yValsDict = {key: getClassFromLogRatio(val,threshold=0.5) for key, val in sortedAbsLogRatiosOfWords.items()}
        # xyDict = {key: (val,xVecsDict[key]) for key, val in yValsDict.items()}
        nplistOfXs = np.array(list(xVecsDict.values()))
        nplistOfYs = np.array(list(yValsDict.values()))
        # These are the true values.








    timeTaken = time.time() - startTime
    print("Time Taken:", timeTaken)