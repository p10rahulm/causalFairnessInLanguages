import time, numpy as np,pandas as pd
from wordSentimentStatsInDataset import getLogRatioOfWordsFromDataset
from tokenizers.normalizers import NFD, StripAccents, Strip, Lowercase, BertNormalizer
from wordUtils import getSpacyVector
from sklearn import svm


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
    xyDict = {key: (val,xVecsDict[key]) for key, val in yValsDict.items()}
    listOfXs = list(xVecsDict.values())
    listOfYs = list(yValsDict.values())
    nplistOfXs = np.array(listOfXs)
    nplistOfYs = np.array(listOfYs)

    # Load the model
    # fit the model
    classifier = svm.SVC(kernel='linear', C=1)
    classifier.fit(listOfXs, listOfYs)

    # Model 2
    from sklearn.svm import NuSVC
    nusvc = NuSVC(nu=0.03)
    nusvc.fit(nplistOfXs, nplistOfYs)

    # Model 3
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    sckitLinear = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5,max_iter=100000))
    sckitLinear.fit(nplistOfXs, nplistOfYs)
    coefficientsOfHyperplane = sckitLinear.named_steps.linearsvc.coef_[0]
    predssckitLinear = sckitLinear.predict(nplistOfXs)
    accuracyTableSckitLinear = pd.crosstab(nplistOfYs, predssckitLinear).to_numpy()
    accuracy,truePositive,trueNegative,falsePositive,falseNegative = getMetrics(accuracyTableSckitLinear)

    print("Linear SVC on Spacy Embeddings 300D")
    print("Is data linearly separable?")
    print("accuracy: %f"%accuracy)
    print("truePositive: %f" % truePositive)
    print("trueNegative: %f" % trueNegative)
    print("falsePositive: %f" % falsePositive)
    print("falseNegative: %f" % falseNegative)






    # print(xVecsDict)
    print(yValsDict)

    timeTaken = time.time() - startTime
    print("Time Taken:", timeTaken)