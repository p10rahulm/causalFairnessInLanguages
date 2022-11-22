import time, numpy as np, pandas as pd, torch, random
from wordSentimentStatsInDataset import getCleanedWordsFromDataset, computeLogRatios
from tokenizers.normalizers import NFD, StripAccents, Strip, Lowercase, BertNormalizer
from wordUtils import getSpacyVector, getTokenTransformerEmbedding, getListOfWordsFromSentences
from transformers import DistilBertTokenizerFast, DistilBertModel
from tqdm import tqdm
from collections import Counter


def getClassFromLogRatio(value, threshold=0.5):
    if abs(value) > threshold:
        return 1
    else:
        return -1


def getMetrics(confusionMatrix):
    accuracy = (confusionMatrix[0, 0] + confusionMatrix[1, 1]) / np.sum(confusionMatrix)
    truePositive = (confusionMatrix[0, 0]) / (confusionMatrix[0, 0] + confusionMatrix[0, 1])
    trueNegative = (confusionMatrix[1, 1]) / (confusionMatrix[1, 0] + confusionMatrix[1, 1])
    falsePositive = (confusionMatrix[0, 1]) / (confusionMatrix[0, 0] + confusionMatrix[0, 1])
    falseNegative = (confusionMatrix[1, 0]) / (confusionMatrix[1, 0] + confusionMatrix[1, 1])
    return accuracy, truePositive, trueNegative, falsePositive, falseNegative


def getBertEmbeddings(words, device):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.to(device)
    model.eval()
    wordsVecDict = {word: getTokenTransformerEmbedding(word, model, tokenizer, device)[0][1] for word in words}
    # print("wordsVecDict[the]",wordsVecDict["the"])
    return wordsVecDict


def getSpacyEmbeddings(words):
    wordsVecDict = {word: getSpacyVector(word) for word in words}
    return wordsVecDict


def printMetrics(embeddingName, modelName, accuracy, truePositive, trueNegative, falsePositive, falseNegative):
    print("-" * 50)
    if (embeddingName == 'transformer' and modelName == 'nuSVC'):
        print("NuSVC on Transformer embeddings 768D")
    elif (embeddingName == 'transformer' and modelName == 'linearSVC'):
        print("LinearSVC on Transformer Embeddings 768D")
    elif (embeddingName == 'spacy' and modelName == 'nuSVC'):
        print("NuSVC on Spacy embeddings 300D")
    else:
        print("LinearSVC on Spacy embeddings 300D")

    print("accuracy: %f" % accuracy)
    print("truePositive: %f" % truePositive)
    print("trueNegative: %f" % trueNegative)
    print("falsePositive: %f" % falsePositive)
    print("falseNegative: %f" % falseNegative)
    print("-" * 50)


def getAvgZW(sentences, w, b, device):
    numZ, numW, sumZ, sumW = 0, 0, 0.0, 0.0
    sentences = " ".join(sentences)
    for word in tqdm(sentences.split()):
        wordVec = torch.from_numpy(getSpacyVector(word)).to(device)
        if (wordVec @ w + b < 0):
            numW += 1
            sumW += wordVec
        else:
            numZ += 1
            sumZ += wordVec
    avgW = sumW / numW if numW > 0 else 0
    avgZ = sumZ / numZ if numW > 0 else 0
    return avgW, avgZ


def getWordVecScore(wordVec, w, b):
    return wordVec @ w + b


def computeAugmentedWordsDict(wordsDict, w, b):
    outDict = {}
    for word, vec in wordsDict.items():
        wordScore = getWordVecScore(vec, w, b)
        outDict[word] = (vec, wordScore, wordScore >= 0)
    return outDict


def getSumNumWZFromWordsDict(wordsDict, device="cpu"):
    avgW, avgZ = torch.zeros(1, requires_grad=True, device=device), torch.zeros(1, requires_grad=True, device=device)
    numW, numZ = 0, 0
    for word, (vec, score, inZ) in tqdm(wordsDict.items()):
        if inZ:
            avgZ = avgZ + vec
            numZ += 1
        else:
            avgW = avgW + vec
            numW += 1
    avgW = avgW / numW if numW != 0 else 0.0
    avgZ = avgZ / numZ if numZ != 0 else 0.0
    return avgW, avgZ


def computeAvgLossFromWordsDict(posWordsDict, negWordsDict, device):
    avgWPos, avgZPos = getSumNumWZFromWordsDict(posWordsDict, device)
    avgWNeg, avgZNeg = getSumNumWZFromWordsDict(negWordsDict, device)
    return avgWPos, avgZPos, avgWNeg, avgZNeg


if __name__ == "__main__":
    startTime = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(8)
    random.seed(8)
    print("device=", device)

    filename = "datasets/combDev.csv"
    # filename = "datasets/combTrain.csv"
    normalizerSequence = [NFD(), StripAccents(), Strip(), Lowercase()]
    sentenceColName = 'Text'
    sep = '\t'
    # First create the true dataset.
    posCleanedSentences, negCleanedSentences = \
        getCleanedWordsFromDataset(filename, normalizerSequence, sentenceColName=sentenceColName, sep=sep)
    sortedLogRatiosOfWords, sortedAbsLogRatiosOfWords = computeLogRatios(posCleanedSentences, negCleanedSentences)

    print("finished ps and neg cleaned sentences")
    posWords = getListOfWordsFromSentences(posCleanedSentences)
    negWords = getListOfWordsFromSentences(negCleanedSentences)
    allwords = posWords + negWords
    print("got list of words")

    # Get bert embeddings
    posVecs = getBertEmbeddings(posWords, device)
    negVecs = getBertEmbeddings(negWords, device)
    allVecs = posVecs | negVecs
    print("got BERT embeddings")
    # Randomly initialize Hyperplane to 768D for size of bert embeddings
    Hyp_w = torch.randn(768, 1, requires_grad=True, device=device)  # 768 rows and 1 columns for spacy
    Hyp_b = torch.randn(1, 1, requires_grad=True, device=device)

    posWordsCount = Counter(posWords)
    negWordsCount = Counter(negWords)

    print("created counters")

    for words, wordCounts in posWordsCount.items():
        posVecs[words] = posVecs[words] * wordCounts

    posVecTuples = tuple(posVecs.values())
    posVecMatrix = torch.stack(posVecTuples, 0)

    for words, wordCounts in negWordsCount.items():
        negVecs[words] = negVecs[words] * wordCounts

    negVecTuples = tuple(negVecs.values())
    negVecMatrix = torch.stack(negVecTuples, 0)
    print("posVecMatrix.shape", posVecMatrix.shape)
    print("negVecMatrix.shape", negVecMatrix.shape)

    posValMatrix = torch.matmul(posVecMatrix, Hyp_w)
    posValMatrix = posValMatrix + torch.matmul(torch.ones(posValMatrix.shape[0], 1, device=device), Hyp_b)

    negValMatrix = torch.matmul(negVecMatrix, Hyp_w)
    negValMatrix = negValMatrix + torch.matmul(torch.ones(negValMatrix.shape[0], 1, device=device), Hyp_b)

    posAvgZVecs = torch.sign(torch.nn.functional.relu(posValMatrix))
    posAvgWVecs = torch.sign(torch.nn.functional.relu(-posValMatrix))
    negAvgZVecs = torch.sign(torch.nn.functional.relu(negValMatrix))
    negAvgWVecs = torch.sign(torch.nn.functional.relu(-negValMatrix))

    posAvgZVecs = posAvgZVecs@posValMatrix
    posAvgWVecs = posAvgWVecs @ posValMatrix

    timeTaken = time.time() - startTime
    print("Time Taken:", timeTaken)

    exit()



    posZMatrix = (posValMatrix >= 0) * 1
    posWMatrix = (posValMatrix < 0) * 1

    posnumZs = sum(posZMatrix)[0]
    posnumWs = sum(posWMatrix)[0]

    negZMatrix = (negValMatrix >= 0) * 1
    negWMatrix = (negValMatrix < 0) * 1

    negnumZs = sum(negZMatrix)[0]
    negnumWs = sum(negWMatrix)[0]

    # posAvgZVecs = torch.sign(torch.nn.functional.relu(posValMatrix))
    # posAvgWVecs = torch.sign(torch.nn.functional.relu(-posValMatrix))
    # negAvgZVecs = torch.sign(torch.nn.functional.relu(negValMatrix))
    # negAvgWVecs = torch.sign(torch.nn.functional.relu(-negValMatrix))


    posWordsDict = computeAugmentedWordsDict(posVecs, Hyp_w, Hyp_b)
    negWordsDict = computeAugmentedWordsDict(negVecs, Hyp_w, Hyp_b)
    avgWPos, avgZPos, avgWNeg, avgZNeg = computeAvgLossFromWordsDict(posWordsDict, negWordsDict, device)
    print("computed avgWZPosNeg scores")
    # yValsDict = {key: getClassFromLogRatio(val, threshold=0.5) for key, val in sortedAbsLogRatiosOfWords.items()}

    # # True Values
    # inputs = np.array(list(xVecsDict.values())).astype('float32')
    # target = np.array(list(yValsDict.values())).astype('float32')
    #
    # inputs = torch.from_numpy(inputs)
    # target = torch.from_numpy(target)
    # # Make it a column vector (of unknown number of rows)
    # target = target.reshape(-1, 1)

    # avgWPos, avgZPos = getAvgZW(posCleanedSentences, Hyp_w, Hyp_b, device)
    # avgWNeg, avgZNeg = getAvgZW(negCleanedSentences, Hyp_w, Hyp_b, device)
    # We now compute a loss score for the hyperplane. The lower the score, the better the performance of hyperplane
    lossScore = torch.nn.functional.mse_loss(avgWPos, avgWNeg)

    print("lossScore=", lossScore)
    lossScore.backward()
    print("Hyp_w.grad=", Hyp_w.grad)
    print("Hyp_b.grad=", Hyp_b.grad)


