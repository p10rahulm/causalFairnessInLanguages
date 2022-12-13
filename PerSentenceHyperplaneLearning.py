import time, numpy as np, torch, random,datetime
from wordSentimentStatsInDataset import getCleanedWordsFromDataset, computeLogRatios
from tokenizers.normalizers import NFD, StripAccents, Strip, Lowercase, BertNormalizer
from wordUtils import getSpacyVector, getTokenTransformerEmbedding, getListOfWordsFromSentences
from transformers import DistilBertTokenizerFast, DistilBertModel
from tqdm import tqdm
from collections import Counter
from torch.utils.tensorboard import SummaryWriter


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


def getAvgWZVecsForPosNeg(Hyp_w, Hyp_b, posVecMatrix, negVecMatrix, alpha):
    posValMatrix = torch.matmul(posVecMatrix, Hyp_w)
    posValMatrix = posValMatrix + torch.matmul(torch.ones(posValMatrix.shape[0], 1, device=device), Hyp_b)

    negValMatrix = torch.matmul(negVecMatrix, Hyp_w)
    negValMatrix = negValMatrix + torch.matmul(torch.ones(negValMatrix.shape[0], 1, device=device), Hyp_b)

    # posValSignsZVecs = torch.sigmoid(torch.mul(torch.nn.functional.relu(posValMatrix), alpha))
    # posValSignsWVecs = torch.sigmoid(torch.mul(torch.nn.functional.relu(-posValMatrix), alpha))
    # negValSignsZVecs = torch.sigmoid(torch.mul(torch.nn.functional.relu(negValMatrix), alpha))
    # negValSignsWVecs = torch.sigmoid(torch.mul(torch.nn.functional.relu(-negValMatrix), alpha))

    posValSignsZVecs = torch.sigmoid(torch.mul(posValMatrix, alpha))
    posValSignsWVecs = torch.sigmoid(torch.mul(-posValMatrix, alpha))
    negValSignsZVecs = torch.sigmoid(torch.mul(negValMatrix, alpha))
    negValSignsWVecs = torch.sigmoid(torch.mul(-negValMatrix, alpha))

    posAvgZVec = posValSignsZVecs.T @ posVecMatrix / sum(posValSignsZVecs)
    posAvgWVec = posValSignsWVecs.T @ posVecMatrix / sum(posValSignsWVecs)

    negAvgZVec = negValSignsZVecs.T @ negVecMatrix / sum(negValSignsZVecs)
    negAvgWVec = negValSignsWVecs.T @ negVecMatrix / sum(negValSignsWVecs)

    # print("posValMatrix.shape", posValMatrix.shape)
    # print("negValMatrix.shape", negValMatrix.shape)

    # print("posValSignsZVecs.shape", posValSignsZVecs.shape)
    # print("posValSignsWVecs.shape", posValSignsWVecs.shape)
    # print("negValSignsZVecs.shape", negValSignsZVecs.shape)
    # print("negValSignsWVecs.shape", negValSignsWVecs.shape)
    #
    # print("posVecMatrix.shape", posVecMatrix.shape)
    # print("negVecMatrix.shape", negVecMatrix.shape)

    # print("posAvgZVec.shape", posAvgZVec.shape)
    # print("posAvgWVec.shape", posAvgWVec.shape)
    # print("negAvgZVec.shape", negAvgZVec.shape)
    # print("negAvgWVec.shape", negAvgWVec.shape)
    return posAvgZVec, posAvgWVec, negAvgZVec, negAvgWVec, posValMatrix, negValMatrix, \
           posValSignsZVecs, posValSignsWVecs, negValSignsZVecs, negValSignsWVecs


def computeLossGradients(Hyp_w, Hyp_b, posVecMatrix, negVecMatrix, alpha):
    # Sizes:
    # Hyp_w = dim * 1
    # Hyp_w = 1 * 1
    # posVecMatrix = sizePos * dim
    # negVecMatrix = sizeNeg * dim
    # alpha = scalar
    # posValMatrix = sizePos * 1
    # negValMatrix = sizePos * 1
    # posValSigmoidsZVecs = sizePos * 1
    # posValSigmoidsWVecs = sizePos * 1
    # negValSigmoidsZVecs = sizePos * 1
    # negValSigmoidsWVecs = sizePos * 1
    # posAvgZVec = 1 * dim
    # posAvgWVec = 1 * dim
    # negAvgZVec = 1 * dim
    # negAvgWVec = 1 * dim
    # diffAvgZ = 1 * dim
    # diffAvgW = 1 * dim
    # loss = scalar

    # sigmoidDerivPosZ = sizePos * 1
    # sigmoidDerivPosW = sizePos * 1
    # sigmoidDerivNegZ = sizeNeg * 1
    # sigmoidDerivNegW = sizeNeg * 1
    # avgZDeriv = 1 * dim
    # avgWDeriv = 1 * dim


    posValMatrix = torch.matmul(posVecMatrix, Hyp_w)
    posValMatrix = posValMatrix + torch.matmul(torch.ones(posValMatrix.shape[0], 1, device=device), Hyp_b)

    negValMatrix = torch.matmul(negVecMatrix, Hyp_w)
    negValMatrix = negValMatrix + torch.matmul(torch.ones(negValMatrix.shape[0], 1, device=device), Hyp_b)

    numPosZ = torch.count_nonzero(torch.gt(posValMatrix, 0))
    numPosW = torch.count_nonzero(torch.gt(-posValMatrix, 0))

    numNegZ = torch.count_nonzero(torch.gt(negValMatrix, 0))
    numNegW = torch.count_nonzero(torch.gt(-negValMatrix, 0))

    # posValSignsZVecs = torch.sigmoid(torch.mul(posValMatrix, alpha))
    # posValSignsWVecs = torch.sigmoid(torch.mul(-posValMatrix, alpha))
    # negValSignsZVecs = torch.sigmoid(torch.mul(negValMatrix, alpha))
    # negValSignsWVecs = torch.sigmoid(torch.mul(-negValMatrix, alpha))

    posValSigmoidsZVecs = torch.sigmoid(torch.mul(posValMatrix, alpha))
    posValSigmoidsWVecs = torch.sigmoid(torch.mul(-posValMatrix, alpha))
    negValSigmoidsZVecs = torch.sigmoid(torch.mul(negValMatrix, alpha))
    negValSigmoidsWVecs = torch.sigmoid(torch.mul(-negValMatrix, alpha))

    posAvgZVec = posVecMatrix.T @ posValSigmoidsZVecs  / numPosZ
    posAvgWVec = posVecMatrix.T @ posValSigmoidsWVecs  / numPosW

    negAvgZVec = negVecMatrix.T @ negValSigmoidsZVecs  / numNegZ
    negAvgWVec = negVecMatrix.T @ negValSigmoidsWVecs / numNegW

    diffAvgZ = posAvgZVec - negAvgZVec
    diffAvgW = posAvgWVec - negAvgWVec

    # mseW = torch.nn.functional.mse_loss(posAvgWVec, negAvgWVec)
    # mseZ = torch.nn.functional.mse_loss(posAvgZVec, negAvgZVec)
    loss = torch.square(torch.linalg.norm(diffAvgW, ord="fro")) - torch.square(torch.linalg.norm(diffAvgZ, ord="fro"))

    # print("torch.abs(posAvgWVec-negAvgWVec).T.shape=", (torch.abs(posAvgWVec - negAvgWVec).T).shape)
    # print("posValSignsWVecs.shape=", (posValSignsWVecs).shape)
    # print("1-posValSignsWVecs.shape=", (1 - posValSignsWVecs).shape)
    # print("posValSignsWVecs=", posValSignsWVecs)
    # print("1-posValSignsWVecs=", 1 - posValSignsWVecs)
    # print("(1/numPosW*posValSignsWVecs*(1-posValSignsWVecs)).shape=",
    #       (1 / numPosW * posValSignsWVecs * (1 - posValSignsWVecs)).shape)

    sigmoidDerivPosZ = alpha / numPosZ * torch.mul(posValSigmoidsZVecs, (1 - posValSigmoidsZVecs))
    sigmoidDerivPosW = alpha / numPosW * torch.mul(posValSigmoidsWVecs, (1 - posValSigmoidsWVecs))
    sigmoidDerivNegZ = alpha / numNegZ * torch.mul(negValSigmoidsZVecs, (1 - negValSigmoidsZVecs))
    sigmoidDerivNegW = alpha / numNegW * torch.mul(negValSigmoidsWVecs, (1 - negValSigmoidsWVecs))
    # print("sigmoidDerivPosZ.shape", sigmoidDerivPosZ.shape)
    # print("sigmoidDerivPosW.shape", sigmoidDerivPosW.shape)
    # print("sigmoidDerivPosW.T@posVecMatrix.shape", (sigmoidDerivPosW.T @ posVecMatrix).shape)
    # print("sigmoidDerivNegW.T@posVecMatrix.shape", (sigmoidDerivNegW.T @ negVecMatrix).shape)

    avgZDeriv = torch.mul(torch.abs(diffAvgZ), 2)
    avgWDeriv = torch.mul(torch.abs(diffAvgW), 2)
    # print("avgZDeriv.shape", avgZDeriv.shape, "avgWDeriv.shape", avgWDeriv.shape)

    dLossbydB = (avgWDeriv @ negVecMatrix.T) @ sigmoidDerivNegW - (avgWDeriv @ posVecMatrix.T) @ sigmoidDerivPosW
    # dLossbydB2 = torch.matmul(avgWDeriv, (sigmoidDerivPosW.T @ posVecMatrix - sigmoidDerivNegW.T @ negVecMatrix).T * -1)
    # print("dLossbydB2=", dLossbydB2)
    # print("dLossbydB=", dLossbydB)

    dLossbydB = dLossbydB + \
                 (avgZDeriv @ negVecMatrix.T) @ sigmoidDerivNegZ - (avgZDeriv @ posVecMatrix.T) @ sigmoidDerivPosZ

    # dLossbydB2 = dLossbydB - \
    #             torch.matmul(avgZDeriv, (sigmoidDerivPosZ.T @ posVecMatrix - sigmoidDerivNegZ.T @ negVecMatrix).T)

    # print("dLossbydB2=", dLossbydB2)
    # print("dLossbydB=", dLossbydB)
    #
    # print("avgWDeriv @ negVecMatrix", (negVecMatrix @ avgWDeriv.T).shape)
    # print("sigmoidDerivNegW.shape", sigmoidDerivNegW.shape)
    # print("torch.mul(avgWDeriv @ negVecMatrix, sigmoidDerivNegW).shape",
    #       torch.mul(negVecMatrix @ avgWDeriv.T, sigmoidDerivNegW).shape)

    dLossbydTheta = negVecMatrix.T @ torch.mul(negVecMatrix @ avgWDeriv.T, sigmoidDerivNegW) - \
                    posVecMatrix.T @ torch.mul((posVecMatrix @ avgWDeriv.T), sigmoidDerivPosW)
    dLossbydTheta = dLossbydTheta + \
                    negVecMatrix.T @ torch.mul(negVecMatrix @ avgZDeriv.T,sigmoidDerivNegZ) - \
                    posVecMatrix.T @ torch.mul((posVecMatrix @ avgZDeriv.T), sigmoidDerivPosZ)

    # print("dLossbydTheta=", dLossbydTheta, "\ndLossbydTheta.shape", dLossbydTheta.shape)
    return loss, dLossbydTheta, dLossbydB
    # rmseW * (1 / numPosW * (posValSignsWVecs.T@) - 1 / numNegW * (1)) - 2 * rmseZ * (1 / numPosZ * (1) - 1 / numNegZ * (1))


if __name__ == "__main__":
    startTime = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(8)
    random.seed(8)
    print("device=", device)
    # filename = "datasets/combDev.csv"
    # datetimeStr = 'tensorBoardLogs/Dev_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
    filename = "datasets/combTrain.csv"
    datetimeStr = 'tensorBoardLogs/Train_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '/'

    writer = SummaryWriter(log_dir=datetimeStr)



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
    timeTaken = time.time() - startTime
    print("Time Taken breakpoint 1:", timeTaken)

    # Randomly initialize Hyperplane to 768D for size of bert embeddings
    # Hyp_w = torch.randn(768, 1, requires_grad=True, device=device)  # 768 rows and 1 columns for spacy
    # Hyp_b = torch.randn(1, 1, requires_grad=True, device=device)
    Hyp_w = torch.randn(768, 1, device=device)  # 768 rows and 1 columns for spacy
    Hyp_b = torch.randn(1, 1, device=device)

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
    # print("posVecMatrix.shape", posVecMatrix.shape)
    # print("negVecMatrix.shape", negVecMatrix.shape)

    timeTaken = time.time() - startTime
    print("Time Taken for preliminaries:", timeTaken)
    # alpha = torch.tensor([4], device=device)
    alpha = 4
    # numIters = 100000
    numIters = 20000
    stepSize = 0.01

    for iter in tqdm(range(numIters)):
        # timeLoopStart = time.time()
        with torch.no_grad():
            loss, dLossbydTheta, dLossbydB = computeLossGradients(Hyp_w, Hyp_b, posVecMatrix, negVecMatrix, alpha)
            if(iter%500==0):
                writer.add_scalar("Loss/train", loss, iter)
                # print("loss=",loss)

            Hyp_w = Hyp_w - dLossbydTheta * stepSize
            Hyp_b = Hyp_b - dLossbydB * stepSize
        del loss, dLossbydTheta, dLossbydB
        torch.cuda.empty_cache()
        # print("time for current loop:", time.time() - timeLoopStart)


        # posAvgZVec, posAvgWVec, negAvgZVec, negAvgWVec, posValMatrix, negValMatrix, \
        # posValSignsZVecs, posValSignsWVecs, negValSignsZVecs, negValSignsWVecs = \
        #     getAvgWZVecsForPosNeg(Hyp_w, Hyp_b, posVecMatrix, negVecMatrix, alpha)
        #
        # loss = torch.nn.functional.mse_loss(posAvgWVec, negAvgWVec) - \
        #        torch.nn.functional.mse_loss(posAvgZVec, negAvgZVec)
        # computeLossGradients(Hyp_w, Hyp_b, posVecMatrix, negVecMatrix, alpha)
        # Autogradient taking too long. So we will use manual gradient.
        # loss.backward(retain_graph=True)
        # stepSize = 1
        # print("time since loop start:%f\nNow starting gradient descent", time.time() - timeLoopStart)
        # with torch.no_grad():
        #     Hyp_w -= Hyp_w.grad * stepSize
        #     Hyp_b -= Hyp_b.grad * stepSize
        #     Hyp_w.grad.zero_()
        #     Hyp_b.grad.zero_()

    writer.close()
    print("Total Time taken for %d loops: %f" % (numIters, time.time() - startTime))
