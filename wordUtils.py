
# from transformers import RobertaTokenizer

import time
import spacy
nlp = spacy.load('en_core_web_md')
from nltk.tokenize import sent_tokenize, word_tokenize

from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Strip, Lowercase, BertNormalizer
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import DistilBertTokenizerFast, DistilBertModel
import torch

def getSpacyVector(token):
    return nlp(token).vector

def getGloveVector(token):
    return nlp(token).vector

def getTokenTransformerEmbedding(token,model,tokenizer,device="cpu"):
    # print("token=",token)
    encodingOfToken = tokenizer.encode(token, return_tensors='pt').to(device)
    embedding = model.embeddings.word_embeddings(encodingOfToken)
    # print("model.embeddings.word_embeddings(token)=",embedding)
    return embedding

# We normalize the input text into some standard form, with lowercase, no accents and whitespace removed
# https://huggingface.co/docs/tokenizers/api/normalizers
def normalizeSentence(inputText, normalizerSequence):
    normalizer = normalizers.Sequence(normalizerSequence)
    normalizedInput = normalizer.normalize_str(inputText)
    return normalizedInput

def getListOfWordsFromSentence(sentence):
    words = []
    for word in sentence.split():
        words.append(word)
    return words


def getListOfWordsFromSentences(sentences):
    largeSentence = " ".join(sentences)
    words = []
    for word in largeSentence.split():
        words.append(word)
    return words


if __name__ == "__main__":
    startTime = time.time()
    # https://huggingface.co/docs/tokenizers/api/normalizers
    normalizerSequence = [NFD(), StripAccents(), Strip(), Lowercase()]
    original = " Héllò hôw are ü? 😃 is a smiley "
    normalizedSentence = normalizeSentence(original, normalizerSequence)
    print("normalizedSentence=", normalizedSentence)

    str = "hello how are you?"
    normalStr = normalizeSentence(str, normalizerSequence)
    words = word_tokenize(normalStr)
    print(words)
    for word in words:
        print("word=",word,"getVector(word)=",getSpacyVector(word))

    print("now bert tokenizer")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    lastEnc = torch.tensor([0])
    for word in words:
        encoding = getTokenTransformerEmbedding(word, model, tokenizer)
        print("word=", word,"encoding=",encoding,"lastEnc==encoding?",torch.equal(lastEnc, encoding))
        lastEnc = encoding


    timeTaken = time.time() - startTime
    print("Time Taken:", timeTaken)

