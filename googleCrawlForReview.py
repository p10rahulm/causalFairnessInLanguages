import pandas as pd
from googlesearch import search

query = '"Long, boring, blasphemous. Never have I been so glad to see ending credits roll."' + '+ imdb'

for res in search(query, tld="com", num=1, stop=1, pause=2):
    print(res)

trainingData = pd.read_csv("datasets/origTrain.csv", sep="\t")
trainingData['imdbLink'] = ""
reviews = trainingData['Text'].to_list()
# print("reviews=", reviews)

for reviewNum in range(len(reviews)):
    review = reviews[reviewNum]
    query = 'site:imdb.com ' + '"' + review + '"'
    print("query=",query)
    try:
        link = next(search(query, tld="com", num=1))
    except:
        print("No link for query = ",query)
        link=''
    print("link=", link)
    trainingData.loc[reviewNum, 'imdbLink'] = link
    print("trainingData.loc[reviewNum, 'imdbLink']=",trainingData.loc[reviewNum, 'imdbLink'])


print(trainingData)

