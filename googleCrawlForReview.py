import pandas as pd
from googlesearch import search
import time
from googleapiclient.discovery import build

'''
<script async src="https://cse.google.com/cse.js?cx=72e79523179864001">
</script>
<div class="gcse-search"></div>
api_key= 'AIzaSyD7DnoqSfq_4DWAPh639UBAP4C7WtfJRgo'
my_cse_id = "72e79523179864001"

'''




my_api_key= 'AIzaSyD7DnoqSfq_4DWAPh639UBAP4C7WtfJRgo' #The API_KEY you acquired
my_cse_id = '72e79523179864001' #The search-engine-ID you created

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']

def getResultLink(query):
    results = google_search(query, my_api_key, my_cse_id, num=1)
    result = results[0]
    return result['link']



if __name__=="__main__":


    
    trainingData = pd.read_csv("datasets/origTrain.csv", sep="\t")
    trainingData['imdbReviewLinks'] = ""
    reviews = trainingData['Text'].to_list()
    # print("reviews=", reviews)
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    headers = {'User-Agent': user_agent}

    for reviewNum in range(len(reviews)):
        review = reviews[reviewNum]
        query=review
        # query = 'site:imdb.com ' + review
        # print("query=",query)
        try:
            link = getResultLink(query=query)
            # link = next(search(query, tld="com", num=1, user_agent=user_agent))
        except:
            # print("No link for query = ",query)
            link=''
        # print("link=", link)
        trainingData.loc[reviewNum, 'imdbReviewLinks'] = link
        # print("trainingData.loc[reviewNum, 'imdbReviewLinks']=",trainingData.loc[reviewNum, 'imdbReviewLinks'])
        time.sleep(1)


    print(trainingData)
    trainingData.to_csv('outputs/trainingData.csv',sep='|')

