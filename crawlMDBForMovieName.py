from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd, time


def get_page(url):
    """Get the text of the web page at the given URL
    return a string containing the content"""

    fd = urlopen(url)
    content = fd.read()
    fd.close()

    return content.decode('utf8')


def getMovieLink(reviewURL):
    html_doc = get_page(reviewURL)
    soup = BeautifulSoup(html_doc, 'html.parser')
    # print(soup.prettify())
    movieLink = soup.select('.lister-item-header a')[0].get("href")
    baseLink = 'https://www.imdb.com'
    return baseLink + movieLink


# reviewURL ='https://www.imdb.com/review/rw2327720/'
# movLink = getMovieLink(reviewURL)
# print(movLink)

if __name__ == "__main__":
    trainingData = pd.read_csv('outputs/trainingData.csv', sep='|')
    trainingData['imdbMovieLink'] = ""
    reviewLinks = trainingData['imdbReviewLinks'].to_list()
    # print("reviews=", reviews)
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    headers = {'User-Agent': user_agent}

    for reviewNum in range(len(reviewLinks)):
        reviewURL = reviewLinks[reviewNum]
        if reviewURL == '':
            pass
        else:
            query = reviewURL
            # query = 'site:imdb.com ' + review
            print("query=", query)
            try:
                movieLink = getMovieLink(query)
                # link = next(search(query, tld="com", num=1, user_agent=user_agent))
            except:
                # print("No link for query = ",query)
                movieLink = ''
            print("link=", movieLink)
            trainingData.loc[reviewNum, 'imdbMovieLink'] = movieLink
            print("trainingData.loc[reviewNum, 'imdbMovieLink']=", trainingData.loc[reviewNum, 'imdbMovieLink'])
            time.sleep(1)

    print(trainingData)
    trainingData.to_csv('outputs/trainingDataWithMovieLinks.csv', sep='|')
