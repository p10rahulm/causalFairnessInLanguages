from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import Request
import pandas as pd, time
import re


def get_pageWithUA(url,user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'):
    """Get the text of the web page at the given URL
    return a string containing the content"""
    req = Request(
        url,
        data=None,
        headers={
            'User-Agent': user_agent
        }
    )
    fd = urlopen(req)
    content = fd.read()
    fd.close()

    return content.decode('utf8')


def getMovieGenre(reviewURL):
    html_doc = get_pageWithUA(reviewURL)
    soup = BeautifulSoup(html_doc, 'html.parser')
    # print(soup.prettify())
    listOfScripts = soup.find_all('script')
    rightScript=''
    for scriptElem in listOfScripts:
        if "genre" in str(scriptElem):
            rightScript=str(scriptElem)
            genresSearch = re.search('"genre":\[(.*?)\]', rightScript, re.IGNORECASE)
            if genresSearch:
                genres = genresSearch.group(1)
                return genres

    return ''


# reviewURL ='https://www.imdb.com/review/rw2327720/'
# movLink = getMovieLink(reviewURL)
# print(movLink)

if __name__ == "__main__":
    trainingData = pd.read_csv('outputs/trainingDataWithMovieLinks.csv', sep='|')
    trainingData['imdbGenres'] = ""
    movieLinks = trainingData['imdbMovieLink'].to_list()
    # print("reviews=", reviews)
    # user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    # headers = {'User-Agent': user_agent}

    for movieNum in range(len(movieLinks)):
        movieURL = str(movieLinks[movieNum])
        if ('http' in movieURL):
            query = movieURL
            # query = 'site:imdb.com ' + review
            print("query=", query)
            try:
                movieGenres = getMovieGenre(query)
                # link = next(search(query, tld="com", num=1, user_agent=user_agent))
            except:
                # print("No link for query = ",query)
                movieGenres = ''
            print("genres=", movieGenres)
            trainingData.loc[movieNum, 'imdbGenres'] = movieGenres
            print("trainingData.loc[movieNum, 'imdbGenres']=", trainingData.loc[movieNum, 'imdbGenres'])
            time.sleep(1)



    print(trainingData)
    trainingData.to_csv('outputs/trainingDataWithGenres.csv', sep='|')
