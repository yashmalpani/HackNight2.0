import tweepy
from textblob import TextBlob
from bs4 import BeautifulSoup
import requests
import textblob

news_url = 'https://news.google.com/search?q=microsoft&hl=en-IN&gl=IN&ceid=IN%3Aen'

def data():
    page = requests.get(news_url)
    soup = BeautifulSoup(page.content, 'lxml')
    out = soup.find_all('h3', class_ = 'ipQwMb ekueJc RD0gLb')
    news = []

    for i in out:
        news.append(i.string)
    considerations = 100
    unit = 0
    news = news[0:considerations]
    for new in news:
        blob = textblob.TextBlob(new)
        subjectivity=blob.sentiment.subjectivity
        #print(subjectivity)
        polarity=blob.sentiment.polarity
        unit+= (polarity*subjectivity)
    unit /= considerations
    unit = float(str(unit)[:5])
    return unit


CONSUMER_KEY = "p8R1UPTG7oF1Eq6w4NGO8Zcdp"
CONSUMER_SECRET = "SDcP5DuL7n7FZBCMyAtr8mDnyveXcGAAHqrEda66nFwL7wN4h6"

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
api = tweepy.API(auth)
tweet_search_user = 100
tweet_search_head = 50
def tweets():
    unit1, unit2 = 0,0
    tweets = tweepy.Cursor(api.search, q='Microsoft').items(tweet_search_user)
    for tweet in tweets:
        analysis = TextBlob(tweet.text).sentiment
        subjectivity = analysis.subjectivity
        polarity = analysis.polarity
        unit1 += (polarity * subjectivity)
    tweets = tweepy.Cursor(api.search, q='@satyanadella').items(tweet_search_head)
    for tweet in tweets:
        analysis = TextBlob(tweet.text).sentiment
        subjectivity = analysis.subjectivity
        polarity = analysis.polarity
        unit2 += (polarity * subjectivity)
    unit = unit1/tweet_search_user+unit2/tweet_search_head
    unit=float(str(unit)[:5])
    return unit
