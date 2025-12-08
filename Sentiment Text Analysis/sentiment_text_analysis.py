from textblob import TextBlob
from newspaper import Article
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://en.wikipedia.org/wiki/Mathematics'
article = Article(url)

article.download()
article.parse()
article.nlp()

text = article.text
print(text)


