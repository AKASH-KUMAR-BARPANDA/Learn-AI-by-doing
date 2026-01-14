from newspaper import Article
from textblob import  TextBlob


url = input("enter url here : ")
article = Article(url)

article.download()
article.parse()
article.nlp()

title = article.title
published_date = article.publish_date
authors = article.authors
summery = article.summary

analysis = TextBlob(article.text)
print(f'Title : {title}')
print(f'Published : {published_date}')
print(f'Authors : {authors} \n')
print(f'Summery: {summery}')

if analysis.polarity > 1:
    print("positive")
elif analysis.polarity < -1:
    print('Negative')
else:
    print('Neutral')