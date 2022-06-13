import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
import csv
import warnings
from wordcloud import WordCloud
from nltk.stem.porter import PorterStemmer

warnings.filterwarnings('ignore')

def visialization_most_used_words(df):
    all_words = " ".join([sentence for sentence in df['clean_tweet']])

    wordcloud = WordCloud(width = 800, height = 500, random_state = 42, max_font_size = 100).generate(all_words)

    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def visialization_positive_words(df):
    all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label']==1]])

    wordcloud = WordCloud(width = 800, height = 500, random_state = 42, max_font_size = 100).generate(all_words)

    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def visialization_negative_words(df):
    all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label']==1]])

    wordcloud = WordCloud(width = 800, height = 500, random_state = 42, max_font_size = 100).generate(all_words)

    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


df = pd.read_csv('tweets.csv', sep=";", on_bad_lines='skip', encoding = "ISO-8859-1")

data = {'label':df['ï»¿label'], 'tweet':df['tweet']}
df = pd.DataFrame(data)

df['tweet']= df['tweet'].astype(str)

def remove_pattern(input_text, pattern):
    r = re.findall(pattern, input_text)
    for word in r:
        input_text = re.sub(word, "", input_text)
    return input_text

#remove @user handles
df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
#remove special characters
df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
#remove short words
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))

#changing label of positive word to 1 for f1_score usage
df['label'] = df['label'].apply(lambda x: 1 if x == 4 else 0)

#splitting every word in tweet
tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())

#stemming
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])

#combining every word from stemming into one list
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])

df['clean_tweet'] = tokenized_tweet

def hashtag_extract(tweets):
    hashtags = []
    for tweet in tweets:
        ht = re.findall(r"#(\w+)", tweet)
        hashtags.append(ht)
    return hashtags

#listy pozytywnych i negatywnych hashtagow
ht_positive = hashtag_extract(df['clean_tweet'][df['label']==4])
ht_negative = hashtag_extract(df['clean_tweet'][df['label']==0])
#zrobienie jednej listy
ht_positive = sum(ht_positive, [])
ht_negative = sum(ht_negative, [])

freq = nltk.FreqDist(ht_negative)
d = pd.DataFrame({'Hashtag': list(freq.keys()),
                  'Count': list(freq.values())})
#select top 1 hashtags
d = d.nlargest(columns='Count', n = 10)
#plt.figure(figsize=(15,9))
#sns.barplot(data=d, x='Hashtag', y='Count')
#plt.show()

#input split
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df = 0.90, min_df = 2, max_features = 1000, stop_words='english')
bow = bow_vectorizer.fit_transform(df['clean_tweet'])

#split input for trainign and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(bow, df['label'], random_state = 42, test_size = 0.25)


#MODEL TRAINING
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

#training
model = LogisticRegression()
model.fit(x_train, y_train)

#testing
pred = model.predict(x_test)
print(f"F1 score: {f1_score(y_test, pred)}")
print(f"Precycja: {accuracy_score(y_test, pred)}")