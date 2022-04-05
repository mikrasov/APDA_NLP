#%%%% Setup project
import nltk
import ssl
import pandas as pd
import sys
import gensim
import os

#Workaround for failing SSL cert
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download("vader_lexicon")

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent_analyzer = SentimentIntensityAnalyzer()


os.makedirs("summaries/", exist_ok=True)

def sentiment(row):
  sentiment = sent_analyzer.polarity_scores(row["response"])
  row["neg"] = sentiment["neg"]
  row["neu"] = sentiment["neu"]
  row["pos"] = sentiment["pos"]
  row["compound"] = sentiment["compound"]

  if(sentiment['compound']>= 0.05):
    row["overall"] = "positive"
    row["is_positive"] = 1

  elif(sentiment['compound']<= -0.05):
    row["overall"] = "negative"
    row["is_negative"] = 1

  else:
    row["overall"] = "neutral"
    row["is_neutral"] = 1

  return row


#%% Import Dataset
if len(sys.argv) < 2 :
    print("ERROR: Too few arguments passed in\n")
    print(f"Usage {sys.argv[0]} Filename1.csv Filename2.csv]\n")
    exit()

sys.argv = [
  "vader.py",
  "./data/PublicSurveyResponsesAPDA_4b.csv",
  "./data/PublicSurveyResponsesAPDA_6b.csv",
  "./data/PublicSurveyResponsesAPDA_7b.csv",
  "./data/PublicSurveyResponsesAPDA_8.csv",
]

datasets = []
for arg in sys.argv[1:]:
    print(f"Parsing  {arg}")
    df = pd.read_csv(arg, header=0, names=["response"])
    df["source"] = arg
    datasets.append(df)


data = pd.concat(datasets)
data["response"] = data["response"].str.replace("true##","")
data["original"] = data["response"]

stopwords = pd.read_csv("data/stopwords.csv", names=["word"]).word.to_list()
mapping = pd.read_csv("data/mapping.csv", names=["word"]).word.to_list()

def remove_stopwords(text):
  # Tokenize
  tokens = gensim.utils.simple_preprocess(str(text), deacc=True)

  # Remove Stop words
  tokens = [word for word in gensim.utils.simple_preprocess(str(tokens)) if word not in stopwords]

  # Map Words
  map = pd.read_csv("data/mapping.csv", index_col="From")["To"].to_dict()
  tokens = [map[word] if word in map else word for word in tokens]
  return ' '.join(tokens)

print("Data Loaded")

#%%% Apply Sentiment Analysis

data["response"] = data["response"].apply(remove_stopwords)
data = data.apply(sentiment, axis=1)
data.to_csv("summaries/sentiment_all.csv", index=False)

#%%% Aggregated stats
stats= data.groupby("source").agg({
  'neg' : 'mean',
  'pos' : 'mean',
  'neu' : 'mean',
  'is_negative' : 'sum',
  'is_positive' : 'sum',
  'is_neutral' : 'sum',
  'response': 'count'
})
stats["per_negative"] = stats["is_negative"]/stats["response"]
stats["per_positive"] = stats["is_positive"]/stats["response"]
stats["per_neutral"] = stats["is_neutral"]/stats["response"]

stats.to_csv("summaries/sentiment_stats.csv")
