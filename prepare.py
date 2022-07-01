import spacy
import sys
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import collections
from nltk.corpus import wordnet
import gensim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from wordcloud import WordCloud

os.makedirs("summaries/", exist_ok=True)

# Setup lemmatization
try:
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
except Exception:
    print("ERROR: Spacy can't find 'en_core_web_sm', run 'python -m spacy download en_core_web_sm'")
    exit(1)

# Setup stopwords
stopwords = pd.read_csv("data/stopwords.csv", names=["word"]).word.to_list()


# Setup VADER
nltk.download("vader_lexicon")
nltk.download('wordnet')
nltk.download('omw-1.4')
sent_analyzer = SentimentIntensityAnalyzer()

#Setup Graphs
sns.set(rc={'figure.figsize':(9,5)}, font="Calibri", font_scale = 1.2)
sns.set_style("whitegrid", {'axes.grid' : False})

def plotWordCloud(tokens, filename):
    wordcloud_text = ""
    for r in tokens:
        wordcloud_text += " ".join(r) + " "

    wordcloud = WordCloud(width = 1024, height = 768, regexp=None, background_color="white").generate(wordcloud_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(filename, bbox_inches='tight')
    #plt.show()



#%% Import Dataset

dataset = None
try:
    print(f"Parsing file '{sys.argv[1]}'")
    dataset = pd.read_csv(sys.argv[1], low_memory=False)

except Exception as e:
    print(e)

    print(f"Usage {sys.argv[0]} Public_and_Private_Responses.csv]")

    print("\nExpecting a single file path as an argument. The file should be a csv with 4 columns and a header (id, question, response).\n")
    print("Example:")
    with open('data/example.csv', 'r') as f:
        for line in f:
            print("|",line, end="")
    print("\n")
    sys.exit(1)


dataset["response"] = dataset["response"].str.replace("true##","")
print("Data Loaded")

#%% Prepare Tokens
# Tokenize
print("Tokenizing")
dataset["raw_tokens"] = dataset["response"].apply(lambda r: gensim.utils.simple_preprocess(str(r), deacc=True))


print("Removing Stop Words")
dataset["tokens"] = dataset["raw_tokens"].apply(lambda r: [word for word in gensim.utils.simple_preprocess(str(r)) if word not in stopwords])


print("Applying Mapping")
mapping = pd.read_csv("data/mapping.csv", index_col="From")["To"].to_dict()
dataset["tokens"] = dataset["tokens"].apply(lambda r: [mapping[word] if word in mapping else word for word in r])


#%%% Starting Sentiment Analysis

def sentiment(row):
  sentiment = sent_analyzer.polarity_scores(row["response"])
  row["sent_neg"] = sentiment["neg"]
  row["sent_neu"] = sentiment["neu"]
  row["sent_pos"] = sentiment["pos"]
  row["sent_compound"] = sentiment["compound"]

  if(sentiment['compound']>= 0.05):
    row["sent_overall"] = "positive"
    row["sent_is_positive"] = 1

  elif(sentiment['compound']<= -0.05):
    row["sent_overall"] = "negative"
    row["sent_is_negative"] = 1

  else:
    row["sent_overall"] = "neutral"
    row["sent_is_neutral"] = 1

  return row


print("Running Sentiment Analysis (VADAR) on Responses-Stopwords")
dataset = dataset.apply(sentiment, axis=1)

#%%% Form Bigrams
print("Make Bigrams/ TriGrams")
bigram = gensim.models.Phrases(dataset["tokens"], min_count=1, threshold=0.01,  connector_words=gensim.models.phrases.ENGLISH_CONNECTOR_WORDS)
bigram_mod = gensim.models.phrases.Phraser(bigram)
dataset["lda_tokens"] = dataset["tokens"].apply(lambda r: bigram_mod[r])




#%%% Lemmatize
print("Preforming Lemmatization (for LDA)")

dataset["lda_tokens"] = dataset["lda_tokens"].apply(lambda r: [token.lemma_ for token in nlp(" ".join(r)) if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']])


print("Removing Stop Words (again after lemmatization)")
dataset["lda_tokens"] = dataset["lda_tokens"].apply(lambda r: [word for word in gensim.utils.simple_preprocess(str(r)) if word not in stopwords])


print("Re-Applying Mapping (after lemmatization)")
mapping = pd.read_csv("data/mapping.csv", index_col="From")["To"].to_dict()
dataset["lda_tokens"] = dataset["lda_tokens"].apply(lambda r: [mapping[word] if word in mapping else word for word in r])



#%%% Synonym
print("Condensing Synonyms")

def generate_synonym_map(tokens):
    uniqueWords = []
    for index, value in tokens.items():
        uniqueWords += value

    uniqueWords_count = collections.Counter(uniqueWords)

    syn_map = {}

    REPLACEMENT_THRESHOLD=100000
    for w,c in uniqueWords_count.most_common():
        synonyms= []
        for syn in wordnet.synsets(w):
            for l in syn.lemmas():
                if not w == l.name():
                    synonyms.append(l.name())

        foundInMap = False
        for s in set(synonyms):
            if c< REPLACEMENT_THRESHOLD and s in syn_map:
                syn_map[w] = syn_map[s]
                foundInMap = True
                break
        if not foundInMap:
            syn_map[w] = w

    return syn_map

syn_map = generate_synonym_map(dataset["lda_tokens"])

with open('summaries/synonym_map.csv', 'w') as f:
    for key, val in syn_map.items():
        f.write(key+", "+val+"\n")

dataset["syn_tokens"] = dataset["lda_tokens"].apply(lambda r: [syn_map[t] for t in r])

#%% Create Dictionary for LDA
CORPUSES = [
    ("raw", "raw_tokens"),
    ("clean", "lda_tokens"),
    ("no_syn","syn_tokens"),
]

for name, token_col in CORPUSES:
    CORPUS_PATH = f"summaries/{name}_DO_NOT_SHARE.pickle"
    TERM_PATH = f"summaries/{name}_term_freq.csv"
    WORDCLOUD_PATH = f"summaries/{name}_WordCloud.png"
    print(f"Saving '{name}' Corpus for LDA Modeling to '{CORPUS_PATH}'")
    tokens = dataset[token_col].to_list()
    id2word = gensim.corpora.Dictionary(tokens)
    corpus = [id2word.doc2bow(response) for response in tokens]

    with open(CORPUS_PATH, 'wb') as handle:
        pickle.dump((corpus,id2word,tokens), handle, protocol=pickle.HIGHEST_PROTOCOL)


    print(f"Saving '{name}' Term Frequency to '{TERM_PATH}'")
    termDfs = [(pd.DataFrame([(id2word[id], freq) for id, freq in c], columns=["term","freq"])) for c in corpus]
    terms = pd.concat(termDfs).groupby("term").sum().to_csv(TERM_PATH)

    print(f"Saving '{name}' Wordcloud image to '{WORDCLOUD_PATH}'")
    plotWordCloud(dataset[token_col], WORDCLOUD_PATH)

#%% Sanitized file
dataset.drop(columns=["response","raw_tokens", "tokens", "lda_tokens", "syn_tokens"]).to_csv("summaries/dataset_sanitized.csv")

