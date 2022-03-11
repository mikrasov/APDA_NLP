import pandas as pd
import gensim
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from wordcloud import WordCloud
import sys
import os

# Setup Graphs
sns.set(rc={'figure.figsize':(9,5)}, font="Calibri", font_scale = 1.2)
sns.set_style("whitegrid", {'axes.grid' : False})

os.makedirs("summaries/", exist_ok=True)

# Setup stopwords
stopwords = pd.read_csv("data/stopwords.csv", names=["word"]).word.to_list()

# Setup lemmatization
try:
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
except Exception:
    print("ERROR: Spacy can't find 'en_core_web_sm', run 'python -m spacy download en_core_web_sm'")
    exit()

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
if len(sys.argv) < 2 :
    print("ERROR: Too few arguments passed in\n")
    print(f"Usage {sys.argv[0]} Filename1.csv Filename2.csv]\n")
    exit()

datasets = []
for arg in sys.argv[1:]:
    print(f"Parsing  {arg}")
    df = pd.read_csv(arg, header=0, names=["response"])
    df["source"] = arg
    datasets.append(df)


df = pd.concat(datasets)
data = df.response.values.tolist()
data = [r.replace("true##","") for r in data]
print("Data Loaded")

#%% Prepare Tokens
# Tokenize
print("Tokenizing")
tokens = [gensim.utils.simple_preprocess(str(r), deacc=True) for r in data]

#Remove Stop words
print("Remove Stop Words")
tokens = [[word for word in gensim.utils.simple_preprocess(str(response)) if word not in stopwords] for response in tokens]

#Map Words
print("Applying Mapping")
map = pd.read_csv("data/mapping.csv", index_col="From")["To"].to_dict()
tokens = [[map[word] if word in map else word for word in response] for response in tokens]

# Save Raw Term Frequency
print("Saving RAW Term Frequency")
id2word = gensim.corpora.Dictionary(tokens)
corpus = [id2word.doc2bow(response) for response in tokens]
termDfs = [(pd.DataFrame([(id2word[id], freq) for id, freq in c], columns=["term","freq"])) for c in corpus]
terms = pd.concat(termDfs).groupby("term").sum().to_csv("summaries/term_raw_freq.csv")
plotWordCloud(tokens, "summaries/WordCloud_nostop.png")



#Make Bigrams
print("Make Bigrams/ TriGrams")
bigram = gensim.models.Phrases(tokens, min_count=1, threshold=0.01,  connector_words=gensim.models.phrases.ENGLISH_CONNECTOR_WORDS)
dl = pd.DataFrame(bigram.export_phrases().items(), columns=["Bigram","PMI Score"])
dr = pd.DataFrame(bigram.vocab.items(), columns=["Bigram","Frequency"])
dl.merge(dr, on="Bigram").sort_values("Frequency",ascending=False).to_csv("summaries/bigram_freq.csv", index=False)

bigram_mod = gensim.models.phrases.Phraser(bigram)

tokens = [bigram_mod[r] for r in tokens]


# Lemmatization
print("Lemmatize")
tokens = [([token.lemma_ for token in nlp(" ".join(response)) if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']]) for response in tokens]
plotWordCloud(tokens, "summaries/WordCloud_final.png")



# Create Dictionary
print("Create Dictionary for Modeling")
id2word = gensim.corpora.Dictionary(tokens)
corpus = [id2word.doc2bow(response) for response in tokens]

with open('summaries/dict.pickle', 'wb') as handle:
    pickle.dump((tokens,id2word,corpus), handle, protocol=pickle.HIGHEST_PROTOCOL)




# Save Term Frequency
print("Saving Term Frequency")
termDfs = [(pd.DataFrame([(id2word[id], freq) for id, freq in c], columns=["term","freq"])) for c in corpus]
terms = pd.concat(termDfs).groupby("term").sum().to_csv("summaries/term_freq.csv")


print("Done")