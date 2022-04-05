# How To Use The Scripts



## Setup
 1. Install [Python 3](https://www.python.org/downloads/)
 2. Install necessary packages. 
    1. Run `pip install -r requirements.txt`. 
    2. Note you may need to use pip3 install (if you have python 2 and python3 installed on one system).
    3. You may also install requirements in a virtual environment like [virtualenv](https://realpython.com/python-virtual-environments-a-primer/) or [Anaconda](https://www.anaconda.com/).
 3. Download English Lemmatization Data Set
    1. Run `python -m spacy download en_core_web_sm`

## Configuring Stop Word and Mapping
Stop Words are editable in `data/stopwords.csv`. Append a new line with the word to remove,

To fix common typos, or merge related terms you can edit `data/mapping.csv`.  

## Running Scripts

### Term Frequencies
1. Make sure all your input documents are well formatted CSVs in ASCII or UTF8 Encoding
2. Run `python prepare.py FIILENAME_1.csv FIILENAME_2.csv FIILENAME_3.csv ...`
3. Term frequencies and wordclouds should now be in a `/summaries` Folder.

Order of Steps:
1. Tokenize
2. Remove Stop Words
3. Applying Mapping
4. Make Bigrams/ TriGrams
5. Lemmatize
6. Create Dictionary for Modeling

### LDA Topic Model
1. Make sure you have run `prepare.py`
2. Run `python lda.py`
3. Wait, This takes a while
4. Term frequencies and wordclouds should now be in a `/summaries` Folder.

### VADER Sentiment Analysis
1. Run `python vader.py FIILENAME_1.csv FIILENAME_2.csv FIILENAME_3.csv `
2. Summaries are generated into `/summaries` Folder

## Output Files

| File                 | Description                                                                                                                                                     |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| bigram_freq.csv      | Contains all bigrams with frequency of occurrence and [PMI Score](https://medium.com/dataseries/understanding-pointwise-mutual-information-in-nlp-e4ef75ecb57a) |
| dict.pickle          | The generated Dictionary for LDA Modeling                                                                                                                       |
| term_freq.csv        | Term occurance frequency (after removal of stopwords, maping, bigram formation, and lemmatization)                                                              |
| sentiment_all.csv    | Per-response sentiment scores                                                                                                                                   |
| sentiment_stats.csv  | Aggregate sentiment scores per file                                                                                                                             |
 | WordCloud_final.png  | Graphic: Word Cloud of term occurance (after removal of stopwords, maping, bigram formation, and lemmatization)                                                 |
 | WordCloud_nostop.png | Graphic: Word Cloud of term occurance (after removal of stopwords and maping)                                                                                   |
