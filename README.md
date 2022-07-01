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

### Running LDA + VADER (Sentiment Analysis)
1. Make sure  your input document is well formatted CSVs in ASCII or UTF8 Encoding
2. File expected to have a header: `id, question, response`'`
3. Run `python prepare.py example.csv`
4. Term frequencies and wordclouds should now be in a `/summaries` Folder.

Order of Steps:
1. Tokenize
2. Remove Stop Words
3. Applying Mapping
4. Run VADER
5. Make Bigrams/ TriGrams
6. Lemmatize 
7. Remove Stop Words (Again)
8. Re-Applying Mapping (Again)
9. Create Dictionary for LDA Modeling

### LDA Topic Model
1. Make sure you have run `prepare.py`
2. Run `python lda_find_k.py`
3. Wait, This takes a while
4. Topics by k-value (number of total topics) are now in `/summaries` Folder.



## Output Files

Files in the `DO_NOT_SHARE` directory contain full responses from participants. Do not share these with anyone not on IRB.

Files in `summaries` folder should be safe to share. All private data including response text is stripped.

 There are three prefixes
* raw: based on raw tokens from responses
* clean: raw with stopwords removed, mapping applied, and lemmatized
* no_syn: clean with synonyms merged

| File                                  | Description                                                                                    |
|---------------------------------------|------------------------------------------------------------------------------------------------|
| dataset_sanitized.csv                 | Contains stats about individual responses, with sensative data (the response content stripped) |
| synonym_map.csv                       | Generated mapping of words to synonyms                                                         |
| {prefix}_model.pickle                 | LDA model, can be reused to tag public data.                                                   |
| {prefix}_Coherence_and_Perplexity.png | Graph of LDA topic coherence vs perplexity by k (number of topics                              |
| {prefix}_term_freq.csv                | Term occurance frequency                                                                       |
| {prefix}_WordCloud.png                | Graphic: Word Cloud of term occurance                                                          |
