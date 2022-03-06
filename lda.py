import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def buildModel(corpus, id2word, num_topics ):
    return gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=1000,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)



def find_optimal_k(tokens,id2word,corpus):
    min_topics = 2
    max_topics = 25

    print(f"Searching for optimal number of topics")
    model_list = []
    for k in range(min_topics, max_topics, 1):
        print("\nRunning K=", k)
        model = buildModel(corpus, id2word, k)
        coherence_model = gensim.models.CoherenceModel(model=model, texts=tokens, dictionary=id2word, coherence='c_v')

        for i in range(0, model.num_topics):
            topic = [token for token, score in model.show_topic(i, topn=10)]
            print(topic)
            model_list.append([k, coherence_model.get_coherence(), model.log_perplexity(corpus), topic])

    topicDf = pd.DataFrame(model_list, columns=["Num_Topics", "Coherence", "Perplexity", "Terms"])
    topicDf.to_csv("summaries/Topics_by_k.csv", index=False)

    dx = topicDf.groupby("Num_Topics").first()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    ax2.set_ylim(-5, -20)
    sns.lineplot(ax=ax, x="Num_Topics", y="Coherence", data=dx, label="Coherence")
    sns.lineplot(ax=ax2, x="Num_Topics", y="Perplexity", data=dx, color="orange", label="Perplexity")

    ax.set(ylabel="Coherence score", xlabel="Num Topics")

    fig.tight_layout()
    plt.savefig(f"summaries/Coherence and Perplexity.png", bbox_inches='tight')
    plt.show()
    print("Done")

# Setup Graphs
sns.set(rc={'figure.figsize':(9,5)}, font="Calibri", font_scale = 1.2)
sns.set_style("whitegrid", {'axes.grid' : False})



#%% Load Data
# Load data (deserialize)
with open('summaries/dict.pickle', 'rb') as handle:
    loaded_data = pickle.load(handle)
(tokens,id2word,corpus) = loaded_data

#%% Find optimal num topics
if __name__ == '__main__':
    find_optimal_k(*loaded_data)