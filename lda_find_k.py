import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd

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



def find_optimal_k(corpus,id2word,tokens, file_prefix):
    min_topics = 2
    max_topics = 25
    step = 2

    print(f"Searching for optimal number of topics for '{file_prefix}'")
    print("Try to maximize Coherence and minimize Perplexity")
    model_list = []
    for k in range(min_topics, max_topics, step):
        print(f"\nRunning K={k} for '{file_prefix}'")
        model = buildModel(corpus, id2word, k)

        perplexity= model.log_perplexity(corpus)
        print(f"\tPerplexity:\t\t\t{perplexity:1.2f}")

        coherence_model_c_v = gensim.models.CoherenceModel(model=model, texts=tokens, dictionary=id2word, coherence='c_v')
        coherence_c_v = coherence_model_c_v.get_coherence()
        print(f"\tCoherence (c_v):\t{coherence_c_v:1.2f}")

        for i in range(0, model.num_topics):
            topic = [token for token, score in model.show_topic(i, topn=10)]
            print(f"\t{i:<2}: {topic}")
            model_list.append([k, coherence_c_v,  perplexity, i, topic ])

    topicDf = pd.DataFrame(model_list, columns=["Num_Topics", "Coherence", "Perplexity", "Topic Number", "Terms"])
    topicDf.to_csv(f"summaries/{file_prefix}_Topics_by_k.csv", index=False, float_format="%.3f")

    dx = topicDf.groupby("Num_Topics").first()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    ax2.set_ylim(-5, -20)
    sns.lineplot(ax=ax, x="Num_Topics", y="Coherence", data=dx, label="Coherence")
    sns.lineplot(ax=ax2, x="Num_Topics", y="Perplexity", data=dx, color="orange", label="Perplexity")

    ax.set(ylabel="Coherence score", xlabel="Num Topics")

    fig.tight_layout()
    plt.savefig(f"summaries/{file_prefix}_Coherence_and_Perplexity.png", bbox_inches='tight')
    plt.show()
    print("Done")


#%% Find optimal num topics
if __name__ == '__main__':

    # Setup Graphs
    sns.set(rc={'figure.figsize': (9, 5)}, font="Calibri", font_scale=1.2)
    sns.set_style("whitegrid", {'axes.grid': False})


    CORPUSES = [
        ("clean", "lda_tokens"),
        # ("raw", "raw_tokens"),
        #  ("no_syn", "syn_tokens"),
    ]

    for name, token_col in CORPUSES:
        CORPUS_PATH = f"summaries/{name}_DO_NOT_SHARE.pickle"


        with open(CORPUS_PATH, 'rb') as handle:
            loaded_data = pickle.load(handle)
        (tokens, id2word, corpus) = loaded_data

        find_optimal_k(*loaded_data, name)