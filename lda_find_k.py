import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import numpy as np

CORPUSES = [
    ("clean", "lda_tokens"),
    # ("raw", "raw_tokens"),
    #  ("no_syn", "syn_tokens"),
]



def buildModel(corpus, id2word, num_topics ):
    return gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=1000,
                                                update_every=1,
                                                chunksize=100000,
                                                passes=20,
                                                alpha='auto',
                                                per_word_topics=True)



def find_optimal_k(corpus,id2word,tokens, file_prefix):
    min_topics = 2
    max_topics = 50
    step = 1

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

    if len(model_list)>1:
        print("Calculating optimal perplexity and coherence")
        topicDf["Coherence_norm"] = (topicDf.Coherence - topicDf.Coherence.min()) / (topicDf.Coherence.max() - topicDf.Coherence.min())
        topicDf["Perplexity_norm"] = 1- ((topicDf.Perplexity - topicDf.Perplexity.min()) / (topicDf.Perplexity.max() - topicDf.Perplexity.min()))

        topicDf["Distance"] = topicDf.apply(lambda r: np.sqrt(2*((r.Coherence_norm - topicDf.Coherence_norm.max()) ** 2) + (r.Perplexity_norm - topicDf.Perplexity_norm.max()) ** 2), axis=1)
        BEST_K = topicDf[topicDf.Distance == topicDf.Distance.min()].iloc[0]
        print(f"\n Best Num Topics {BEST_K.Num_Topics}")

    topicDf.to_csv(f"summaries/{file_prefix}_Topics_by_k.csv", index=False, float_format="%.3f")



#%% Find optimal num topics
if __name__ == '__main__':


    for name, token_col in CORPUSES:
        CORPUS_PATH = f"DO_NOT_SHARE/{name}.pickle"


        with open(CORPUS_PATH, 'rb') as handle:
            loaded_data = pickle.load(handle)
        (tokens, id2word, corpus) = loaded_data

        find_optimal_k(*loaded_data, name)


#%% Graph
if __name__ == '__main__':

    for name, token_col in CORPUSES:
        CORPUS_PATH = f"DO_NOT_SHARE/{name}.pickle"

        # Setup Graphs
        sns.set(rc={'figure.figsize': (9, 5)}, font="Calibri", font_scale=1.2)
        sns.set_style("whitegrid", {'axes.grid': False})

        topicDf = pd.read_csv(f"summaries/{name}_Topics_by_k.csv")
        dx = topicDf.groupby("Num_Topics", as_index=False).first().copy()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax2 = ax.twinx()
        ax2.set_ylim(-7.5, -9)
        sns.lineplot(ax=ax, x="Num_Topics", y="Coherence", data=dx, label="Coherence")
        sns.lineplot(ax=ax2, x="Num_Topics", y="Perplexity", data=dx, color="orange", label="Perplexity")

        ax.set(ylabel="Coherence score", xlabel="Num Topics")

        fig.tight_layout()
        plt.savefig(f"summaries/{name}_Coherence_and_Perplexity.png", bbox_inches='tight')
        plt.show()
    print("Done")

