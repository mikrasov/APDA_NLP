# %%%% Setup project
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.stats import chi2_contingency


NUM_TOPICS = 19
TOPIC_CUTOFF = 0.33
NUM_RESPONSES = 2062
OUT_FOLDER = "analysis/"

QUESTION = {
    "4B": "Rate your satisfaction with this program’s efforts to foster a healthy, respectful academic culture or climate.",
    "6B": "When you interact with other philosophers in professional and social settings, how comfortable do you find yourself?",
    "7B": "How welcoming do you find academic philosophy to be toward students who are members of underrepresented groups, e.g. women, racial or ethnic minorities, members of the LGBTQ community, people with low socio-economic status, veterans and members of the military, and people with disabilities?",
    "8": "What steps should philosophy take to become more inclusive, if any?",
}

MANUAL_GROUP = [
    ("resources", "Time, Space, Resources"),
    ("individuals", "Individuals"),
    ("interactions", "Interactions"),
    ("action", "Action/Reaction"),
]

MANUAL_TOPICS = [
    ("changes", "Changes", "resources"),
    ("context", "Context", "resources"),
    ("fairness", "Fairness", "resources"),

    ("support", "Support", "resources"),
    ("inclusion", "Accessibility/Inclusion", "resources"),
    ("gender", "Gender", "individuals"),
    ("race", "Race/Ethnicity", "individuals"),
    ("socioeconomic", "Socioeconomic Status", "individuals"),
    ("identity", "Other Identities", "individuals"),
    ("internalized", "Internalized Attitude", "individuals"),
    ("agreement", "Agreement", "interactions"),
    ("engagement", "Engagement", "interactions"),
    ("behaviors", "Attitudes/Behaviors", "interactions"),
    ("abuse", "Harassment/Assault...", "interactions"),
    ("interventions", "Interventions", "action"),
    ("responsive", "Responsive", "action"),
    ("tolerance", "Tolerance", "action"),
    ("resilient", "Resilient", "action"),
]

COLS_MANUAL_TOPICS = [f"manual_{name}" for name, label, group in MANUAL_TOPICS]
COLS_MANUAL_GROUPS = [f"manual_group_{group}" for group, label in MANUAL_GROUP]
COLS_LDA_TOPICS = [f"lda_topic_{t}" for t in range(NUM_TOPICS)]

COLS_ALL_TOPICS = COLS_MANUAL_TOPICS + COLS_LDA_TOPICS

QUESTIONS = ["4B", "6B", "7B", "8"]

sns.set(rc={'figure.figsize': (9, 5)}, font="Calibri", font_scale=1.2)
sns.set_style("whitegrid", {'axes.grid': False})

results = pd.read_csv("results/dataset_sanitized.csv").set_index('id')
results["sent_overall"] = results["sent_overall"].map({'positive': 1, 'neutral': 0, 'negative': -1})

personMap = pd.read_csv("results/LinkedIDs.csv").set_index('id')
results = results.join(personMap)

public = pd.read_csv("results/PublicSurveyResponsesAPDA.csv").set_index('id')
public = public.drop(columns=['question', 'likert_scale'])
public["response"] = public.response.str.replace("true##", "")
results = results.join(public)

manual_tags = pd.read_csv("results/manual_tags.csv").set_index('id').add_prefix('manual_')
for name, label, group in MANUAL_TOPICS:
    manual_tags.loc[~manual_tags[f"manual_{name}"].isnull(), f"manual_group_{group}"] = manual_tags[
        "manual_overall_valence"]
results = results.join(manual_tags)

likert = pd.read_csv("results/Likert.csv").set_index('id')
likert = likert.stack().reset_index()
likert = likert.rename(columns={"level_1": "question", 0: "likert"})
likert.loc[likert.likert == 3, "likert_overall"] = 0
likert.loc[likert.likert > 3, "likert_overall"] = 1
likert.loc[likert.likert < 3, "likert_overall"] = -1

results = results.reset_index().merge(likert, how="left", left_on=['id', 'question'],
                                      right_on=['id', 'question']).set_index('id')

results = results.drop(columns=["Unnamed: 0"])

results["lda_all_topics"] = 0
for t in range(NUM_TOPICS):
    results.loc[results[f"lda_topic_{t}_prob"] > TOPIC_CUTOFF, f"lda_topic_{t}"] = 1
    results["lda_all_topics"] = results["lda_all_topics"] + results[f"lda_topic_{t}"].fillna(0)
results.to_csv(f"{OUT_FOLDER}/results.csv", float_format='%.2f')

def toSign(col):
    return col.apply(lambda x : "-" if x < 0 else "")
#%%% Count ratio of under-represented to represented in the sample

under_rep = results.groupby("personId").first()["Combined_Underrepresented"]
print("Percent under-represented People", (under_rep.sum() / under_rep.count()))

under_rep = results["Combined_Underrepresented"]
print("Percent under-represented Responses", (under_rep.sum() / under_rep.count()))

# %%%% Table of Topics

# Calculate topic Occurrence
df_count = results.groupby("question").count()
df_sent = results.copy()

df_sent["likert"] = (df_sent["likert"] - 3) / 2
df_sent[COLS_ALL_TOPICS] = df_sent[COLS_ALL_TOPICS].where(df_sent.isnull(), 1)

df_sent_likert = df_sent[COLS_ALL_TOPICS].multiply(df_sent["likert"], axis="index").mean().rename('likert')
df_sent_vader = df_sent[COLS_ALL_TOPICS].multiply(df_sent["sent_compound"],axis="index").mean().rename('vader')
df_sent_manual = df_sent[COLS_ALL_TOPICS].multiply(df_sent["manual_overall_valence"],axis="index").mean().rename('manual')

for q in QUESTIONS:
    df_count.loc[q, COLS_LDA_TOPICS] = df_count.loc[q, COLS_LDA_TOPICS] / df_count.loc[q, COLS_LDA_TOPICS].sum()
    df_count.loc[q, COLS_MANUAL_TOPICS] = df_count.loc[q, COLS_MANUAL_TOPICS] / df_count.loc[q, COLS_MANUAL_TOPICS].sum()

df = df_count[COLS_ALL_TOPICS].copy().transpose()
df["Topic"] = [label for name, label, group in MANUAL_TOPICS] + [str(t) for t in range(NUM_TOPICS)]
df["Group"] = [MANUAL_GROUP[0][1], "", "", "", "", MANUAL_GROUP[1][1], "", "", "", "", MANUAL_GROUP[2][1], "", "", "",
               MANUAL_GROUP[3][1], "", "", "", "LDA"] + ["" for t in range(NUM_TOPICS - 1)]

df_sent = results.groupby("question").mean()

df = df[["Group", "Topic"] + QUESTIONS].round(decimals=2)
df[QUESTIONS] = (df[QUESTIONS] * 100).astype(int)
df = pd.concat([df, df_sent_likert, df_sent_vader, df_sent_manual], axis=1).round(decimals=1)

table = df.to_latex(index=False)
with open(OUT_FOLDER + '/table_topics.txt', 'w') as f:
    f.write(table)

# %%%% Correlation Matrix - GROUP present
df = results[COLS_MANUAL_GROUPS + COLS_LDA_TOPICS]
df = df.where(df.isnull(), 1).fillna(0).astype(int)
corr_matrix = df.corr(method="spearman",min_periods=10).iloc[:len(COLS_MANUAL_GROUPS), len(COLS_MANUAL_GROUPS):]

chi2 = pd.DataFrame()
for group in COLS_MANUAL_GROUPS:
    for topic in COLS_LDA_TOPICS:
        contigency = pd.crosstab(df[group], df[topic])
        c, p, dof, expected = chi2_contingency(contigency)
        chi2.loc[group,topic] = p

fig, axs = plt.subplots(2,1, sharex=True)
fig.set_size_inches(9, 4)

sns.heatmap(corr_matrix, ax=axs[0], cmap=sns.diverging_palette(230, 20, as_cmap=True),
            vmin=-0.2, vmax=0.2, center=0, square=True,
            yticklabels=[label for group, label in MANUAL_GROUP],
            xticklabels=range(NUM_TOPICS),
            annot=corr_matrix.apply(toSign),  fmt = '',
)
axs[0].set(ylabel="", xlabel="")
axs[0].set_title("Spearman Correlation", fontsize=13, fontweight="bold")

sns.heatmap(chi2, ax=axs[1], cmap=sns.light_palette("seagreen", as_cmap=True, reverse=True),
            vmin=0, vmax=0.1, center=0, square=True,
            yticklabels=[label for group, label in MANUAL_GROUP],
            xticklabels=range(NUM_TOPICS),
)
axs[1].set(ylabel="", xlabel="LDA Topic")
axs[1].set_title("Chi-2", fontsize=13, fontweight="bold")


fig.tight_layout()
plt.savefig(OUT_FOLDER + "Fig - Correlation Manual Group vs LDA.png")
plt.show()

# %%%% Correlation Matrix - Topic present


df = results[COLS_ALL_TOPICS]
df = df.where(df.isnull(), 1).fillna(0).astype(int)
corr_matrix = df.corr(method="spearman",min_periods=10).iloc[:len(COLS_MANUAL_TOPICS), len(COLS_MANUAL_TOPICS):]


chi2 = pd.DataFrame()
for manual in COLS_MANUAL_TOPICS:
    for topic in COLS_LDA_TOPICS:
        contigency = pd.crosstab(df[manual], df[topic])
        c, p, dof, expected = chi2_contingency(contigency)
        chi2.loc[manual,topic] = p

fig, axs = plt.subplots(1,2, sharey=True)
fig.set_size_inches(12, 6)


sns.heatmap(corr_matrix, ax=axs[0], cmap=sns.diverging_palette(230, 20, as_cmap=True),
            vmin=-0.2, vmax=0.2, center=0, square=True, cbar_kws={"shrink": .64},
            annot=corr_matrix.apply(toSign),  fmt = '',
            yticklabels=[label for name, label, group in MANUAL_TOPICS],
            xticklabels=range(NUM_TOPICS))
axs[0].set(ylabel="", xlabel="LDA Topic")
axs[0].set_title("Spearman Correlation", fontsize=13, fontweight="bold")
axs[0].set_xticklabels(axs[0].get_xticklabels(),rotation=0, fontsize=11)

sns.heatmap(chi2, ax=axs[1], cmap=sns.light_palette("seagreen", as_cmap=True, reverse=True),
            vmin=0, vmax=0.1, center=0, square=True, cbar_kws={"shrink": .64},
            yticklabels=[label for name, label, group in MANUAL_TOPICS],
            xticklabels=range(NUM_TOPICS))
axs[1].set(ylabel="", xlabel="LDA Topic")
axs[1].set_title("Chi-2", fontsize=13, fontweight="bold")
axs[1].set_xticklabels(axs[1].get_xticklabels(),rotation=0, fontsize=11)
fig.tight_layout()
plt.savefig(OUT_FOLDER + "Fig - Correlation Manual Topic vs LDA.png")
plt.show()

# %%%% Sentiment Histograms
fig = plt.figure(constrained_layout=True)
fig.set_size_inches(12, 6)
spec2 = gridspec.GridSpec(ncols=5, nrows=3, figure=fig)
axs = [0, 0, 0, 0]
axs[0] = fig.add_subplot(spec2[0, 0:3])
axs[1] = fig.add_subplot(spec2[1, 0:3])
axs[2] = fig.add_subplot(spec2[2, 0:3])
axs[3] = fig.add_subplot(spec2[0:3, 3:5])

df = results.copy()
df["likert"] = (df["likert"] - 3) / 2

dx = df.rename(columns={"likert": "Likert", "sent_compound": "VADER", "manual_overall_valence": "Manual"})
dx = dx[["Likert", "VADER", "Manual"]].stack().reset_index().set_index("id")
dx = dx.rename(columns={"level_1": "Type", 0: "Sentiment", })

sns.histplot(ax=axs[0], x="likert", data=df, color=sns.color_palette()[0], bins=15)
sns.histplot(ax=axs[1], x="sent_compound", data=df, color=sns.color_palette()[1], bins=15)
sns.histplot(ax=axs[2], x="manual_overall_valence", bins=15, data=df, color=sns.color_palette()[2])
sns.boxplot(ax=axs[3], y="Type", x="Sentiment", orient="h", data=dx,
            medianprops={"color": '#CCCCCC', "linestyle": '--', "linewidth": 3, },
            meanprops={"color": '#FF0000', "linestyle": (0, (1, 4)), "linewidth": 3, }, showmeans=True, meanline=True
            )

axs[0].set(ylabel="Count", xlabel="")
axs[1].set(ylabel="Count", xlabel="")
axs[2].set(ylabel="Count", xlabel="Sentiment")
axs[3].set(ylabel="", xlabel="Sentiment")

axs[0].axvline(ymin=0, ymax=1, x=df["likert"].median(), color="#CCC", lw=3, linestyle='--')
axs[1].axvline(ymin=0, ymax=1, x=df["sent_compound"].median(), color="#CCC", lw=3, linestyle='--')
axs[2].axvline(ymin=0, ymax=1, x=df["manual_overall_valence"].median(), color="#CCC", lw=3, linestyle='--')

axs[0].axvline(ymin=0, ymax=1, x=df["likert"].mean(), color="#F00", lw=3, linestyle=(0, (1, 4)))
axs[1].axvline(ymin=0, ymax=1, x=df["sent_compound"].mean(), color="#F00", lw=3, linestyle=(0, (1, 4)))
axs[2].axvline(ymin=0, ymax=1, x=df["manual_overall_valence"].mean(), color="#F00", lw=3, linestyle=(0, (1, 4)))

axs[0].axvline(ymin=0, ymax=1, x=0, color="#666", lw=1)
axs[1].axvline(ymin=0, ymax=1, x=0, color="#666", lw=1)
axs[2].axvline(ymin=0, ymax=1, x=0, color="#666", lw=1)
axs[3].axvline(ymin=0, ymax=1, x=0, color="#666", lw=1)

plt.savefig(OUT_FOLDER + f"Fig - Sentiment Histograms.png")
plt.show()

# %%%% Correlation Matrix
fig, axs = plt.subplots(2, 5, sharex= True)
fig.set_size_inches(12, 5)

for n,title,df in [
    (0,"Question 4", results[results.question == "4B"]),
    (1,"Question 6", results[results.question == "6B"]),
    (2,"Question 7", results[results.question == "7B"]),
    (3,"Question 8", results[results.question == "8"]),
    (4,"All", results),
]:

    dx = df[["question", "likert", "sent_compound", "manual_overall_valence"]].copy()
    dx = dx.rename(columns={"likert": "Likert", "sent_compound": "VADER", "manual_overall_valence": "Manual"})

    corr_matrix = dx.corr(method="spearman",min_periods=10).iloc[1:, :-1]
    corr_matrix.at['VADER', 'VADER'] = np.nan

    dx = df[["question", "likert", "sent_overall", "manual_overall_valence"]].copy()
    dx = dx.rename(columns={"likert": "Likert", "sent_overall": "VADER", "manual_overall_valence": "Manual"})

    chi2 = pd.DataFrame()
    for col1, col2 in [ ("VADER", "Likert"),("Manual", "Likert"),("Manual", "VADER"), ]:
        if title == "Question 8" and col2 == "Likert":
            chi2.loc[col1, col2] = np.nan
            continue
        contigency = pd.crosstab(dx[col1], dx[col2])
        c, p, dof, expected = chi2_contingency(contigency)
        chi2.loc[col1,col2] = p
        chi2 = chi2.round(decimals=3)

    sns.heatmap(corr_matrix, ax=axs[0,n], cmap=sns.diverging_palette(230, 20, as_cmap=True), vmin=0, vmax=0.67, center=0, square=True, annot=True, cbar=False)
    axs[0,n].set_title(title, fontsize=13, fontweight="bold")

    sns.heatmap(chi2, ax=axs[1,n], cmap=sns.light_palette("seagreen", as_cmap=True, reverse=True), vmin=0, vmax=0.1, center=0, square=True, annot=True, cbar=False, fmt=".3f")

axs[0,0].set(ylabel="Spearman", xlabel="")
axs[1,0].set(ylabel="Chi-2", xlabel="")

fig.tight_layout()
plt.savefig(OUT_FOLDER + "Fig - Likert.png")
plt.show()

# %%% Chi Squared - by Question
chi2 = pd.DataFrame(columns=["Slice", "Variable", "Comparison", "p"])
res_matrix = pd.DataFrame()

for question_name, df in [
    ("All", results),
    ("Q4", results[results.question == "4B"]),
    ("Q6", results[results.question == "6B"]),
    ("Q7", results[results.question == "7B"]),
    ("Q8", results[results.question == "8"]),
]:
    corr = df[["Combined_Underrepresented", "likert", "manual_overall_valence", "sent_compound"]].copy()
    corr = corr.corr(method="spearman",min_periods=10).iloc[:1, 1:]
    corr["question"] = question_name

    df["sent_compound"] = df["sent_overall"]
    for var_col in ["Combined_Underrepresented", "likert", "manual_overall_valence", "sent_compound"]:
        for sentiment_col in ["likert", "manual_overall_valence", "sent_compound", ]:
            if (sentiment_col == "likert" or var_col == "likert") and question_name == "Q8": continue
            contigency = pd.crosstab(df[var_col], df[sentiment_col])

            c, p, dof, expected = chi2_contingency(contigency)
            chi2.loc[len(chi2.index)] = [question_name, var_col, sentiment_col, p]

    corr.index = ["Spearman"]
    chi_row = chi2.loc[
        (chi2.Variable == "Combined_Underrepresented") & (chi2["Slice"] == question_name),
        ["Comparison", "p"]
    ].set_index("Comparison").T

    chi_row["type"] = "Chi-2"
    chi_row["question"] = question_name
    chi_row = chi_row.round(decimals=3)
    corr["type"] = "Pearson"
    corr = corr.round(decimals=2)
    res_matrix = pd.concat([res_matrix, corr, chi_row], ignore_index=True)


# Make actual Graphs
fig, axs = plt.subplots(2, 5, sharey=True, sharex=True)
fig.set_size_inches(12, 3)


for n, question in enumerate(["Q4", "Q6", "Q7", "Q8", "All"]):
    df_pearson = res_matrix[(res_matrix.question == question) & (res_matrix.type == "Pearson")].drop(columns=["question","type"])
    df_chi2 = res_matrix[(res_matrix.question == question) & (res_matrix.type == "Chi-2")].drop(columns=["question","type"])

    sns.heatmap(data=df_pearson, ax=axs[0, n], cmap=sns.diverging_palette(230, 20, as_cmap=True),
                vmin=-0.21, vmax=0.21, center=0, annot=True, cbar=False, xticklabels=["Likert", "Manual", "VADER"],
                )
    sns.heatmap(data=df_chi2, ax=axs[1, n], cmap=sns.light_palette("seagreen", as_cmap=True, reverse=True),
                vmin=0, vmax=0.2, center=0, annot=True, cbar=False, xticklabels=["Likert", "Manual", "VADER"],fmt=".3f",
                )
    axs[0,n].set_yticks([])
    axs[1,n].set_yticks([])
    axs[0,n].set_title(question, fontsize=13, fontweight="bold")
    axs[0,n].set(ylabel="", xlabel="")
    axs[1,n].set(ylabel="", xlabel="")

axs[0,0].set(ylabel="Spearman", xlabel="")
axs[1,0].set(ylabel="Chi-2", xlabel="")

fig.tight_layout()
plt.savefig(OUT_FOLDER + f"Fig - Representation v Sentiment Correlation - By question.png")
plt.show()

# %%% Underrepresented correlation to occurrence of a topic

chi2 = pd.DataFrame()
res_matrix = pd.DataFrame()

for topic in COLS_ALL_TOPICS:
    for sentiment_col, sentiment_name in [
        ("likert", "Likert"),
        ("manual_overall_valence", "Manual"),
        ("sent_overall", "VADER"),
    ]:
        df = results.copy()
        df[topic] = df[topic].where(df[topic].isnull(), 1).multiply(df[sentiment_col], axis="index")

        contigency = pd.crosstab(df["Combined_Underrepresented"], df[topic])
        c, p, dof, expected = chi2_contingency(contigency)

        chi2.loc[topic,sentiment_name] = p


corr_matrix = pd.DataFrame()
for sentiment_col, sentiment_name in [
    ("likert", "Likert"),
    ("manual_overall_valence", "Manual"),
    ("sent_compound", "VADER"),
]:
    df = results.copy()
    df[COLS_ALL_TOPICS] = df[COLS_ALL_TOPICS].where(df.isnull(), 1).multiply(df[sentiment_col], axis="index")
    df = df[["Combined_Underrepresented"] + COLS_ALL_TOPICS]

    df = df.rename(columns={"Combined_Underrepresented": sentiment_name})
    df = df.corr(method="spearman",min_periods=10).iloc[1:, 0:1]
    corr_matrix = pd.concat([corr_matrix, df], axis=1)


fig, axs = plt.subplots(1, 2, sharey=True)
fig.set_size_inches(8.5, 11)

labels = [label for name, label, group in MANUAL_TOPICS] + [f"LDA {l}" for l in range(NUM_TOPICS)]
sns.heatmap(corr_matrix, ax=axs[0], cbar=False, cmap=sns.diverging_palette(230, 20, as_cmap=True),
            vmin=-0.3, vmax=0.3, center=0, annot=True, yticklabels=labels,fmt=".2f")
axs[0].set_title("Spearman", fontsize=13, fontweight="bold")

sns.heatmap(chi2, ax=axs[1], cbar=False, cmap=sns.light_palette("seagreen", as_cmap=True, reverse=True),
            vmin=0, vmax=0.2, center=0, annot=True, yticklabels=labels,fmt=".3f",)
axs[1].set_title("Chi-2", fontsize=13, fontweight="bold")

fig.tight_layout()
plt.savefig(OUT_FOLDER + f"Fig - Representation v Sentiment Correlation - by topic.png")
plt.show()


# %%% Sentiment mean by question divided by underrepresented (with Confidence intervals)
df = results[["question", "Combined_Underrepresented", "likert", "sent_compound", "manual_overall_valence"]].reset_index().copy()
df["likert"] = (df["likert"] - 3) / 2

df["Combined_Underrepresented"] = df["Combined_Underrepresented"].map({1: "Underrepresented", 0: 'Represented'})

df = df.rename(columns={"likert": "sent_1", "sent_compound": "sent_2", "manual_overall_valence": "sent_3"})
df = pd.wide_to_long(df, stubnames="sent", i=["id", "question"], j="Type", sep="_").reset_index().set_index("id")
df["Type"] = df["Type"].map({1: "Likert", 2: "VADER", 3: "Manual"}, na_action='ignore')

fig, axs = plt.subplots(1, 5, sharey=True)
fig.set_size_inches(15, 4)
for n, label, dx in [
    (0,"Q4",df[df["question"] == "4B"]),
    (1,"Q6",df[df["question"] == "6B"]),
    (2,"Q7",df[df["question"] == "7B"]),
    (3,"Q8",df[df["question"] == "8"]),
    (4,"ALL",df),
]:
    sns.barplot(ax=axs[n], x="Type", y="sent", data=dx, capsize=.1,
                hue="Combined_Underrepresented", hue_order=["Underrepresented", "Represented"])
    axs[n].set(ylabel="", xlabel=label)
    handles, labels = axs[n].get_legend_handles_labels()
    axs[n].axhline(xmin=0, xmax=1, y=0, color="#666", lw=1)
    axs[n].set_ylim([-0.6, 0.6])
    axs[n].get_legend().remove()

fig.legend(handles=handles, labels=labels, loc=(0.4, 0.9), fancybox=False, shadow=False, frameon=False, ncol=2)
fig.tight_layout(pad=2)
plt.savefig(OUT_FOLDER + f"Fig - Representation v Sentiment by Question.png")
plt.show()

#%%% Sentiment mean by question divided by Topic (with Confidence intervals)
fig, axs = plt.subplots(1,3, sharey=True)
fig.set_size_inches(9, 11)

for n, sentiment_col, sentiment_name in [
    (0,"likert", "Likert"),
    (1,"manual_overall_valence", "Manual"),
    (2,"sent_compound", "VADER"),
]:
    df = results.copy()
    df["likert"] = (df["likert"] - 3) / 2

    df[COLS_MANUAL_TOPICS] = df[COLS_MANUAL_TOPICS].where(df.isnull(), 1).multiply(df[sentiment_col],axis="index")

    # Break into rows
    col_underrepresented = df["Combined_Underrepresented"].map({1: "Underrepresented", 0: 'Represented'})
    df = df[COLS_MANUAL_TOPICS].stack().reset_index().set_index("id").join(col_underrepresented)
    df = df.rename(columns={"level_1": "Topic", 0: sentiment_name})

    sns.barplot(ax=axs[n], y="Topic", x=sentiment_name,  data=df, capsize=.1,
                hue="Combined_Underrepresented", hue_order=["Underrepresented", "Represented"],
                order=COLS_MANUAL_TOPICS)


    axs[n].set(xlabel="Sentiment", ylabel="")
    axs[n].axvline(ymin=0, ymax=1, x=0, color="#666", lw=1)
    axs[n].set_xlim([-1, 1])

    axs[n].set_yticks(range(18))
    axs[n].set_yticklabels([label for name, label, group in MANUAL_TOPICS])
    axs[n].set(xlabel=sentiment_name)
    handles, labels = axs[n].get_legend_handles_labels()
    axs[n].get_legend().remove()



fig.tight_layout(pad=1.5)
fig.subplots_adjust( wspace=0.14)
fig.legend(handles=handles, labels=labels, loc=(0.36, 0.97), fancybox=False, shadow=False, frameon=False, ncol=2)
plt.savefig(OUT_FOLDER + f"Fig - Representation - Mean Sentiment by Topic.png")
plt.show()

#%%% To Grayscale
from PIL import Image
import os, glob
path = 'analysis/'

for filename in glob.glob(os.path.join(path, '*.png')):
    print(f"Converting '{filename}' to grayscale")
    img = Image.open(filename)
    imgGray = img.convert('L')
    imgGray.save(filename)


