# %%
"""
BERTopic clustering on CFPB complaint narratives.
Reads sampled_complaints.csv, runs topic modelling, optionally exports clustered CSV. for further analysis. 
Tuned for 5000 documents to reduce outliers and produce coherent topics.

"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# BERTopic and dependencies
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer

# Suppress harmless warnings (UMAP, HDBSCAN)
warnings.filterwarnings("ignore", category=UserWarning)


# 1. Load data
INPUT_CSV = "ID_sampled_complaints.csv"
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"{INPUT_CSV} not found. Run preprocessing first.")

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows from {INPUT_CSV}")
print(f"Columns: {df.columns.tolist()}")

# Use the cleaned text column (created by preprocessing)
text_column = "text_clean"
if text_column not in df.columns:
    raise ValueError(f"Column '{text_column}' not found. Available: {df.columns.tolist()}")

docs = df[text_column].fillna("").astype(str).tolist()
print(f"Number of documents: {len(docs)}")

'''
# 2. Optional: quick boxplot of narrative lengths - Long narratives can be harder to cluster, they don't match the input size of the embedding model (384 tokens for MiniLM).
narrative_lengths = [len(text.split()) for text in docs]
plt.figure(figsize=(10, 6))
plt.boxplot(narrative_lengths, vert=True, patch_artist=True)
plt.title('Distribution of Narrative Lengths (Word Count)')
plt.ylabel('Number of Words')
plt.xlabel('All Narratives')
plt.grid(True, alpha=0.3)

mean_len = np.mean(narrative_lengths)
median_len = np.median(narrative_lengths)
plt.text(1.1, mean_len, f'Mean: {mean_len:.0f}',
         bbox=dict(facecolor='white', alpha=0.8))
plt.text(1.1, median_len, f'Median: {median_len:.0f}',
         bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.show()
'''

# 3. Configure BERTopic components (tuned for 5k docs)
print("\nInitialising BERTopic components...")

# Embedding model (fast one, questionable clusters) 
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')



# UMAP – reduce to 5 components, local neighbourhood 15
umap_model = umap.UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    low_memory=True,
    random_state=48
)

# HDBSCAN – parameters to drastically reduce outliers
#   min_cluster_size=15  (as suggested for 5000 docs)
#   min_samples=3        (between 1 and 5)
#   cluster_selection_epsilon=0.05  (small = tighter clusters)
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=15,
    min_samples=3,
    metric='euclidean',
    cluster_selection_method='eom',
    cluster_selection_epsilon=0.05,
    prediction_data=True
)

# Vectorizer – adapt min_df to number of documents
vectorizer_model = CountVectorizer(
    stop_words="english",
    min_df=max(2, len(docs) // 100),   # word must appear in at least 2 docs or 1% of corpus
    max_df=0.85                         # ignore words that appear in >85% of docs
)


# 4. Running BERTopic
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=True,
    verbose=True
)

print("Fitting BERTopic model (may take a few minutes)...")
topics, probs = topic_model.fit_transform(docs)

# Reduce outliers using c‑TF‑IDF (often improves coherence)
print("Reducing outliers with c-TF-IDF strategy...")
new_topics = topic_model.reduce_outliers(docs, topics, strategy="c-tf-idf")
topic_model.update_topics(docs, topics=new_topics)
topics = new_topics   # use cleaned topics

# 5. Attach topics to DataFrame

df['topic'] = topics
df['topic_probability'] = probs.max(axis=1) if probs is not None else np.nan

# 6. Analysis and summaries

topic_info = topic_model.get_topic_info()
num_topics = len(topic_info[topic_info['Topic'] != -1])
noise_ratio = (df['topic'] == -1).mean()

print("\n" + "="*60)
print("TOPIC MODEL SUMMARY")
print("="*60)
print(f"Total documents        : {len(df)}")
print(f"Number of topics       : {num_topics}")
print(f"Noise ratio (outliers) : {noise_ratio:.2%}")
print("\nTop 5 topics by count:")
print(topic_info.head(5).to_string(index=False))

# Cluster sizes with top words
print("\nCluster sizes (top 10):")
cluster_counts = Counter([t for t in topics if t != -1])
for cluster_id, count in cluster_counts.most_common(10):
    words = topic_model.get_topic(cluster_id)
    top_3 = ", ".join([w for w, _ in words[:3]]) if words else ""
    print(f"  Topic {cluster_id:3d} : {count:4d} docs  | keywords: {top_3}")

# 7. Visualisations
# Interactive (plotly)
try:
    print("\nGenerating interactive visualisations (plotly)...")
    topic_model.visualize_topics().show()
    topic_model.visualize_barchart().show()
    topic_model.visualize_heatmap().show()
except Exception as e:
    print(f"Interactive visualisations skipped Error: {e}")

# Static matplotlib fallback
print("\nGenerating static matplotlib visualisation...")
plt.figure(figsize=(12, 5))
topic_counts_plot = topic_info[topic_info['Topic'] != -1]['Count'].values
topic_labels = [f"T{tid}" for tid in topic_info[topic_info['Topic'] != -1]['Topic'].values]
plt.bar(range(len(topic_counts_plot)), topic_counts_plot, color='steelblue')
plt.xlabel('Topic ID')
plt.ylabel('Number of Documents')
plt.title('Topic Distribution (non‑noise)')
plt.xticks(range(len(topic_counts_plot)), topic_labels, rotation=45)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 8. Save final clustered data OPTIONAL – includes 'topic' and 'topic_probability' columns
# ------------------------------------------------------------
OUTPUT_CSV = "IDcomplaints_clusters_finetuned.csv"
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"\n✅ Saved clustered data to '{OUTPUT_CSV}'")
print(f"   Columns: {df.columns.tolist()}")
print(f"   First 3 rows preview:")
print(df[['topic', text_column]].head(3))

# %%
