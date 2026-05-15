# ============================================================
# FULL PIPELINE: STRATIFIED SAMPLING + BERTopic CLUSTERING
# ============================================================
# Author: [Eylül Semercioglu]
# Purpose: Bachelor thesis - 
# ============================================================
'''
maximise narrative variety for unsupervised clustering
embedding strategy 
chunking and averaging function for vector embeddings
language specific functions were tried, checking to see the difference again

'''
#%% ------------------------------
# 1. IMPORTS
# ------------------------------
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import re
import torch

# Spark (if using PySpark)
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.sql.functions import col, length, lit, sum as spark_sum, to_date, regexp_replace, monotonically_increasing_id, trim
from pyspark.sql.types import StructType, StructField, StringType
import os
# NLP & Clustering
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')

# Visualisation
import matplotlib.pyplot as plt

# %%------------------------------
# 2. LOAD DATA (assume Spark DataFrame `df_valid` already exists)
# ------------------------------

spark = SparkSession.builder \
    .appName("CustomerSupportAnalysis") \
    .master("local[6]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "48") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.driver.host", "localhost") \
    .config("spark.driver.port", "0") \
    .getOrCreate()

print(f"✅ Spark {spark.version} running on CPU")
print(f"Spark Home: {os.environ.get('SPARK_HOME')}")

# read in the file
df_original = spark.read.csv("complaints.csv", header=True, multiLine=True, quote='"', escape='"', inferSchema=False)

# snake_case helper function for column names
def to_snake_case(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

# standardize column names to snake_case for easier access in code
df_original = df_original.toDF(*[to_snake_case(c) for c in df_original.columns])

# Delete unnecessary columns
columns_to_drop = ["zip_code","consumer_consent_provided", "tags", "consumer_disputed", "submitted_via","date_sent_to_company", "timely_response", "state", "company_response_to_consumer", "company_public_response", "company"]
df_original = df_original.drop(*columns_to_drop)

# remove the instances where consumer_complaint_narrative is null, since it's the most important column for our analysis
df_original = df_original.filter(col("consumer_complaint_narrative").isNotNull())
print(df_original.count())

# Convert the date so it's not a string anymore
df_original = df_original.withColumn("date_received", to_date(col("date_received"), "yyyy-MM-dd"))
 
# keep only rows where text_clean is not null and not empty string EDITED TO 50 chars
df_valid = df_original.filter(F.length(F.col("consumer_complaint_narrative")) > 50) 

print(f"Original rows: {df_original.count()} | After removing null/empty text_clean: {df_valid.count()}")

# If after filtering you have zero rows, raise an error or handle it
if df_valid.count() == 0:
    raise ValueError("No valid text_clean rows available for sampling.")

# %%
# check the new product representation 
df_valid.groupBy("product").count().orderBy("count", ascending=False).show(truncate=False, n=df_valid.select("product").distinct().count())


# %%============================================================
# PART A: STRATIFIED SAMPLING (maximise variety)
# ============================================================

# ------------------------------
# A1. Parameters
# ------------------------------
total_sample_n = 100000          # adjustable
min_per_product = 300             # every product appears at least this many times
min_per_issue_large_product = 5  # for large product(s): min rows per issue
max_product_fraction = 0.35       # largest product can take max 35% of sample
seed = 25


# ------------------------------
# A3. Get product count dictionaries
# ------------------------------
product_counts = df_valid.groupBy("product").count().collect()
product_counts_dict = {row["product"]: row["count"] for row in product_counts}
total_rows = sum(product_counts_dict.values())

# Separate small products (take all)
small_products = {p: c for p, c in product_counts_dict.items() if c <= min_per_product}
large_products = {p: c for p, c in product_counts_dict.items() if c > min_per_product}

# ------------------------------
# A4. Allocate samples per product
# ------------------------------
# Reserve all small product rows
reserved_small = sum(small_products.values())

# For large products: base min_per_product + proportional share of remaining budget
remaining_budget = total_sample_n - reserved_small - (len(large_products) * min_per_product)
if remaining_budget < 0:
    raise ValueError(f"total_sample_n too small: need at least {reserved_small + len(large_products)*min_per_product} rows")

# Proportional allocation among large products (based on original counts)
large_total = sum(large_products.values())
proportional = {}
for prod, cnt in large_products.items():
    ideal = remaining_budget * (cnt / large_total)
    # Cap each product to max_product_fraction of total_sample_n
    max_allowed = total_sample_n * max_product_fraction
    proportional[prod] = min(ideal, max_allowed)

# Final allocation: base + proportional
final_allocation = {}
for prod in large_products:
    final_allocation[prod] = min_per_product + proportional[prod]
    # Ensure we don't allocate more than population
    final_allocation[prod] = min(final_allocation[prod], large_products[prod])

# Add small products (take all)
for prod, cnt in small_products.items():
    final_allocation[prod] = cnt

# Adjust total (in case caps reduced sum)
total_allocated = sum(final_allocation.values())
if total_allocated > total_sample_n:
    # Scale down large products proportionally
    scale = total_sample_n / total_allocated
    for prod in large_products:
        final_allocation[prod] = int(final_allocation[prod] * scale)
    # Add back small products unchanged
    total_allocated = sum(final_allocation.values())
    # Subtract excess from largest product if still over
    if total_allocated > total_sample_n:
        diff = total_allocated - total_sample_n
        largest_prod = max(large_products, key=lambda p: large_products[p])
        final_allocation[largest_prod] -= diff

print("\nSample allocation per product:")
for prod, n in sorted(final_allocation.items(), key=lambda x: x[1], reverse=True):
    print(f"  {prod}: {n}")

# ------------------------------
# A5. Identify largest product for issue stratification
# ------------------------------
largest_product = max(product_counts_dict, key=product_counts_dict.get)
print(f"\nLargest product: {largest_product}")

# ------------------------------
# A6. Sample the largest product with issue stratification
# ------------------------------
sampled_parts = []

if largest_product in final_allocation and final_allocation[largest_product] > min_per_issue_large_product * 2:
    # Get all rows of largest product
    df_largest = df_valid.filter(col("product") == largest_product)
    
    # Get issue counts
    issue_counts = df_largest.groupBy("issue").count().collect()
    issue_counts_dict = {row["issue"]: row["count"] for row in issue_counts}
    total_issue_rows = sum(issue_counts_dict.values())
    
    target_largest_n = final_allocation.pop(largest_product)  # remove from general allocation
    
    # Allocate per issue: each issue gets min_per_issue_large_product, then proportionally
    issue_allocation = {}
    reserved_min = len(issue_counts_dict) * min_per_issue_large_product
    remaining_issues = target_largest_n - reserved_min
    
    if remaining_issues < 0:
        # Budget too small: just take proportionally with floor 1
        for issue, cnt in issue_counts_dict.items():
            issue_allocation[issue] = max(1, int(target_largest_n * (cnt / total_issue_rows)))
    else:
        for issue, cnt in issue_counts_dict.items():
            prop = remaining_issues * (cnt / total_issue_rows)
            issue_allocation[issue] = min_per_issue_large_product + prop
            issue_allocation[issue] = min(issue_allocation[issue], cnt)
    
    # Adjust total to match target
    total_issue_alloc = sum(issue_allocation.values())
    if total_issue_alloc != target_largest_n:
        max_issue = max(issue_counts_dict, key=issue_counts_dict.get)
        issue_allocation[max_issue] += (target_largest_n - total_issue_alloc)
    
    # Sample using sampleBy
    sampling_fractions_largest = {}
    for issue, cnt in issue_counts_dict.items():
        desired = issue_allocation[issue]
        sampling_fractions_largest[issue] = desired / cnt if cnt > 0 else 0.0
    
    sampled_largest = df_largest.sampleBy(col="issue", fractions=sampling_fractions_largest, seed=seed)
    sampled_largest = sampled_largest.withColumn("sampling_stage", lit("issue_stratified"))
    sampled_parts.append(sampled_largest)
else:
    # If largest product not issue-stratified, treat normally (will be sampled later)
    final_allocation[largest_product] = final_allocation.get(largest_product, min_per_product)

# ------------------------------
# A7. Sample remaining products (simple stratified)
# ------------------------------
for prod, target_n in final_allocation.items():
    if target_n <= 0:
        continue
    prod_pop = product_counts_dict[prod]
    if target_n >= prod_pop:
        sampled_sub = df_valid.filter(col("product") == prod)
    else:
        fraction = target_n / prod_pop
        sampled_sub = df_valid.filter(col("product") == prod).sample(False, fraction, seed=seed)
    sampled_sub = sampled_sub.withColumn("sampling_stage", lit("product_stratified"))
    sampled_parts.append(sampled_sub)

# Union all parts
sampled_df = sampled_parts[0]
for part in sampled_parts[1:]:
    sampled_df = sampled_df.union(part)

# Limit to exact total_sample_n (just in case)
sampled_df = sampled_df.limit(total_sample_n)

# Add a unique ID
sampled_df = sampled_df.withColumn("sample_id", monotonically_increasing_id())
# %%============================================================
# PART A.X: Handling redaction placeholders, normalizing numbers and removing stopwords in the text column
# ============================================================
# 
def clean_cfpb_text_col(col: F.Column, normalize_numbers: bool = True) -> F.Column:
    c = F.regexp_replace(col, r"\u00A0", " ")
    c = F.regexp_replace(c, r"\s+", " ")
    
    # Replace redactions
    c = F.regexp_replace(c, r"(?i)\bX+\b", " ")
    c = F.regexp_replace(c, r"(?i)\*{2,}", " ")
    c = F.regexp_replace(c, r"(?i)\b(?:X{2,}[-/])+(?:X{2,})\b", " ")
    c = F.regexp_replace(c, r"(?i)\[(redacted|removed)\]|\(redacted\)", " ")

    # Normalize long act names to acronyms
    c = F.regexp_replace(c, r"(?i)\bfair credit reporting act\b", "FCRA")
    c = F.regexp_replace(c, r"(?i)\bfair debt collection practices act\b", "FDCPA")
    c = F.regexp_replace(c, r"(?i)\btruth in lending act\b", "TILA")
    c = F.regexp_replace(c, r"(?i)\bequal credit opportunity act\b", "ECOA")
    c = F.regexp_replace(c, r"(?i)\breal estate settlement procedures act\b", "RESPA")
    c = F.regexp_replace(c, r"(?i)\bhealth insurance portability and accountability act\b", "HIPAA")
    c = F.regexp_replace(c, r"(?i)\bgramm leach bliley act\b", "GLBA")
    c = F.regexp_replace(c, r"\s*-\s*", "-")
    if normalize_numbers:
        # ----- 1. Protect full legal citations (including hyphens & subsections) -----
        
        # --- Existing patterns (US Code, etc.) ---
        c = F.regexp_replace(c, r"(?i)\b(?:title\s+)?(\d+)\s+(?:U\.?S\.?\.?\s*C(?:ode)?\.?|USC)\s+(\d+(?:[a-z])?(?:[-.]\d+(?:[a-z])?)*)\b", r"us_code_$1_$2")
        c = F.regexp_replace(c, r"(?i)\b(?:U\.?S\.?\.?\s*C(?:ode)?\.?|USC)\s+(\d+(?:[a-z])?(?:[-.]\d+(?:[a-z])?)*)\b", r"us_code_$1")
        
        # --- NEW: Handle act acronym + number in parentheses/brackets, with optional "section" ---
        # Pattern 1: "FCRA (807)" or "FCRA(807)" or "FCRA ( section 807 )"
        c = F.regexp_replace(c, r"(?i)\b(FCRA|FDCPA|TILA|ECOA|RESPA|HIPAA|GLBA)\s*(?:\(|\s*\[\s*)(?:section\s+)?(\d+(?:[a-z])?(?:[-.]\d+(?:[a-z])?)*)\s*(?:\)|\s*\])", r"$1_$2")
        
        # Pattern 2: Act acronym followed by "section" or "§" and number (no brackets) – merge as one token
        c = F.regexp_replace(c, r"(?i)\b(FCRA|FDCPA|TILA|ECOA|RESPA|HIPAA|GLBA)\s+(?:section\s+|§\s*)(\d+(?:[a-z])?(?:[-.]\d+(?:[a-z])?)*)\b", r"$1_$2")
        
        # Pattern 3: Act acronym + space + number (no "section", no brackets) – e.g. "FCRA 807"
        c = F.regexp_replace(c, r"(?i)\b(FCRA|FDCPA|TILA|ECOA|RESPA|HIPAA|GLBA)\s+(\d+(?:[a-z])?(?:[-.]\d+(?:[a-z])?)*)\b", r"$1_$2")
        
        # For clarity, I'd replace the old generic ones with these more specific patterns.
        # If you still want generic "section_807" when no act is named, keep that:
        c = F.regexp_replace(c, r"(?i)\b(section|act|part)\s+(\d+(?:[a-z])?(?:[-.]\d+(?:[a-z])?)*)\b", r"$1_$2")
        c = F.regexp_replace(c, r"§\s*(\d+(?:[a-z])?(?:[-.]\d+(?:[a-z])?)*)", r"section_$1")
        c = F.regexp_replace(c, r"\b\d+(?:[.,]\d+)?\b", " [NUM] ")
    # No global [NUM] replacement – numbers inside citations remain.    
    # Lowercase
    c = F.lower(c)
    # Collapse spaces
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)

# 2. Apply cleaning (add text_clean column)
sampled_df = sampled_df.withColumn(
    "text_clean",
    clean_cfpb_text_col(F.coalesce(F.col("consumer_complaint_narrative"), F.lit("")), normalize_numbers=True)
)

# %%
# Cache
sampled_df.cache()
print(f"\n✅ Final stratified sample size: {sampled_df.count()} rows")

# Convert to Pandas for BERTopic (assuming manageable size)
df = sampled_df.select(
    "text_clean", "product", "issue", "sub_product", 
    "sub_issue", "date_received", "sample_id"
).toPandas()

# Drop rows with null narrative
df = df.dropna(subset=["text_clean"])
print(f"Rows after dropping null narratives: {len(df)}")
# %%
spark.stop()
# %%

# ============================================================
# PART B: BERTopic CLUSTERING (with lemmatisation & outlier reduction)
# ============================================================

# ------------------------------
# B1. Prepare documents and metadata
# ------------------------------
docs = df["text_clean"].astype(str).tolist()
timestamps = pd.to_datetime(df["date_received"]).dt.strftime("%Y-%m-%d").tolist()  # for temporal analysis

# ------------------------------
# B2. Lemmatisation helper (to reduce word variants)
# ------------------------------
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    words = text.split()
    return " ".join([lemmatizer.lemmatize(w.lower()) for w in words])

# Apply lemmatisation for the CountVectorizer (but keep original docs for SBERT embeddings)
docs_lemmatized = [lemmatize_text(doc) for doc in docs]

# ------------------------------
# B3. Stop words (NLTK + extended financial boilerplate)
# ------------------------------
EXTRA_STOP_WORDS = {
    "please", "thank", "hello", "dear", "regarding", "however", "therefore", 
    "also", "get", "make", "time", "still", "even", "though", "well", 
    "really", "actually", "basically", "complaint", "complaints", "consumer", 
    "cfpb", "company", "bank", "account", "would", "could", "said", "us", 
    "one", "two", "three", "first", "second", "also", "may", "said", "see",
    "like", "just", "know", "need", "want", "asked", "told", "called", "went"
}
STOP_WORDS = list(set(nltk_stopwords.words('english')).union(EXTRA_STOP_WORDS))

# ------------------------------
# B4. CountVectorizer with lemmatised input
# ------------------------------
vectorizer_model = CountVectorizer(
    stop_words=STOP_WORDS,
    min_df=4,
    max_df=0.85,
    preprocessor=lambda x: x,   # we already lemmatised
    max_features=40000,          # Limit memory
    token_pattern=r'(?u)\b\w+\b',
    ngram_range=(1, 2)
)

# ------------------------------
# B5. UMAP (preserve local structure, cosine distance)
# ------------------------------
umap_model = umap.UMAP(
    n_neighbors=25,
    n_components=12,
    min_dist=0.1,
    metric="cosine",
    low_memory=True,
    random_state=24,
    verbose=True
)

# ------------------------------
# B6. HDBSCAN (tuned for ~30-50 topics directly)
# ------------------------------
# For ~100k docs, min_cluster_size=300 gives ~30-50 clusters.
# Adjust based on your final sample size.
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=80,        # higher = fewer clusters
    min_samples=20,
    metric='euclidean',
    cluster_selection_method='eom',
    cluster_selection_epsilon=0.2,
    prediction_data=True,
    gen_min_span_tree=True,
    core_dist_n_jobs=-1          # Parallel processing for speed
)

# ------------------------------
# B7. Sentence embedding model
# ------------------------------
embedding_model_name = "all-MiniLM-L6-v2"  # good balance speed/quality
embedding_model = SentenceTransformer(embedding_model_name)
if torch.cuda.is_available():
    embedding_model = embedding_model.to('cuda')
    print("Using GPU for embeddings")

# ------------------------------
# B8. Fit BERTopic with pre‑computed embeddings (optional: let BERTopic compute them)
# ------------------------------
print("\nComputing document embeddings (may take a few minutes)...")
embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=256)

# %%
embeddings_bigfin = np.load('doc_embeddings_full.npy')

# Check your embeddings BEFORE fitting
print("=== DIAGNOSTICS ===")
print(f"Embeddings BigFin shape: {embeddings_bigfin.shape}")
print(f"Embeddings dtype: {embeddings_bigfin.dtype}")
print(f"Any NaN: {np.isnan(embeddings_bigfin).any()}")
print(f"Any Inf: {np.isinf(embeddings_bigfin).any()}")
print(f"Embeddings min value: {embeddings_bigfin.min()}")
print(f"Embeddings max value: {embeddings_bigfin.max()}")
print(f"All embeddings identical? {np.all(embeddings_bigfin[0] == embeddings)}")

# Check variance (if all vectors are identical, variance = 0)
variance = np.var(embeddings_bigfin, axis=0)
print(f"Embedding variance range: {variance.min():.2e} to {variance.max():.2e}")
print(f"All zero variance? {np.all(variance < 1e-10)}")

# Check if embeddings are normalized
norms = np.linalg.norm(embeddings, axis=1)
print(f"Embedding norms - min: {norms.min():.4f}, max: {norms.max():.4f}")
print(f"All norms = 1? {np.allclose(norms, 1.0)}")

# %%

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=True,
    verbose=True,

)

print("Fitting BERTopic...")
topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)



# Comprehensive topics check
def diagnose_topics_object(topics):
    """Check what the topics object actually is"""
    print("=== TOPICS DIAGNOSTIC ===")
    print(f"Type: {type(topics)}")
    print(f"Is None? {topics is None}")
    print(f"Is boolean? {isinstance(topics, bool)}")
    print(f"Is list? {isinstance(topics, list)}")
    print(f"Is numpy array? {isinstance(topics, np.ndarray)}")
    
    if hasattr(topics, 'shape'):
        print(f"Shape: {topics.shape}")
    if hasattr(topics, 'dtype'):
        print(f"Dtype: {topics.dtype}")
    if hasattr(topics, '__len__'):
        print(f"Length: {len(topics)}")
    
    # Try to get first few elements safely
    try:
        if len(topics) > 0:
            print(f"First 5 elements: {topics[:5]}")
    except:
        print(f"First element: {topics}")
    
    # Check unique values
    try:
        unique_vals = set(topics)
        print(f"Unique values: {unique_vals}")
        if -1 in unique_vals:
            print(f"Contains noise (-1): Yes")
        print(f"Number of unique topics: {len(unique_vals) - (1 if -1 in unique_vals else 0)}")
    except:
        print("Cannot get unique values")
    
    return topics is not None and not isinstance(topics, bool) and len(topics) > 0

# Use it
is_valid = diagnose_topics_object(topics)

# %%------------------------------
# B8. Initial Topic Modeling Results
# ------------------------------
unique_topics = set(topics)
num_topics = len(unique_topics - {-1})
noise_ratio = (topics == -1).mean()

print("\n" + "="*60)
print("INITIAL TOPIC MODEL RESULTS")
print("="*60)
print(f"Number of topics (excluding outliers): {num_topics}")
print(f"Noise ratio: {noise_ratio:.2%}")
print(f"Total documents: {len(topics):,}")

# Topic size distribution (excluding noise)
topic_sizes = pd.Series([t for t in topics if t != -1]).value_counts()
print(f"\nTopic size statistics:")
print(f"  Min topic size: {topic_sizes.min()}")
print(f"  Max topic size: {topic_sizes.max()}")
print(f"  Mean topic size: {topic_sizes.mean():.0f}")
print(f"  Median topic size: {topic_sizes.median():.0f}")
print(f"  Small topics (<100 docs): {(topic_sizes < 100).sum()}")
print(f"  Large topics (>1000 docs): {(topic_sizes > 1000).sum()}")

# %%------------------------------
# B9. Reduce outliers (without merging topics)
# preserves rare topics better, handles uncertainty, more nuanced
# may keep too much noise and over preserve outliers
# ------------------------------
print("Reducing outliers...")
new_topics = topic_model.reduce_outliers(docs, topics, strategy="distributions", threshold=0.12)
topic_model.update_topics(docs, topics=new_topics)
topics = new_topics
'''
# ------------------------------------------------------------------
# B10. (Optional) If you still have many topics, adjust earlier parameters.
#      Avoid reduce_topics – instead, re-run with larger min_cluster_size.
# ------------------------------------------------------------------
unique_topics = set(topics)
num_topics = len(unique_topics - {-1})
'''
print(f"\nNumber of topics (excluding outliers): {num_topics}")
print(f"Noise ratio: {(topics == -1).mean():.2%}")

# ------------------------------
# B11. Attach topics to dataframe
# ------------------------------
df['topic'] = topics
df['topic_probability'] = np.max(probs, axis=1)

# ------------------------------
# B12. Temporal analysis (if timestamps available)
# ------------------------------
if timestamps and len(timestamps) == len(docs):
    print("\nBuilding topics over time...")
    topics_over_time = topic_model.topics_over_time(
        docs, timestamps, nr_bins=20, datetime_format="%Y-%m-%d"
    )
    fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
    fig.show()

# ------------------------------
# B13. Visualisations
# ------------------------------
print("\nGenerating visualisations...")
topic_model.visualize_topics().show()
topic_model.visualize_barchart(top_n_topics=15).show()
topic_model.visualize_heatmap().show()

# ------------------------------
# B14. Save results
# ------------------------------
output_file = "complaints_clustered.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Saved clustered data to {output_file}")

# Print top topics for inspection
print("\nTop 10 topics by count:")
topic_info = topic_model.get_topic_info()
print(topic_info.head(10).to_string(index=False))


# ============================================================
# END OF CODE
# ============================================================

# %%
