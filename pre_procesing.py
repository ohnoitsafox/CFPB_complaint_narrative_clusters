#%%
# Diagnostics in case 
print("=== DIAGNOSTIC ===")

# 1. Check PySpark installation
try:
    import pyspark
    print(f"✓ PySpark version: {pyspark.__version__}")
    print(f"  PySpark location: {pyspark.__file__}")
except ImportError as e:
    print(f"✗ PySpark import failed: {e}")
    print("  Run: pip install pyspark")
    exit()

# 2. Check Java availability
import subprocess
import sys

try:
    # Try to find Java
    if sys.platform == "win32":
        result = subprocess.run(['where', 'java'], capture_output=True, text=True)
    else:
        result = subprocess.run(['which', 'java'], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Java found: {result.stdout.strip()}")
        # Check Java version
        version_result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        print(f"  Java version: {version_result.stderr.splitlines()[0]}")
    else:
        print("✗ Java not found in PATH")
        print("  Please install Java 8, 11, or 17 from https://adoptium.net/")
except Exception as e:
    print(f"✗ Error checking Java: {e}")

# 3. Check Spark environment
import os
spark_home = os.environ.get('SPARK_HOME')
if spark_home:
    print(f"✓ SPARK_HOME set: {spark_home}")
else:
    print("⚠ SPARK_HOME not set (PySpark should work without it)")

# 4. Test minimal SparkContext
print("\n=== TESTING MINIMAL SPARKCONTEXT ===")
try:
    from pyspark import SparkContext
    sc = SparkContext.getOrCreate()
    print("✓ SparkContext created successfully!")
    print(f"  Master: {sc.master}")
    print(f"  App name: {sc.appName}")
    sc.stop()
except Exception as e:
    print(f"✗ Failed to create SparkContext: {e}")
    print(f"  Error details: {type(e).__name__}: {str(e)}")
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")


import sys
print(sys.executable)
import os
import pyspark
import pip
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.sql.functions import col, sum, to_date, regexp_replace, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, StringType
import re
import numpy as np


# Simple Spark session setup with some optimizations for local CPU usage, for a 5k sample and a bigger model in the future
spark = SparkSession.builder \
    .appName("CustomerSupportAnalysis") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

print(f"✅ Spark {spark.version} running on CPU")
print(f"Spark Home: {os.environ.get('SPARK_HOME')}")
# read in the file
df = spark.read.csv("complaints.csv", header=True, multiLine=True, quote='"', escape='"', inferSchema=False)

# snake_case helper function for column names
def to_snake_case(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

# standardize column names to snake_case for easier access in code
df = df.toDF(*[to_snake_case(c) for c in df.columns])

# Delete unnecessary columns
columns_to_drop = ["zip_code","consumer_consent_provided?", "tags", "consumer_disputed?", "submitted_via","date_sent_to_company", "timely_response?"]
df = df.drop(*columns_to_drop)

# remove the instances where consumer_complaint_narrative is null, since it's the most important column for our analysis
df = df.filter(col("consumer_complaint_narrative").isNotNull())

# Convert the date so it's not a string anymore
df = df.withColumn("date_received", to_date(col("date_received"), "yyyy-MM-dd"))

# Handling redaction placeholders and normalizing numbers in the text column
def clean_cfpb_text_col(col: F.Column, normalize_numbers: bool = True) -> F.Column:
    c = F.regexp_replace(col, r"\u00A0", " ")         # NBSP -> space
    c = F.regexp_replace(c, r"\s+", " ")              # collapse whitespace

    # ---- remove redaction placeholders ----
    c = F.regexp_replace(c, r"\bX+\b", " ")
    c = F.regexp_replace(c, r"\*{2,}", " ")
    c = F.regexp_replace(c, r"\b(?:X{2,}[-/])+(?:X{2,})\b", " ")
    c = F.regexp_replace(c, r"(?i)\[(redacted|removed)\]|\(redacted\)", " ")

    # normalizing numbers to reduce noise for topic level understanding 
    if normalize_numbers:
# Protect Legal Citations (e.g., 15 USC 1681, Section 604)
        # We replace the space with an underscore so the number isn't 'naked'
        c = F.regexp_replace(c, r"(?i)\b(section|act|usc|u\.s\.c\.|part)\b\s*(\d+)", "$1_$2")
        
        # 2. Normalize all remaining 'naked' numbers
        c = F.regexp_replace(c, r"\b\d+(?:[.,]\d+)?\b", " <NUM> ")
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)

# Apply text cleaning to the consumer_complaint_narrative column
TEXT_COL = "consumer_complaint_narrative"
df = df.withColumn(
    "text_clean",
    clean_cfpb_text_col(F.coalesce(F.col(TEXT_COL), F.lit("")), normalize_numbers=True)
)

# Add a unique ID column
df = df.withColumn("sample_id", monotonically_increasing_id())

# Step 1: Sample the data safely

try:
    # Use DataFrame sampling
    sampled_df = (
        df
        .filter(F.col("text_clean").isNotNull())
        .filter(F.trim(F.col("text_clean")) != "")
        .select(
            "text_clean",
            "product",
            "sub_product",
            "issue",
            "sub_issue",
            "date_received",
            "sample_id"
        )
        .sample(False, 0.05, seed=256).limit(5000)  # Cap at 5000 rows 

    )
    
    # Collect to driver
    sample_rows = sampled_df.collect()
    print(f"Successfully collected {len(sample_rows)} rows")
    
    # Convert to list of dictionaries
    sample_list = []
    for row in sample_rows:
        sample_list.append({
            "text": row["text_clean"],
            "product": row["product"],
            "sub_product": row["sub_product"],
            "issue": row["issue"],
            "sub_issue": row["sub_issue"],
            "date_received": row["date_received"],
            "sample_id": row["sample_id"]
        })
    
    # Extract documents
    docs = [row["text"] for row in sample_list]
    print(f"Created {len(docs)} documents for analysis")
    
except Exception as e:
    print(f"Error during sampling: {e}")

print(type(docs), len(docs))
# %%
OUTPUT_PATH = "ID_sampled_complaints.csv"
# Convert to Pandas (sample is small) and save
pandas_df = sampled_df.toPandas()
pandas_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {OUTPUT_PATH} with columns: {list(pandas_df.columns)}")

# %%
# Stop the Spark session when done
spark.stop()

# %%
