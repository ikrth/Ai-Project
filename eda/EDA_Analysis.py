# ===== portable_eda.py =====

# ===== Libraries =====
import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from collections import Counter

# ===== Paths =====
ZIP_FILE = "dataset.zip"   # <-- Change this to your ZIP file name
DATASET_FOLDER = "dataset"
PLOTS_FOLDER = "plots"

# ===== Create folders =====
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# ===== Extract ZIP =====
with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
    zip_ref.extractall(DATASET_FOLDER)

print("Files in dataset folder:", os.listdir(DATASET_FOLDER))

# ===== Find TSV files =====
tsv_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith('.tsv')]
print("TSV files found:", tsv_files)

# ===== Helper function: get language from filename =====
def get_language(tsv_name):
    if "-en" in tsv_name:
        return 'en'
    elif "-es" in tsv_name:
        return 'es'
    elif "-zh" in tsv_name:
        return 'zh'
    else:
        return 'en'  # default

# ===== Load SpaCy models =====
def load_spacy_model(lang):
    model_map = {
        'en': "en_core_web_sm",
        'es': "es_core_news_sm",
        'zh': "zh_core_web_sm"
    }
    model_name = model_map.get(lang, "en_core_web_sm")
    try:
        nlp = spacy.load(model_name)
    except:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", model_name])
        nlp = spacy.load(model_name)
    return nlp

# ===== Process each TSV file =====
for tsv_file in tsv_files:
    print(f"\n=== Processing {tsv_file} ===")
    
    # Load TSV
    df = pd.read_csv(os.path.join(DATASET_FOLDER, tsv_file), sep='\t')
    print(f"Number of entries: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Text columns
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Text columns detected: {text_columns}")
    
    # Sample rows
    print("Sample rows:")
    print(df.head(3).to_string())
    
    # Check duplicate URLs if 'url' column exists
    if 'url' in df.columns:
        dup_urls = df['url'].duplicated().sum()
        print(f"Duplicate URLs: {dup_urls}")
    
    # Select SpaCy model based on language
    lang = get_language(tsv_file)
    nlp = load_spacy_model(lang)
    
    # ===== Text EDA =====
    for col in text_columns:
        # Skip columns with all '-' as in word1/word2
        if (df[col] == '-').all():
            continue
        
        print(f"\nEDA for text column: {col}")
        
        # Tokenize using SpaCy
        df['token_count'] = df[col].apply(lambda x: len([t.text for t in nlp(str(x))]) if str(x) != '-' else 0)
        
        # Token count histogram
        plt.figure(figsize=(8,4))
        sns.histplot(df['token_count'], bins=30, kde=True)
        plt.title(f"Token Count Distribution ({col})")
        plt.xlabel("Number of Tokens")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(PLOTS_FOLDER, f"token_count_{tsv_file}_{col}.pdf"))
        plt.close()
        
        # Top 20 words
        tokens = [t.text.lower() for text in df[col] for t in nlp(str(text)) if str(text) != '-']
        token_counts = Counter(tokens)
        top_words = dict(token_counts.most_common(20))
        plt.figure(figsize=(10,5))
        sns.barplot(x=list(top_words.keys()), y=list(top_words.values()))
        plt.xticks(rotation=45)
        plt.title(f"Top 20 Words ({col})")
        plt.savefig(os.path.join(PLOTS_FOLDER, f"top20_words_{tsv_file}_{col}.pdf"))
        plt.close()
    
    # ===== Label/Category EDA =====
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    label_cols = [c for c in categorical_cols if c not in text_columns]
    for col in label_cols:
        print(f"\nEDA for label column: {col}")
        plt.figure(figsize=(6,4))
        sns.countplot(x=col, data=df)
        plt.title(f"Label Distribution ({col})")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.savefig(os.path.join(PLOTS_FOLDER, f"label_distribution_{tsv_file}_{col}.pdf"))
        plt.close()
    
    # ===== Special plot for URLs =====
    if 'url' in df.columns:
        df['domain'] = df['url'].apply(lambda x: x.split('/')[2])
        plt.figure(figsize=(8,4))
        sns.countplot(y='domain', data=df, order=df['domain'].value_counts().index)
        plt.title(f"Image URL Source Distribution ({tsv_file})")
        plt.xlabel("Count")
        plt.ylabel("Domain")
        plt.savefig(os.path.join(PLOTS_FOLDER, f"url_source_distribution_{tsv_file}.pdf"))
        plt.close()

print("\nAll plots saved in folder:", PLOTS_FOLDER)
