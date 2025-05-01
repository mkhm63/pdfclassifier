# app.py (flattened)
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import spacy
import pycountry
import gdown
import zipfile

from extract_text import extract_pdfs_to_txt
from classify import update_labels_csv, load_texts_with_labels, classify_texts
from visualize import load_texts_by_label, plot_wordclouds_by_label
from keyword_barcharts import plot_top_keywords_by_label

st.set_page_config(page_title="COVID Research Classifier", layout="wide")
st.title("COVID Scientific Article Classifier")

if os.path.exists("data/logo.png"):
    st.sidebar.image("data/logo.png", use_container_width=True)
else:
    st.sidebar.warning("⚠️ Logo not found at data/logo.png")

st.sidebar.header("📂 Upload PDF Files")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", accept_multiple_files=True, type=["pdf"])

pdf_dir = "data/pdfs"
os.makedirs(pdf_dir, exist_ok=True)

if uploaded_files:
    max_pdfs = st.sidebar.number_input("Number of PDFs to use", min_value=1, max_value=len(uploaded_files), value=len(uploaded_files))
    selected_files = uploaded_files[:max_pdfs]
    filenames_to_extract = []
    for file in selected_files:
        with open(os.path.join(pdf_dir, file.name), "wb") as f:
            f.write(file.getbuffer())
        filenames_to_extract.append(file.name)
    st.sidebar.success(f"Uploaded {len(selected_files)} file(s)")
else:
    filenames_to_extract = []

if st.sidebar.button("Extract Text from PDFs"):
    if filenames_to_extract:
        extract_pdfs_to_txt(pdf_filenames=filenames_to_extract)
        st.success(f"Text extracted from {len(filenames_to_extract)} uploaded PDF(s).")
    else:
        st.warning("No uploaded PDF files found.")

# Monkey patch classify.update_labels_csv to use spacy and pycountry
import classify

def improved_update_labels_csv(text_dir="output/texts", label_file="output/labels.csv", allowed_filenames=None):
    from spacy import load
    from collections import Counter

    nlp = load("en_core_web_sm")

    country_to_region = {
        "france": "europe", "germany": "europe", "italy": "europe", "spain": "europe", "united kingdom": "europe",
        "china": "asia", "japan": "asia", "india": "asia", "south korea": "asia",
        "nigeria": "africa", "south africa": "africa", "egypt": "africa",
        "united states": "north america", "canada": "north america", "mexico": "north america",
        "brazil": "south america", "argentina": "south america", "chile": "south america",
        "australia": "australia", "new zealand": "australia",
        "united arab emirates": "middle east", "qatar": "middle east"
    }

    country_list = {country.name.lower() for country in pycountry.countries}
    known_regions = {"europe", "asia", "africa", "south america", "north america", "middle east", "australia"}

    all_files = sorted(f for f in os.listdir(text_dir) if f.endswith(".txt"))
    existing = pd.read_csv(label_file) if os.path.exists(label_file) else pd.DataFrame(columns=["filename"])
    processed_files = set(existing["filename"].tolist())

    new_data = []
    for fname in all_files:
        if allowed_filenames and fname not in allowed_filenames:
            continue

        path = os.path.join(text_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().lower()

        if any(term in text for term in ["mortality", "death", "icu"]):
            risk = "high"
        elif any(term in text for term in ["hospital", "infection", "spread"]):
            risk = "medium"
        else:
            risk = "low"

        if any(term in text for term in ["model", "forecast", "simulation"]):
            study = "modeling"
        elif any(term in text for term in ["x-ray", "scan", "detection", "diagnostic"]):
            study = "diagnostic"
        elif "vaccine" in text:
            study = "vaccine"
        else:
            study = "other"

        doc = nlp(text)
        found = [ent.text.lower() for ent in doc.ents if ent.label_ == "GPE"]

        locations = Counter(found) if found else Counter()

        detected_country = "global"
        detected_region = "global"

        if locations:
            for loc, _ in locations.most_common():
                loc = loc.lower()
                if loc in country_list:
                    detected_country = loc
                    detected_region = country_to_region.get(loc, "global")
                    break
                elif loc in known_regions:
                    detected_region = loc
                    break

        new_data.append({
            "filename": fname,
            "risk_level": risk,
            "study_type": study,
            "country": detected_country,
            "region": detected_region
        })

    updated_df = pd.concat([existing, pd.DataFrame(new_data)], ignore_index=True)
    updated_df.to_csv(label_file, index=False)
    print(f"✅ labels.csv updated with {len(new_data)} new entries. Total: {len(updated_df)}")

classify.update_labels_csv = improved_update_labels_csv

st.sidebar.markdown("### 📥 Load CORD-19 Metadata")
load_option = st.sidebar.radio(
    "Download type:",
    ["Sample only", "Full metadata"],
    index=0
)

if st.sidebar.button("Load CORD-19 Metadata", key="load_meta_button"):
    from extract_text import load_metadata
    sample = load_option == "Sample only"
    meta_df = load_metadata(sample_only=sample, sample_size=1000)

    if meta_df is not None:
        st.success(f"{'Sampled' if sample else 'Full'} metadata loaded: {meta_df.shape[0]} rows")
        st.session_state.saved_meta_filenames = [f"meta_{i}.txt" for i in range(len(meta_df))]
        os.makedirs("output/texts", exist_ok=True)
        for idx, row in meta_df.iterrows():
            content = f"{row['title']}\n{row['abstract']}"
            with open(f"output/texts/meta_{idx}.txt", "w", encoding="utf-8") as f:
                f.write(content)
    else:
        st.error("Failed to load metadata.")
        
        
# Allow user to group visualizations by Country or Region
group_by = st.sidebar.radio(
    "Group visualizations by:",
    ["Country", "Region"],
    index=1
)

st.sidebar.markdown("### 🧪 Train Model")
with st.sidebar.expander("🧠 Train Model"):
    data_source = st.radio(
        "Select Data Source for Training",
        ["All", "Only PDFs", "Only CORD-19 Metadata", "Mix: Top N PDFs + Top N Metadata"],
        index=0
    )
    max_meta = st.number_input("Number of metadata entries to load", min_value=10, max_value=700000, value=500, step=100)
    random_seed = st.number_input("Random seed (optional)", min_value=0, value=42, step=1, help="Leave as default or change for reproducible sampling") if st.checkbox("Use random seed") else None
    filter_year = st.text_input("Filter metadata by publish year (e.g., 2020)", "")
    filter_journal = st.text_input("Filter by journal (leave blank for all)", "")
    filter_keyword = st.text_input("Filter by keyword in title/abstract", "")

    if st.button("Load CORD-19 Metadata", key="train_load_meta_button"):
        metadata_path = "data/metadata.csv"
        if os.path.exists(metadata_path):
            st.write("Reading metadata.csv...")
            meta_df = pd.read_csv(metadata_path, low_memory=False)
            st.write("Initial metadata shape:", meta_df.shape)

            if filter_year and "publish_time" in meta_df.columns:
                meta_df = meta_df[meta_df["publish_time"].astype(str).str.startswith(filter_year)]
            if filter_journal and "journal" in meta_df.columns:
                meta_df = meta_df[meta_df["journal"].astype(str).str.contains(filter_journal, case=False, na=False)]
            if filter_keyword:
                meta_df = meta_df[
                    meta_df["title"].astype(str).str.contains(filter_keyword, case=False, na=False) |
                    meta_df["abstract"].astype(str).str.contains(filter_keyword, case=False, na=False)
                ]
            st.write("Filtered metadata shape:", meta_df.shape)

            meta_df = meta_df.dropna(subset=["title", "abstract"])
            if random_seed is not None:
                meta_df = meta_df.sample(n=int(max_meta), random_state=random_seed).reset_index(drop=True)
            else:
                meta_df = meta_df.sample(n=int(max_meta)).reset_index(drop=True)
            os.makedirs("output/texts", exist_ok=True)
            saved_meta_filenames = []
            for idx, row in meta_df.iterrows():
                cord_uid = f"meta_{idx}"
                content = f"{row['title']}\n{row['abstract']}"
                filename = f"{cord_uid}.txt"
                with open(f"output/texts/{filename}", "w", encoding="utf-8") as f:
                    f.write(content)
                saved_meta_filenames.append(filename)
            st.session_state.saved_meta_filenames = saved_meta_filenames
            st.success(f"Processed {len(meta_df)} metadata entries into text files.")
        else:
            st.error("metadata.csv not found in /data")

    if st.button("📊 View Labels Preview"):
        label_file_path = "output/labels.csv"
        if os.path.exists(label_file_path):
            label_df = pd.read_csv(label_file_path)
            st.subheader(f"Labels Table Preview ({len(label_df)} rows)")
            st.dataframe(label_df.head(20))
        else:
            st.error("labels.csv not found in /output")

    if st.button("🗑️ Dump Labels.csv (Empty)"):
        label_file_path = "output/labels.csv"
        pd.DataFrame(columns=["filename", "risk_level", "study_type", "region"]).to_csv(label_file_path, index=False)
        st.warning("labels.csv has been emptied.")

    st.sidebar.markdown("### 🏷 Update Labels")
if st.sidebar.button("Update Labels Automatically", key="move_update_labels_button"):
    allowed_filenames = set()
    if 'saved_meta_filenames' in st.session_state:
        allowed_filenames.update(st.session_state.saved_meta_filenames)
    if uploaded_files:
        allowed_filenames.update(file.name.replace(".pdf", ".txt") for file in uploaded_files)
    update_labels_csv(allowed_filenames=allowed_filenames)
    st.success(f"Updated labels for {len(allowed_filenames)} files.")

selected_label_columns = st.sidebar.multiselect(
        "Choose Label Columns for Training",
        ["risk_level", "study_type", "region"],
        default=["risk_level"]
    )

train_trigger = st.sidebar.button("🤖 Train Classifier & Visualize")

if train_trigger:
    progress = st.progress(0, text="Loading labels...")
    try:
        labels_df = pd.read_csv("output/labels.csv")
        progress.progress(10, text="Filtering dataset...")

        if data_source == "Only PDFs" and uploaded_files:
            uploaded_names = {file.name.replace(".pdf", ".txt") for file in uploaded_files}
            new_files = labels_df[labels_df['filename'].isin(uploaded_names)]
        elif data_source == "Only CORD-19 Metadata":
            if "saved_meta_filenames" in st.session_state:
                meta_filenames = set(st.session_state.saved_meta_filenames)
                new_files = labels_df[labels_df['filename'].isin(meta_filenames)]
            else:
                st.warning("No metadata files loaded in this session. Please click 'Load CORD-19 Metadata' first.")
                new_files = pd.DataFrame(columns=labels_df.columns)
        elif data_source == "Mix: Top N PDFs + Top N Metadata":
            max_pdfs = st.sidebar.number_input("Max PDFs to use (for training)", min_value=1, value=5)
            max_meta = st.sidebar.number_input("Max metadata to use (for training)", min_value=1, value=5)
            uploaded_names = set()
            if uploaded_files:
                uploaded_names = {file.name.replace(".pdf", ".txt") for file in uploaded_files[:max_pdfs]}
            meta_filenames = set()
            if "saved_meta_filenames" in st.session_state:
                meta_filenames = set(st.session_state.saved_meta_filenames[:max_meta])
            combined = uploaded_names.union(meta_filenames)
            new_files = labels_df[labels_df['filename'].isin(combined)]
        else:
            if "saved_meta_filenames" in st.session_state and uploaded_files:
                uploaded_names = {file.name.replace(".pdf", ".txt") for file in uploaded_files}
                meta_filenames = set(st.session_state.saved_meta_filenames)
                combined = uploaded_names.union(meta_filenames)
                new_files = labels_df[labels_df['filename'].isin(combined)]
            elif "saved_meta_filenames" in st.session_state:
                meta_filenames = set(st.session_state.saved_meta_filenames)
                new_files = labels_df[labels_df['filename'].isin(meta_filenames)]
            elif uploaded_files:
                uploaded_names = {file.name.replace(".pdf", ".txt") for file in uploaded_files}
                new_files = labels_df[labels_df['filename'].isin(uploaded_names)]
            else:
                st.warning("No recent metadata or PDF uploads detected. Please load data before training.")
                new_files = pd.DataFrame(columns=labels_df.columns)

        texts = []
        y = []
        valid_filenames = set()
        if "saved_meta_filenames" in st.session_state:
            valid_filenames.update(st.session_state.saved_meta_filenames)
        if uploaded_files:
            valid_filenames.update(file.name.replace(".pdf", ".txt") for file in uploaded_files)

        filtered_files = new_files[new_files['filename'].isin(valid_filenames)]

        for i, (_, row) in enumerate(filtered_files.iterrows()):
            file_path = os.path.join("output/texts", row["filename"])
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    texts.append((row["filename"], f.read()))  # Save (filename, text) tuple
                    if group_by == "Region":
                        label = " | ".join(str(row[col]) for col in selected_label_columns if col != "country") + " | " + row["region"]
                    else:
                        label = " | ".join(str(row[col]) for col in selected_label_columns if col != "region") + " | " + row["country"]
                    y.append(label)
            if i % 10 == 0:
                progress.progress(min(50, int((i / len(filtered_files)) * 50)), text=f"Loading texts... ({i}/{len(filtered_files)})")

        st.write(f"Loaded {len(texts)} newly labeled documents for: {', '.join(selected_label_columns)}")
        if not texts:
            st.warning("No new texts found. Please upload or load something first.")
            st.stop()

        progress.progress(60, text="Training model...")

        if all(text.strip() == '' for _, text in texts):
            st.error("All loaded texts are empty or contain only stopwords. Please check your dataset.")
            st.stop()

        # Pass only text (not filenames) into the classifier
        clf, vectorizer = classify_texts([text for _, text in texts], y)

        progress.progress(80, text="Organizing label groups...")
        label_texts = {label: [] for label in set(y)}
        for (filename, text), label in zip(texts, y):
            label_texts[label].append((filename, text))  # Keep filenames for visualization

        st.session_state.label_texts = label_texts
        st.session_state.classifier_ready = True
        progress.progress(100, text="✅ Done! Ready to visualize.")
        st.success("Model trained. Now select a label to visualize.")
    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")

st.sidebar.markdown("---")

# Initialize session state
if 'classifier_ready' not in st.session_state:
    st.session_state.classifier_ready = False
if 'label_texts' not in st.session_state:
    st.session_state.label_texts = {}

# Show visualizations after classifier is ready
if st.session_state.classifier_ready and st.session_state.label_texts:
    detected_labels = list(st.session_state.label_texts.keys())
    selected_label_value = st.selectbox("Select a combined label to visualize:", detected_labels)
    filtered_texts = {selected_label_value: st.session_state.label_texts[selected_label_value]}

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Top Keywords for {selected_label_value}")

        from sklearn.feature_extraction.text import TfidfVectorizer

        # Unpack only texts (not filenames) for TF-IDF
        texts_flat = [" ".join(text for _, text in docs) for label, docs in filtered_texts.items()]

        vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
        X = vectorizer.fit_transform(texts_flat)

        feature_names = vectorizer.get_feature_names_out()
        sums = X.sum(axis=0)
        data = []

        for col_idx, term in enumerate(feature_names):
            data.append((term, sums[0, col_idx]))

        ranking = sorted(data, key=lambda x: x[1], reverse=True)
        keywords, scores = zip(*ranking[:30])  # first 30 to have choice

        # ===== Filter keywords (no short words, no numbers)
        banlist = {
            "https", "http", "com", "org", "www", "pdf", "doi", "pmid", "pmc", "license",
          "article", "preprint", "medrxiv", "biorxiv", "creativecommons",
          "et", "al", "fig", "figure", "table", "supplementary", "preprints",
          "study", "studies", "result", "results", "data", "author", "authors",
          "method", "methods", "analysis", "analyses", "report", "review", "journal",
          "introduction", "discussion", "conclusion", "abstract", "background",
          "sars", "cov", "covid", "covid19", "coronavirus", "infection", "pandemic",
          "day", "week", "month", "year", "january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december",
          "ml", "mg", "kg", "cm", "mm", "vs", "e.g", "i.e", "etc", "hr", "hrs",
          "baseline", "number", "value", "values", "level", "levels", "type", "types",
          "high", "low", "group", "groups", "included", "sample", "samples", "baseline"
          }

        filtered_keywords = [
            word for word in keywords
            if word.isalpha() and len(word) > 2 and word.lower() not in banlist
        ][:10]

        plt.barh(filtered_keywords, scores[:len(filtered_keywords)])
        plt.xlabel("Importance")
        plt.gca().invert_yaxis()
        st.pyplot(plt.gcf())

    with col2:
        st.subheader(f"Word Cloud for {selected_label_value}")
        plt.figure(figsize=(12, 6))
        plot_wordclouds_by_label(filtered_texts)
        st.pyplot(plt.gcf())

    # ===== NEW SECTION: Track where top keywords appear =====
    st.markdown("---")
    st.header("🔍 Keyword Matches and Text Previews")

    selected_keywords = st.multiselect(
        "Select keywords to display matches for:",
        filtered_keywords,
        default=list(filtered_keywords)
    )

    keyword_matches = {keyword: [] for keyword in filtered_keywords}

    for filename, text in filtered_texts[selected_label_value]:
        for keyword in filtered_keywords:
            if keyword.lower() in text.lower():
                start = text.lower().find(keyword.lower())
                snippet = text[max(0, start-100):start+400]  # Longer preview
                keyword_matches[keyword].append((filename, snippet))

    # Display matches for selected keywords
    for keyword in selected_keywords:
        matches = keyword_matches.get(keyword, [])
        if matches:
            st.subheader(f"🔵 Keyword: **{keyword}**")
            for filename, snippet in matches:
                highlighted_snippet = snippet.replace(
                    keyword.lower(), f"**{keyword.upper()}**"
                )
                display_snippet = ""
                if not snippet.startswith(" "):
                    display_snippet += "..."
                display_snippet += highlighted_snippet
                if not snippet.endswith(" "):
                    display_snippet += "..."
                st.markdown(f"**Document `{filename}`:** {display_snippet}")

    # ===== Optional: Export to CSV
    import pandas as pd
    import io

    if keyword_matches:
        export_data = []
        for keyword, matches in keyword_matches.items():
            for filename, snippet in matches:
                export_data.append({
                    "Keyword": keyword,
                    "Document": filename,
                    "Snippet": snippet
                })

        if export_data:
            df_export = pd.DataFrame(export_data)
            csv_buffer = io.StringIO()
            df_export.to_csv(csv_buffer, index=False)

            st.download_button(
                label="📥 Download Matches as CSV",
                data=csv_buffer.getvalue(),
                file_name="keyword_matches.csv",
                mime="text/csv"
            )

def download_from_drive(file_id, output_path):
    if os.path.exists(output_path):
        print(f"✅ {output_path} already exists. Skipping download.")
        return
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print(f"✅ Downloaded: {output_path}")
    except Exception as e:
        print(f"⚠️ Failed to download {output_path}: {e}")
        st.warning(f"⚠️ Failed to download {output_path}. Please upload manually if needed.")


# Replace with your actual Drive file IDs
download_from_drive("1css7SaET-MLgubUb_mFz35w5STQ_Oki1", "data/metadata.csv")
download_from_drive("1FYwV0xC4EZPd-9cHOyuBUW5cUCCLlxC4", "output/texts.zip")

# Extract the PDFs if not already extracted
if not os.path.exists("data/pdfs"):
    with zipfile.ZipFile("data/pdfs.zip", 'r') as zip_ref:
        zip_ref.extractall("data/pdfs")
    print("📂 Extracted PDFs")
