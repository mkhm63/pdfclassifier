import fitz  # PyMuPDF
import os
import gdown
import zipfile
import pandas as pd


def extract_pdfs_to_txt(pdf_filenames=None, input_dir="data/pdfs", output_dir="output/texts"):
    os.makedirs(output_dir, exist_ok=True)
    if pdf_filenames is None:
        pdf_filenames = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    for filename in pdf_filenames:
        pdf_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".pdf", ".txt"))
        with fitz.open(pdf_path) as doc:
            text = "\n".join([page.get_text() for page in doc])
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)


def download_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)


def load_metadata(metadata_path="data/metadata.csv", sample_only=True, sample_size=1000):
    if not os.path.exists(metadata_path):
        file_id = "1css7SaET-MLgubUb_mFz35w5STQ_Oki1"  # Replace with actual Google Drive file ID
        download_from_drive(file_id, metadata_path)
    if os.path.exists(metadata_path):
        if sample_only:
            return pd.read_csv(metadata_path, low_memory=False, nrows=sample_size)
        else:
            return pd.read_csv(metadata_path, low_memory=False)
    return None
