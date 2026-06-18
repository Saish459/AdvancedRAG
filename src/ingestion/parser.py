from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import nest_asyncio
from llama_parse import LlamaParse
from src.config import Config



nest_asyncio.apply()


def parse_pdf(file_path: str):
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    cache_path = os.path.join(os.path.dirname(file_path), f"{name_without_ext}.md")

    if os.path.exists(cache_path):
        print(f"Loading '{base_name}' from local cache...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    print(f"Parsing '{base_name}' via LlamaParse ...")

    parser = LlamaParse(
        api_key=Config.LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        verbose=True,
        language="en",
    )

    documents = parser.load_data(file_path)
    full_markdown = "\n\n".join([doc.text for doc in documents])


    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)
        print(f"Saved cached markdown to: {cache_path}")

    return full_markdown


def load_data(data_dir: str):
 
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in data/ folder.")
        return []

    parsed_docs = []
    
    for pdf_file in pdf_files:
        full_path = os.path.join(data_dir, pdf_file)
        try:
            markdown_text = parse_pdf(full_path)
            parsed_docs.append({
                "filename": pdf_file,
                "text": markdown_text
            })
        except Exception as e:
            print(f"Error parsing {pdf_file}: {e}")

    return parsed_docs


if __name__ == "__main__":
    print("Parsing Documents ...")
    docs = load_data(Config.DATA_DIR)
    print(f"\nSuccessfully parsed {len(docs)} documents.")