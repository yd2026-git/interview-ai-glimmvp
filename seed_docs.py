# seed_docs.py
#
# Builds the FAISS RAG index for all previous batch interview feedback.
#
# Place your .txt files in: data/feedback_docs/
# Then run: python seed_docs.py
#
# Output:
#   saved_index/faiss.index
#   saved_index/meta.pkl
#

import os
from rag import RAGIndex

FEEDBACK_PATH = "data/feedback_docs"

def load_feedback_txts(folder=FEEDBACK_PATH):
    docs = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            fpath = os.path.join(folder, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                docs.append({
                    "id": fname,
                    "text": text,
                    "meta": {"source": fname}
                })
                print(f"Loaded: {fname}")
            except Exception as e:
                print(f"Failed to load {fname}: {e}")
    return docs


if __name__ == "__main__":
    print("🔄 Building RAG Index...")
    rag = RAGIndex(index_dir="saved_index")

    docs = load_feedback_txts()
    if not docs:
        print("❌ No .txt files found in data/feedback_docs/")
        exit()

    rag.build_from_texts(docs)
    print("\n✅ RAG index successfully built.")
    print("Files saved under: saved_index/")
