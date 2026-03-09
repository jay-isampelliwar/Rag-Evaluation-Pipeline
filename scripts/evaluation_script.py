import os
import glob
import json
from typing import List


from src.retriever import Retriever
from src.chroma_db import ChromaDatabase
from src.embedding_manager import EmbeddingManager
from rag_pipeline import RagPipeline
from dotenv import load_dotenv

load_dotenv()

DEFAULT_EVAL_DATASET = "data/evaluation/evaluation_dataset.json"
CHROMA_COLLECTION=os.getenv("CHROMA_COLLECTION")

def main():

    vector_store = ChromaDatabase(collection_name=CHROMA_COLLECTION)
    embedding_manager = EmbeddingManager()
    retriever = Retriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
    )

    pipeline = RagPipeline(
        retriever=retriever,
    )

    print(f"📊 Evaluating using questions from: {DEFAULT_EVAL_DATASET}")
    with open(DEFAULT_EVAL_DATASET, "r") as file:
        questions = json.load(file)
    pipeline.execute(questions)




def get_files_in_directory(source_path: str) -> List[str]:
    if os.path.isfile(source_path):
        return [source_path]
    return glob.glob(os.path.join(source_path, "*"))


if __name__ == "__main__":
    main()