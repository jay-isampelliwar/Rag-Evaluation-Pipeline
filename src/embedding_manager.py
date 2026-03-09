from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):

        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        try:

            if self.model_name == "":
                raise Exception("Model is not loaded.")

            print(f"Initializing model {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model {self.model_name} initialized.")


        except Exception as e:
           print(f"Initializing model {self.model_name} failed. \n {e}")
           raise e

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:

        try:

            if not self.model:
                raise Exception("Model is not loaded.")

            print(f"Generating embeddings for {len(texts)} documents...")
            embeddings = self.model.encode(texts , show_progress_bar=True)
            print(f"Embeddings for {len(texts)} documents generated.")

            return embeddings
        except Exception as e:
            print(f"Generating embeddings for {len(texts)} documents failed. \n {e}")
            raise e