import re
from typing import List, Dict, Tuple
from name_extractor import extract_name
from sentence_transformers import SentenceTransformer
import numpy as np
from extractor import extract_text_from_resume
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

class SemanticChunker:
    def __init__(self,
                 model_name='all-MiniLM-L6-v2',
                 max_chunk_size=500,
                 overlap_size=100):
        """
        Initialize semantic chunker with configurable parameters

        Args:
            model_name (str): Sentence transformer model for embeddings
            max_chunk_size (int): Maximum characters per chunk
            overlap_size (int): Number of characters to overlap between chunks
        """
        self.model = SentenceTransformer(model_name)
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    def _preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text

        Args:
            text (str): Input text to preprocess

        Returns:
            str: Cleaned text
        """
        # Remove extra whitespaces, normalize newlines
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK

        Args:
            text (str): Input text to split

        Returns:
            List[str]: List of sentences
        """
        return nltk.sent_tokenize(text)

    def _semantic_clustering(self, sentences: List[str], threshold: float = 0.75) -> List[List[str]]:
        """
        Cluster sentences semantically based on embedding similarity

        Args:
            sentences (List[str]): List of sentences to cluster
            threshold (float): Cosine similarity threshold for clustering

        Returns:
            List[List[str]]: Grouped sentences with semantic similarity
        """
        if not sentences:
            return []

        # Get sentence embeddings
        embeddings = self.model.encode(sentences)

        # Initial clusters
        clusters = [[sentences[0]]]

        for i in range(1, len(sentences)):
            current_sentence = sentences[i]
            added_to_cluster = False

            for cluster in clusters:
                # Check similarity with cluster representatives
                cluster_embedding = self.model.encode(cluster)
                similarities = np.dot(self.model.encode(current_sentence), cluster_embedding.T)

                if np.max(similarities) >= threshold:
                    cluster.append(current_sentence)
                    added_to_cluster = True
                    break

            if not added_to_cluster:
                clusters.append([current_sentence])

        return clusters

    def chunk_cv(self, text: str) -> List[Dict[str, str]]:
        """
        Create semantic chunks from CV text

        Args:
            text (str): Full CV text

        Returns:
            List[Dict[str, str]]: List of semantic chunks with metadata
        """
        # Preprocess text
        cleaned_text = self._preprocess_text(text)

        # Split into sentences
        sentences = self._split_into_sentences(cleaned_text)

        # Semantic clustering
        sentence_clusters = self._semantic_clustering(sentences)

        # Create chunks from clusters
        chunks = []
        current_chunk = ""
        current_chunk_start = 0

        for cluster in sentence_clusters:
            cluster_text = " ".join(cluster)

            # Check if adding this cluster exceeds max chunk size
            if len(current_chunk) + len(cluster_text) > self.max_chunk_size:
                # Create chunk
                chunks.append({
                    "text": current_chunk.strip(),
                    "start_index": current_chunk_start,
                    "end_index": current_chunk_start + len(current_chunk)
                })

                # Reset for new chunk with overlap
                current_chunk = current_chunk[-self.overlap_size:] + cluster_text
                current_chunk_start = current_chunk_start + len(current_chunk) - len(current_chunk)
            else:
                current_chunk += " " + cluster_text

        # Add final chunk if not empty
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "start_index": current_chunk_start,
                "end_index": current_chunk_start + len(current_chunk)
            })

        return chunks

    def chunk_cv_with_metadata(self, text: str, name: str) -> Tuple[List[Dict], str]:
        """Chunks CV text and extracts candidate name"""

        chunks = self.chunk_cv(text)

        # Add name to chunk metadata
        for chunk in chunks:
            chunk['metadata'] = {
                'candidate_name': name,
                'start_index': chunk['start_index'],
                'end_index': chunk['end_index']
            }

        return chunks