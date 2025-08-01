from typing import Dict, List
from chunking import SemanticChunker
from extractor import extract_text_from_resume
from name_extractor import extract_name


def process_resume(file_path):
    """
    Process resume and generate semantic chunks

    Args:
        file_path (str): Path to resume file

    Returns:
        List[Dict[str, str]]: Semantic chunks of resume
    """
    # Extract text from resume
    resume_text = extract_text_from_resume(file_path)

    # Initialize semantic chunker
    chunker = SemanticChunker()

    # Generate semantic chunks
    name = extract_name(file_path)
    semantic_chunks = chunker.chunk_cv_with_metadata(resume_text, name)


    return semantic_chunks, name

def prepare_for_vector_db(semantic_chunks: List[Dict], candidate_name: str) -> List[Dict]:
    """Prepare chunks for vector DB with candidate name"""
    chunker = SemanticChunker()
    return [{
        "id": f"{candidate_name}_{hash(chunk['text'])}",
        "text": chunk['text'],
        "embedding": chunker.model.encode(chunk['text']).tolist(),
        "metadata": {
            "candidate_name": candidate_name,
            "start_index": chunk['metadata']['start_index'],
            "end_index": chunk['metadata']['end_index']
        }
    } for chunk in semantic_chunks]