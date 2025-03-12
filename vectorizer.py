import os
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def split_text_into_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Split a text into overlapping chunks of specified size.
    
    Args:
        text: The input text to split
        chunk_size: Size of each chunk in words
        overlap: Number of overlapping words between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
            
    return chunks

def vectorize_text(text: str) -> Dict:
    """
    Vectorize text for RAG by splitting into chunks and creating embeddings.
    
    Args:
        text: The input text to vectorize
        
    Returns:
        Dictionary containing text chunks and their vector embeddings
    """
    if not text:
        return {"chunks": [], "embeddings": []}
    
    # Split text into chunks
    chunks = split_text_into_chunks(text)
    
    # Create embeddings for each chunk
    embeddings = model.encode(chunks, convert_to_tensor=True)
    
    # Convert to numpy arrays for easier handling
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    
    return {
        "chunks": chunks,
        "embeddings": embeddings
    }

def find_similar_chunks(query: str, vectorized_data: Dict, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Find chunks most similar to the query using cosine similarity.
    
    Args:
        query: The query text to find similar chunks for
        vectorized_data: Dictionary with chunks and embeddings
        top_k: Number of top similar chunks to return
        
    Returns:
        List of tuples containing (chunk_text, similarity_score)
    """
    if not vectorized_data["chunks"]:
        return []
    
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    if torch.is_tensor(query_embedding):
        query_embedding = query_embedding.cpu().numpy()
    
    # Calculate similarity scores
    similarities = []
    for i, chunk_embedding in enumerate(vectorized_data["embeddings"]):
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
        )
        similarities.append((vectorized_data["chunks"][i], float(similarity)))
    
    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k similar chunks
    return similarities[:top_k]

def get_rag_context(text: str, query: str = None) -> str:
    """
    Get RAG context for story extension.
    
    Args:
        text: The original story text
        query: Query to use for finding similar chunks (defaults to last chunk of text)
        
    Returns:
        Context string to use for story extension
    """
    vectorized_data = vectorize_text(text)
    
    if not query and vectorized_data["chunks"]:
        # Use the last chunk as query if none provided
        query = vectorized_data["chunks"][-1]
    
    # Find similar chunks
    similar_chunks = find_similar_chunks(query, vectorized_data)
    
    # Combine similar chunks into context
    context = "\n\n".join([chunk for chunk, _ in similar_chunks])
    
    return context
