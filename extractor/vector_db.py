import uuid
from typing import List, Optional, Union
from pathlib import Path
from tika import parser
import chromadb
from sentence_transformers import SentenceTransformer
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    pass

class PDFParsingError(DocumentProcessingError):
    pass

class ExtractionError(DocumentProcessingError):
    pass


class VectorDB:
    """
    Handles vectorization and storage of text chunks with improved semantic search.
    """
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.client = None
        self.collection = None
        self.batch_size = 32

    def initialize(self, chunks: List[str]) -> None:
        embeddings = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True 
            )
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings)
        
        self.client = chromadb.Client()
        collection_name = f"doc_{uuid.uuid4().hex[:8]}"
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,  # Increased accuracy during index construction
                "hnsw:search_ef": 100,  # Increased accuracy during search
                "hnsw:M": 62  # Increased connections per node
            }
        )
        
        ids = [str(i) for i in range(len(chunks))]
        metadata_list = [{"text": chunk, "relevance_score": 0.0} for chunk in chunks]
        
        for i in range(0, len(chunks), self.batch_size):
            end_idx = min(i + self.batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:end_idx],
                documents=chunks[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                metadatas=metadata_list[i:end_idx]
            )
        
        logger.info(f"Vector DB created with collection '{collection_name}' and {len(chunks)} chunks.")

    def search(self, query: str, n_results: int = 10, context_window: int = 2) -> List[str]:
        if not self.collection:
            logger.error("Vector search attempted without an initialized collection.")
            return []

        query_embedding = self.embedding_model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        expanded_n_results = min(n_results * (2 * context_window + 1), self.collection.count())
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=expanded_n_results
        )
        
        docs = results.get('documents', [])
        distances = results.get('distances', [])
        ids = results.get('ids', [])
        
        if docs and docs[0]:
            combined_results = []
            for doc, score, doc_id in zip(docs[0], distances[0], ids[0]):
                relevance_score = self._calculate_relevance_score(query, doc)
                combined_score = 0.7 * (1 - score) + 0.3 * relevance_score
                combined_results.append((doc, combined_score, int(doc_id)))
            combined_results.sort(key=lambda x: x[1], reverse=True)

            contextualized_results = []
            seen_ids = set()
            
            for doc, score, doc_id in combined_results[:n_results]:
                context_docs = []
                for i in range(max(0, doc_id - context_window), doc_id):
                    if str(i) not in seen_ids:
                        context_docs.append(self.collection.get(ids=[str(i)])['documents'][0])
                        seen_ids.add(str(i))
 
                if str(doc_id) not in seen_ids:
                    context_docs.append(doc)
                    seen_ids.add(str(doc_id))
 
                for i in range(doc_id + 1, min(doc_id + context_window + 1, self.collection.count())):
                    if str(i) not in seen_ids:
                        context_docs.append(self.collection.get(ids=[str(i)])['documents'][0])
                        seen_ids.add(str(i))
                
                contextualized_results.append(" ".join(context_docs))
            
            logger.info(f"Vector search returned {len(contextualized_results)} results with context windows.")
            return contextualized_results
            
        logger.info("Vector search returned no results.")
        return []

    def _calculate_relevance_score(self, query: str, document: str) -> float:
        query_terms = set(query.lower().split())
        doc_terms = document.lower().split()
        
        score = 0.0
        for i, term in enumerate(doc_terms):
            if term in query_terms:
                score += 1.0 / (i + 1)
        
        return score / len(query_terms) if query_terms else 0.0


class DocumentProcessor:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 1000):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunk_size = chunk_size
        self.vector_db = VectorDB(self.embedding_model)
        self.max_workers = 4
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def parse_pdf(self, file_path: Union[str, Path]) -> str:
        try:
            parsed = parser.from_file(str(file_path))
            content = parsed.get('content', '').strip()
            if not content:
                raise PDFParsingError("No content extracted from PDF")

            return content
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            raise PDFParsingError(f"Failed to parse PDF: {str(e)}") from e

    def chunk_text(self, text: str, chunk_size: Optional[int] = None) -> List[str]:
        if chunk_size:
            self.text_splitter.chunk_size = chunk_size
        
        chunks = self.text_splitter.split_text(text)
        filtered_chunks = [chunk for chunk in chunks if self._is_meaningful_chunk(chunk)]
        
        logger.info(f"Text split into {len(filtered_chunks)} meaningful chunks.")
        return filtered_chunks

    def _is_meaningful_chunk(self, chunk: str) -> bool:
        if len(chunk.split()) < 3:
            return False
 
        financial_indicators = r'\b(balance|total|amount|payment|transfer|deposit|withdrawal|ticker|symbol|shares|price)\b'
        if re.search(financial_indicators, chunk, re.IGNORECASE):
            return True

        if re.search(r'\d+', chunk):
            return True
            
        return False

    def process_document(self, file_path: Union[str, Path]) -> None:
        content = self.parse_pdf(file_path)
        chunks = self.chunk_text(content)
        self.vector_db.initialize(chunks)
        logger.info("Document processing complete: vector DB is populated with cleaned financial data.")
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        results = self.vector_db.search(query, top_k)
        filtered_results = []
        for result in results:
            if result:
                word_count = len(re.findall(r'\b[A-Za-z]+\b', result))
                digit_count = len(re.findall(r'\d', result))
                if digit_count > 0 and word_count/digit_count < 5:
                    filtered_results.append(result)
        
        return filtered_results


if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_document(r"C:\Users\Albia\Desktop\Aimleap\pdf_extraction\pydantic_agents_flow\input_files\JPM - x1004 - Statement (1).pdf")
    results = processor.search("what are the ticker values?")
    for result in results:
        print("="*50)
        print(result)
        print("="*50)