"""
Supabase client with pgvector support for RAG inference.
"""
from typing import List, Dict, Optional, Any
from supabase import create_client, Client
from app.config import settings
from app.utils.logger import logger
import numpy as np


class SupabaseVectorClient:
    """
    Client for interacting with Supabase PostgreSQL with pgvector extension.
    Supports vector storage and similarity search for RAG inference.
    """
    
    def __init__(self):
        """Initialize Supabase client with connection details from environment."""
        try:
            supabase_url = getattr(settings, 'supabase_url', None)
            supabase_key = getattr(settings, 'supabase_key', None)
            
            if not supabase_url or not supabase_key:
                raise ValueError(
                    "Supabase URL and key must be set in environment variables. "
                    "Set SUPABASE_URL and SUPABASE_KEY in your .env file."
                )
            
            self.client: Client = create_client(
                supabase_url,
                supabase_key
            )
            self.embedding_dimension = getattr(settings, 'embedding_dimension', 1024)
            logger.info(f"Supabase client initialized successfully (embedding_dimension={self.embedding_dimension})")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    def create_vector_table(self, table_name: str = "embeddings") -> bool:
        """
        Create a vector table with pgvector support.
        
        Note: This requires running SQL directly. In production, use Supabase migrations.
        For now, this method documents the required table structure.
        
        Expected SQL (run in Supabase SQL editor):
        ```sql
        CREATE EXTENSION IF NOT EXISTS vector;
        
        CREATE TABLE IF NOT EXISTS embeddings (
            id BIGSERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector(1024),
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
        ON embeddings 
        USING ivfflat (embedding vector_cosine_ops);
        ```
        
        Args:
            table_name: Name of the table to create
            
        Returns:
            True if successful (in actual implementation, this would execute SQL)
        """
        logger.info(f"Table structure documentation for {table_name} provided")
        logger.warning("Please create the table manually in Supabase SQL editor using the SQL provided in docstring")
        return True
    
    def store_embedding(
        self,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        table_name: str = "embeddings"
    ) -> Optional[Dict[str, Any]]:
        """
        Store an embedding with associated content and metadata.
        
        Args:
            content: The text content associated with the embedding
            embedding: The vector embedding (list of floats)
            metadata: Optional metadata dictionary (e.g., source, document_id, etc.)
            table_name: Name of the table to store in
            
        Returns:
            Dictionary with inserted record data, or None if failed
        """
        try:
            # Ensure embedding matches expected dimension
            if len(embedding) != self.embedding_dimension:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                    f"got {len(embedding)}. Truncating or padding."
                )
                if len(embedding) > self.embedding_dimension:
                    embedding = embedding[:self.embedding_dimension]
                else:
                    embedding = embedding + [0.0] * (self.embedding_dimension - len(embedding))
            
            # Prepare data for insertion
            data = {
                "content": content,
                "embedding": embedding,
                "metadata": metadata or {}
            }
            
            # Insert into Supabase
            result = self.client.table(table_name).insert(data).execute()
            
            if result.data:
                logger.info(f"Successfully stored embedding with id: {result.data[0].get('id')}")
                return result.data[0]
            else:
                logger.error("Failed to store embedding: No data returned")
                return None
                
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return None
    
    def similarity_search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        threshold: float = 0.0,
        table_name: str = "embeddings",
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using cosine distance.
        
        Note: This uses Supabase RPC (Remote Procedure Call) function.
        You need to create this function in Supabase SQL editor:
        
        ```sql
        CREATE OR REPLACE FUNCTION match_embeddings(
            query_embedding vector(1024),
            match_threshold float,
            match_count int,
            filter_metadata jsonb DEFAULT '{}'::jsonb
        )
        RETURNS TABLE (
            id bigint,
            content text,
            metadata jsonb,
            similarity float
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                embeddings.id,
                embeddings.content,
                embeddings.metadata,
                1 - (embeddings.embedding <=> query_embedding) AS similarity
            FROM embeddings
            WHERE 
                (filter_metadata = '{}'::jsonb OR embeddings.metadata @> filter_metadata)
                AND (1 - (embeddings.embedding <=> query_embedding)) > match_threshold
            ORDER BY embeddings.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
        ```
        
        Args:
            query_embedding: The query vector to search for
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)
            table_name: Name of the table to search in
            metadata_filter: Optional metadata filter (e.g., {"source": "legal_docs"})
            
        Returns:
            List of dictionaries containing matched content, metadata, and similarity scores
        """
        try:
            # Ensure embedding matches expected dimension
            if len(query_embedding) != self.embedding_dimension:
                logger.warning(
                    f"Query embedding dimension mismatch: expected {self.embedding_dimension}, "
                    f"got {len(query_embedding)}"
                )
                if len(query_embedding) > self.embedding_dimension:
                    query_embedding = query_embedding[:self.embedding_dimension]
                else:
                    query_embedding = query_embedding + [0.0] * (self.embedding_dimension - len(query_embedding))
            
            # Call RPC function for similarity search
            result = self.client.rpc(
                'match_embeddings',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': threshold,
                    'match_count': limit,
                    'filter_metadata': metadata_filter or {}
                }
            ).execute()
            
            if result.data:
                logger.info(f"Found {len(result.data)} similar embeddings")
                return result.data
            else:
                logger.info("No similar embeddings found")
                return []
                
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            logger.warning("Make sure the match_embeddings RPC function is created in Supabase")
            # Fallback: try direct query (less efficient, but works if RPC not available)
            return self._fallback_similarity_search(query_embedding, limit, table_name)
    
    def _fallback_similarity_search(
        self,
        query_embedding: List[float],
        limit: int,
        table_name: str
    ) -> List[Dict[str, Any]]:
        """
        Fallback similarity search method (less efficient).
        Fetches all embeddings and computes similarity in Python.
        Only use for small datasets.
        """
        try:
            logger.warning("Using fallback similarity search (fetching all embeddings)")
            result = self.client.table(table_name).select("*").execute()
            
            if not result.data:
                return []
            
            # Compute cosine similarity for each embedding
            query_vec = np.array(query_embedding)
            similarities = []
            
            for item in result.data:
                if item.get('embedding'):
                    embedding_vec = np.array(item['embedding'])
                    # Cosine similarity
                    similarity = np.dot(query_vec, embedding_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(embedding_vec)
                    )
                    similarities.append({
                        **item,
                        'similarity': float(similarity)
                    })
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Error in fallback similarity search: {e}")
            return []
    
    def retrieve_context_for_rag(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Retrieve relevant context for RAG inference.
        
        Args:
            query_embedding: The query vector
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            metadata_filter: Optional metadata filter
            
        Returns:
            Combined context string from top similar embeddings
        """
        results = self.similarity_search(
            query_embedding=query_embedding,
            limit=top_k,
            threshold=similarity_threshold,
            metadata_filter=metadata_filter
        )
        
        if not results:
            logger.warning("No relevant context found for RAG")
            return ""
        
        # Combine top results into context
        context_parts = []
        for i, result in enumerate(results, 1):
            content = result.get('content', '')
            similarity = result.get('similarity', 0.0)
            metadata = result.get('metadata', {})
            
            context_parts.append(
                f"[Context {i} - Similarity: {similarity:.3f}]\n{content}\n"
            )
            
            # Add metadata info if available
            if metadata:
                context_parts.append(f"Metadata: {metadata}\n")
            
            context_parts.append("\n---\n\n")
        
        context = "".join(context_parts)
        logger.info(f"Retrieved {len(results)} contexts for RAG (total length: {len(context)} chars)")
        return context
    
    def delete_embedding(self, embedding_id: int, table_name: str = "embeddings") -> bool:
        """
        Delete an embedding by ID.
        
        Args:
            embedding_id: ID of the embedding to delete
            table_name: Name of the table
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.client.table(table_name).delete().eq("id", embedding_id).execute()
            logger.info(f"Deleted embedding with id: {embedding_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting embedding: {e}")
            return False
    
    def update_embedding(
        self,
        embedding_id: int,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        table_name: str = "embeddings"
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing embedding.
        
        Args:
            embedding_id: ID of the embedding to update
            content: Optional new content
            embedding: Optional new embedding vector
            metadata: Optional new metadata (will merge with existing)
            table_name: Name of the table
            
        Returns:
            Updated record data, or None if failed
        """
        try:
            update_data = {}
            
            if content is not None:
                update_data["content"] = content
            
            if embedding is not None:
                if len(embedding) != self.embedding_dimension:
                    logger.warning(f"Embedding dimension mismatch, adjusting...")
                    if len(embedding) > self.embedding_dimension:
                        embedding = embedding[:self.embedding_dimension]
                    else:
                        embedding = embedding + [0.0] * (self.embedding_dimension - len(embedding))
                update_data["embedding"] = embedding
            
            if metadata is not None:
                update_data["metadata"] = metadata
            
            update_data["updated_at"] = "now()"
            
            result = self.client.table(table_name).update(update_data).eq("id", embedding_id).execute()
            
            if result.data:
                logger.info(f"Updated embedding with id: {embedding_id}")
                return result.data[0]
            else:
                logger.error(f"Failed to update embedding with id: {embedding_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error updating embedding: {e}")
            return None
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get information about the Supabase client configuration."""
        return {
            "client_type": "supabase_pgvector",
            "embedding_dimension": self.embedding_dimension,
            "url_configured": hasattr(settings, 'supabase_url') and bool(getattr(settings, 'supabase_url', None)),
            "key_configured": hasattr(settings, 'supabase_key') and bool(getattr(settings, 'supabase_key', None))
        }


# Create a global instance
try:
    supabase_vector_client = SupabaseVectorClient()
except Exception as e:
    logger.warning(f"Supabase client not initialized: {e}. Please configure SUPABASE_URL and SUPABASE_KEY.")
    supabase_vector_client = None