"""
Voyage AI client for generating embeddings.
Voyage AI generates 1024-dimensional vectors optimized for semantic search.
"""
from typing import List, Union
import voyageai
from app.config import settings
from app.utils.logger import logger


class VoyageClient:
    """
    Client for interacting with Voyage AI embedding API.
    Voyage AI provides high-quality embeddings optimized for retrieval tasks.
    """
    
    def __init__(self):
        """Initialize Voyage AI client with API key from environment."""
        try:
            voyage_api_key = getattr(settings, 'voyage_api_key', None)
            
            if not voyage_api_key:
                raise ValueError(
                    "Voyage API key must be set in environment variables. "
                    "Set VOYAGE_API_KEY in your .env file."
                )
            
            self.client = voyageai.Client(api_key=voyage_api_key)
            self.model = getattr(settings, 'embedding_model', 'voyage-2')
            self.dimension = getattr(settings, 'embedding_dimension', 1024)
            logger.info(f"Voyage AI client initialized successfully (model={self.model}, dimension={self.dimension})")
        except Exception as e:
            logger.error(f"Failed to initialize Voyage AI client: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a single embedding for the given text.
        
        Args:
            text: The text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector (1024 dimensions)
        """
        try:
            result = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="document"  # or "query" for query embeddings
            )
            
            # Voyage AI returns result with 'embeddings' attribute
            if hasattr(result, 'embeddings') and result.embeddings and len(result.embeddings) > 0:
                embedding = result.embeddings[0]
                logger.debug(f"Generated embedding of dimension {len(embedding)}")
                return embedding
            elif isinstance(result, list) and len(result) > 0:
                # Handle case where result is directly a list
                embedding = result[0]
                logger.debug(f"Generated embedding of dimension {len(embedding)}")
                return embedding
            else:
                raise ValueError("No embedding returned from Voyage AI")
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_query_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding optimized for query/search tasks.
        
        Voyage AI supports query embeddings that are optimized for retrieval.
        Use input_type="query" for better search performance.
        
        Args:
            text: The query text to generate embedding for
            
        Returns:
            List of floats representing the query embedding vector (1024 dimensions)
        """
        try:
            # Voyage models support query-specific embeddings with input_type="query"
            result = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="query"  # Optimized for search queries
            )
            
            # Voyage AI returns result with 'embeddings' attribute
            if hasattr(result, 'embeddings') and result.embeddings and len(result.embeddings) > 0:
                embedding = result.embeddings[0]
                logger.debug(f"Generated query embedding of dimension {len(embedding)}")
                return embedding
            elif isinstance(result, list) and len(result) > 0:
                # Handle case where result is directly a list
                embedding = result[0]
                logger.debug(f"Generated query embedding of dimension {len(embedding)}")
                return embedding
            else:
                raise ValueError("No query embedding returned from Voyage AI")
                
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            # Fallback to regular embedding if query type not supported
            logger.warning(f"Falling back to regular embedding generation: {e}")
            return self.generate_embedding(text)
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        input_type: str = "document"
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a single API call.
        More efficient than calling generate_embedding multiple times.
        
        Args:
            texts: List of texts to generate embeddings for
            input_type: Type of input - "document" or "query"
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        try:
            if not texts:
                return []
            
            result = self.client.embed(
                texts=texts,
                model=self.model,
                input_type=input_type
            )
            
            if result.embeddings:
                logger.info(f"Generated {len(result.embeddings)} embeddings in batch")
                return result.embeddings
            else:
                raise ValueError("No embeddings returned from Voyage AI")
                
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def get_client_info(self) -> dict:
        """Get information about the Voyage AI client configuration."""
        return {
            "client_type": "voyage_ai",
            "model": self.model,
            "dimension": self.dimension,
            "api_key_configured": bool(getattr(settings, 'voyage_api_key', None))
        }


# Create a global instance
try:
    voyage_client = VoyageClient()
except Exception as e:
    logger.warning(f"Voyage AI client not initialized: {e}. Please configure VOYAGE_API_KEY.")
    voyage_client = None