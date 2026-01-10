"""
Example usage of Supabase pgvector client for RAG inference with Voyage AI.
Architecture: User Question → Voyage AI (1024-dim) → Supabase pgvector → Top-K → Claude API → User Answer

This is a reference example - integrate these patterns into your existing code.
"""

from app.config import settings
from app.services.voyage_client import voyage_client
from app.services.supabase_client import supabase_vector_client
from app.utils.logger import logger

# Example 1: Store embeddings for knowledge base
def store_knowledge_base_documents(documents: list):
    """
    Store documents in Supabase vector database using Voyage AI embeddings.
    
    Args:
        documents: List of dicts with 'content', optional 'metadata'
    """
    if supabase_vector_client is None:
        logger.error("Supabase client not initialized. Please configure SUPABASE_URL and SUPABASE_KEY.")
        return
    
    if voyage_client is None:
        logger.error("Voyage AI client not initialized. Please configure VOYAGE_API_KEY.")
        return
    
    for doc in documents:
        try:
            # Generate embedding using Voyage AI (1024-dim)
            embedding = voyage_client.generate_embedding(doc['content'])
            
            # Store in Supabase
            result = supabase_vector_client.store_embedding(
                content=doc['content'],
                embedding=embedding,
                metadata=doc.get('metadata', {
                    'source': 'knowledge_base',
                    'document_type': 'general'
                })
            )
            
            if result:
                logger.info(f"Stored document with id: {result.get('id')}")
            else:
                logger.warning(f"Failed to store document: {doc.get('content', '')[:50]}...")
                
        except Exception as e:
            logger.error(f"Error storing document: {e}")


# Example 2: Query with RAG
def query_with_rag(user_query: str, top_k: int = 5) -> str:
    """
    Query the knowledge base using RAG and return relevant context.
    
    Architecture:
    1. Voyage AI → Generate query embedding (1024-dim vector)
    2. Supabase pgvector → Vector similarity search
    3. Retrieve top-K most similar parts
    
    Args:
        user_query: User's question/query
        top_k: Number of top similar documents to retrieve
        
    Returns:
        Combined context string from similar documents
    """
    if supabase_vector_client is None:
        logger.error("Supabase client not initialized")
        return ""
    
    if voyage_client is None:
        logger.error("Voyage AI client not initialized")
        return ""
    
    try:
        # Step 1: Generate query embedding using Voyage AI (1024-dim)
        query_embedding = voyage_client.generate_query_embedding(user_query)
        
        # Step 2: Supabase pgvector similarity search
        # Step 3: Retrieve top-K most similar parts
        context = supabase_vector_client.retrieve_context_for_rag(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=0.7
        )
        
        return context
        
    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        return ""


# Example 3: Store police report sections for legal reference
def store_legal_references(legal_texts: list):
    """
    Store legal texts (e.g., Vietnam Road Traffic Law sections) for RAG.
    
    Args:
        legal_texts: List of dicts with 'content', 'article_number', 'section', etc.
    """
    if supabase_vector_client is None:
        logger.error("Supabase client not initialized")
        return
    
    if voyage_client is None:
        logger.error("Voyage AI client not initialized")
        return
    
    for text in legal_texts:
        try:
            # Generate embedding using Voyage AI (1024-dim)
            embedding = voyage_client.generate_embedding(text['content'])
            
            # Store with rich metadata
            result = supabase_vector_client.store_embedding(
                content=text['content'],
                embedding=embedding,
                metadata={
                    'source': 'vietnam_traffic_law',
                    'article_number': text.get('article_number'),
                    'section': text.get('section'),
                    'category': text.get('category', 'general'),
                    'document_id': text.get('document_id', 'traffic_law_36_2024')
                }
            )
            
            if result:
                logger.info(f"Stored legal reference: Article {text.get('article_number')}")
                
        except Exception as e:
            logger.error(f"Error storing legal reference: {e}")


# Example 4: Complete RAG pipeline - User Question to User Answer
def enhanced_claude_query(user_query: str):
    """
    Complete RAG pipeline following the architecture:
    1. Voyage AI → Generate query embedding (1024-dim vector)
    2. Supabase pgvector → Vector similarity search
    3. Retrieve top-K most similar parts
    4. Format as context for Claude
    5. Claude API → Generate natural language response
    
    Args:
        user_query: User's question/query
        
    Returns:
        Dictionary with Claude's response and context used
    """
    if supabase_vector_client is None:
        logger.error("Supabase client not initialized")
        return None
    
    if voyage_client is None:
        logger.error("Voyage AI client not initialized")
        return None
    
    try:
        # Steps 1-3: Get relevant context using RAG pipeline
        context = query_with_rag(user_query, top_k=5)
        
        if not context:
            logger.warning("No relevant context found for RAG")
            context = "No relevant context available in knowledge base."
        
        # Step 4: Format as context for Claude
        enhanced_prompt = f"""
Context from knowledge base:
{context}

---

User Question: {user_query}

Please answer the question using the provided context from the knowledge base. 
If the context doesn't contain relevant information to fully answer the question, 
please state that and provide the best answer you can based on general knowledge.
"""
        
        # Step 5: Claude API → Generate natural language response
        from app.services.claude_client import claude_client
        
        message = claude_client.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{
                "role": "user",
                "content": enhanced_prompt
            }]
        )
        
        return {
            "response": message.content[0].text,
            "context_used": context,
            "usage": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens
            }
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced Claude query: {e}")
        return None


# Example usage (commented out - uncomment to test)
if __name__ == "__main__":
    # Check if clients are configured
    if supabase_vector_client:
        logger.info("Supabase client is ready")
        info = supabase_vector_client.get_client_info()
        logger.info(f"Supabase client info: {info}")
    else:
        logger.warning("Supabase client not initialized. Configure SUPABASE_URL and SUPABASE_KEY.")
    
    if voyage_client:
        logger.info("Voyage AI client is ready")
        voyage_info = voyage_client.get_client_info()
        logger.info(f"Voyage AI client info: {voyage_info}")
    else:
        logger.warning("Voyage AI client not initialized. Configure VOYAGE_API_KEY.")
    
    # Example: Store documents
    # sample_docs = [
    #     {
    #         "content": "Article 10: All road users must comply with traffic regulations...",
    #         "metadata": {"article": 10, "source": "traffic_law"}
    #     }
    # ]
    # store_knowledge_base_documents(sample_docs)
    
    # Example: Query with RAG
    # result = query_with_rag("What are the liability rules for traffic accidents?")
    # print(result)