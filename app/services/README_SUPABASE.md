# Supabase pgvector RAG Setup Guide with Voyage AI

This guide explains how to use Supabase with pgvector and Voyage AI for RAG (Retrieval-Augmented Generation) inference.

## Architecture

The RAG pipeline follows this flow:

```
User Question
     ↓
1. Voyage AI → Generate query embedding (1024-dim vector)
     ↓
2. Supabase pgvector → Vector similarity search
     ↓
3. Retrieve top-K most similar parts
     ↓
4. Format as context for Claude
     ↓
5. Claude API → Generate natural language response
     ↓
User Answer
```

## Prerequisites

1. **Supabase Account**: Sign up at https://supabase.com
2. **Supabase Project**: Create a new project or use an existing one
3. **Voyage AI Account**: Sign up at https://www.voyageai.com and get an API key
4. **Environment Variables**: Configure the following in your `.env` file:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-or-service-role-key

# Voyage AI Configuration
VOYAGE_API_KEY=your-voyage-ai-api-key
EMBEDDING_DIMENSION=1024  # Voyage AI default (voyage-2, voyage-lite-02, voyage-large-2)
EMBEDDING_MODEL=voyage-2  # Voyage AI model name

# Claude Configuration (already configured)
ANTHROPIC_API_KEY=your-anthropic-api-key
```

## Database Setup

1. Open your Supabase project dashboard
2. Navigate to **SQL Editor**
3. Run the SQL script from `supabase_setup.sql` in the project root
4. This will:
   - Enable the pgvector extension
   - Create the `embeddings` table with 1024-dimensional vectors
   - Create vector index for efficient similarity search
   - Create the `match_embeddings` RPC function

## Usage Examples

### 1. Generate Embeddings with Voyage AI

Voyage AI generates 1024-dimensional vectors optimized for semantic search:

```python
from app.services.voyage_client import voyage_client
from app.services.supabase_client import supabase_vector_client

# Generate embedding using Voyage AI (1024-dim)
text = "Your text content here"
embedding = voyage_client.generate_embedding(text)

# Store in Supabase
result = supabase_vector_client.store_embedding(
    content=text,
    embedding=embedding,  # 1024-dimensional vector
    metadata={
        "source": "police_report",
        "document_id": "report_123",
        "section": "accident_details"
    }
)
```

### 2. Perform Similarity Search (Steps 1-3 of Architecture)

```python
from app.services.voyage_client import voyage_client
from app.services.supabase_client import supabase_vector_client

# Step 1: Voyage AI → Generate query embedding (1024-dim vector)
query_text = "What are the liability rules for traffic accidents?"
query_embedding = voyage_client.generate_query_embedding(query_text)

# Step 2: Supabase pgvector → Vector similarity search
# Step 3: Retrieve top-K most similar parts
results = supabase_vector_client.similarity_search(
    query_embedding=query_embedding,
    limit=5,
    threshold=0.7,  # Minimum similarity score (0.0 to 1.0)
    metadata_filter={"source": "legal_docs"}  # Optional filter
)

for result in results:
    print(f"Content: {result['content']}")
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Metadata: {result['metadata']}")
    print("---")
```

### 3. Complete RAG Pipeline (All 5 Steps)

```python
from app.services.voyage_client import voyage_client
from app.services.supabase_client import supabase_vector_client
from app.services.claude_client import claude_client

def complete_rag_pipeline(user_query: str):
    """
    Complete RAG pipeline following the architecture:
    1. Voyage AI → Generate query embedding (1024-dim vector)
    2. Supabase pgvector → Vector similarity search
    3. Retrieve top-K most similar parts
    4. Format as context for Claude
    5. Claude API → Generate natural language response
    """
    # Step 1: Voyage AI → Generate query embedding (1024-dim vector)
    query_embedding = voyage_client.generate_query_embedding(user_query)
    
    # Steps 2-3: Supabase pgvector → Vector similarity search → Retrieve top-K
    context = supabase_vector_client.retrieve_context_for_rag(
        query_embedding=query_embedding,
        top_k=5,
        similarity_threshold=0.7,
        metadata_filter={"source": "legal_docs"}
    )
    
    if not context:
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

# Usage
result = complete_rag_pipeline("What are the liability rules for traffic accidents?")
print(result["response"])
```

### 4. Batch Operations

Store multiple documents efficiently:

```python
from app.services.voyage_client import voyage_client
from app.services.supabase_client import supabase_vector_client

def store_documents_batch(documents: list[dict]):
    """
    Store multiple documents with their embeddings.
    
    Args:
        documents: List of dicts with 'content', optional 'metadata'
    """
    # Generate embeddings for all documents in batch
    texts = [doc['content'] for doc in documents]
    embeddings = voyage_client.generate_embeddings_batch(texts, input_type="document")
    
    # Store each document with its embedding
    for doc, embedding in zip(documents, embeddings):
        supabase_vector_client.store_embedding(
            content=doc['content'],
            embedding=embedding,
            metadata=doc.get('metadata', {})
        )
```

## Voyage AI Models

Voyage AI offers several models, all generating 1024-dimensional vectors:

- **voyage-2**: General-purpose model (default)
- **voyage-lite-02**: Lighter, faster model
- **voyage-large-2**: Higher quality, slower model

Update `EMBEDDING_MODEL` in your `.env` file to use a different model.

## Best Practices

1. **Embedding Dimension**: Voyage AI always generates 1024-dimensional vectors. The SQL setup is configured for this.

2. **Query vs Document Embeddings**: 
   - Use `generate_query_embedding()` for search queries (step 1 of architecture)
   - Use `generate_embedding()` or `generate_embeddings_batch()` with `input_type="document"` for storing documents

3. **Batch Processing**: Use `generate_embeddings_batch()` when storing multiple documents for better performance.

4. **Similarity Threshold**: Start with 0.7 for semantic similarity. Adjust based on your use case:
   - Higher (0.8-0.9): More strict, fewer but more relevant results
   - Lower (0.5-0.6): More lenient, more results but potentially less relevant

5. **Chunking**: For long documents, split into smaller chunks (200-500 tokens) before generating embeddings for better retrieval.

6. **Metadata Filtering**: Use metadata to filter by document type, source, date, etc., for more relevant results:
   ```python
   metadata_filter={"source": "vietnam_traffic_law", "article_number": "10"}
   ```

## Integration with Existing Code

You can integrate RAG into your existing Claude-based endpoints. For example, in `report_summarizer_api.py`:

```python
from app.services.voyage_client import voyage_client
from app.services.supabase_client import supabase_vector_client

# Before calling Claude, enhance with RAG context
user_query = "Analyze this police report"

# Get relevant legal context
query_embedding = voyage_client.generate_query_embedding(user_query)
legal_context = supabase_vector_client.retrieve_context_for_rag(
    query_embedding=query_embedding,
    top_k=5,
    similarity_threshold=0.7,
    metadata_filter={"source": "vietnam_traffic_law"}
)

# Include in Claude prompt
enhanced_prompt = f"""
Legal Context:
{legal_context}

[Original prompt continues...]
"""
```

## Troubleshooting

### "Voyage AI client not initialized"
- Check that `VOYAGE_API_KEY` is set in your `.env` file
- Verify your Voyage AI API key is valid

### "match_embeddings function does not exist"
- Make sure you ran the SQL setup script in Supabase SQL Editor
- Check that the function name matches exactly

### "Embedding dimension mismatch"
- Voyage AI always generates 1024-dimensional vectors
- Ensure your SQL setup uses `vector(1024)` (which it does by default)

### Slow queries
- Ensure the IVFFlat index is created
- Increase the `lists` parameter if you have many embeddings (>100k)
- Consider using HNSW index for larger datasets (requires pgvector 0.5+)

### No results returned
- Lower the `similarity_threshold`
- Check that embeddings are stored correctly
- Verify metadata filters aren't too restrictive

### Voyage AI API errors
- Check your API key and account status
- Verify you have sufficient API credits
- Check Voyage AI status page for service issues

## Example: Storing Legal Documents

```python
# Store Vietnam Road Traffic Law sections
legal_sections = [
    {
        "content": "Article 10: All road users must comply with traffic regulations...",
        "metadata": {
            "source": "vietnam_traffic_law",
            "article_number": "10",
            "document_id": "traffic_law_36_2024"
        }
    },
    # ... more sections
]

# Generate embeddings in batch
texts = [section['content'] for section in legal_sections]
embeddings = voyage_client.generate_embeddings_batch(texts)

# Store in Supabase
for section, embedding in zip(legal_sections, embeddings):
    supabase_vector_client.store_embedding(
        content=section['content'],
        embedding=embedding,
        metadata=section['metadata']
    )
```

## See Also

- `example_rag_usage.py` - Complete working examples
- `app/services/voyage_client.py` - Voyage AI client implementation
- `app/services/supabase_client.py` - Supabase pgvector client implementation
- `supabase_setup.sql` - Database setup SQL script