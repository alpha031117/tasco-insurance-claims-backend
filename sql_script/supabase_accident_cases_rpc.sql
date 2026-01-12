-- SQL Setup for accident_cases table with pgvector similarity search
-- Run this in Supabase SQL Editor

-- 1. Enable vector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create RPC function for embedding similarity search
CREATE OR REPLACE FUNCTION search_similar_accident_cases(
    query_embedding vector(1024),
    match_threshold float DEFAULT 0.85,
    match_count int DEFAULT 5
)
RETURNS TABLE (
    id uuid,
    case_id text,
    summary_text text,
    metadata jsonb,
    similarity float,
    accident_date date,
    accident_city text,
    vehicle_plate text,
    is_fraud boolean,
    fraud_score float8,
    created_at timestamptz
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        accident_cases.id,
        accident_cases.case_id,
        accident_cases.summary_text,
        accident_cases.metadata,
        1 - (accident_cases.embedding <=> query_embedding) AS similarity,
        accident_cases.accident_date,
        accident_cases.accident_city,
        accident_cases.vehicle_plate,
        accident_cases.is_fraud,
        accident_cases.fraud_score,
        accident_cases.created_at
    FROM accident_cases
    WHERE 
        accident_cases.embedding IS NOT NULL
        AND (1 - (accident_cases.embedding <=> query_embedding)) > match_threshold
    ORDER BY accident_cases.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 3. Create index for faster vector similarity search (if not exists)
CREATE INDEX IF NOT EXISTS accident_cases_embedding_idx 
ON accident_cases 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 4. Test the function (optional)
-- SELECT * FROM search_similar_accident_cases(
--     query_embedding := '[0.1, 0.2, ...]'::vector(1024),
--     match_threshold := 0.85,
--     match_count := 5
-- );

-- Notes:
-- - The <=> operator computes cosine distance (lower = more similar)
-- - Cosine similarity = 1 - cosine distance
-- - Similarity threshold of 0.85 means 85% similar or higher
-- - Adjust match_threshold and match_count as needed for your use case
