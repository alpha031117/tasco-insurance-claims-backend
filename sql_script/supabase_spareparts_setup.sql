-- Generated SQL for Supabase
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table
CREATE TABLE IF NOT EXISTS motorcycle_spareparts (
    id BIGSERIAL PRIMARY KEY,
    partnumber VARCHAR(500) NOT NULL,
    oemnumber VARCHAR(500) NOT NULL,
    partname VARCHAR(500) NOT NULL,
    brand VARCHAR(500) NOT NULL,
    category VARCHAR(500) NOT NULL,
    make VARCHAR(500) NOT NULL,
    model VARCHAR(500) NOT NULL,
    enginecapacitycc INTEGER NOT NULL,
    region VARCHAR(500) NOT NULL,
    retailpricevnd INTEGER NOT NULL,
    costpricevnd INTEGER NOT NULL,
    repaircode VARCHAR(500) NOT NULL,
    repairname VARCHAR(500) NOT NULL,
    totalrepairpricevnd INTEGER NOT NULL,
    
    -- Vector embedding for semantic search (1024 dimensions for Voyage AI)
    embedding vector(1024),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS motorcycle_spareparts_embedding_idx 
ON motorcycle_spareparts 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS motorcycle_spareparts_brand_idx ON motorcycle_spareparts(brand);
CREATE INDEX IF NOT EXISTS motorcycle_spareparts_make_model_idx ON motorcycle_spareparts(make, model);
CREATE INDEX IF NOT EXISTS motorcycle_spareparts_category_idx ON motorcycle_spareparts(category);
CREATE INDEX IF NOT EXISTS motorcycle_spareparts_region_idx ON motorcycle_spareparts(region);

-- Auto-update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_motorcycle_spareparts_updated_at 
    BEFORE UPDATE ON motorcycle_spareparts 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Search function for vector similarity with filters
-- Note: All VARCHAR columns are cast to TEXT to match RETURNS TABLE declaration
CREATE OR REPLACE FUNCTION search_motorcycle_parts(
    query_embedding vector(1024),
    match_limit int DEFAULT 5,
    brand_filter text DEFAULT NULL,
    make_filter text DEFAULT NULL,
    region_filter text DEFAULT NULL,
    max_price integer DEFAULT NULL
)
RETURNS TABLE (
    id bigint,
    partnumber text,
    oemnumber text,
    partname text,
    brand text,
    category text,
    make text,
    model text,
    enginecapacitycc integer,
    region text,
    retailpricevnd integer,
    costpricevnd integer,
    repaircode text,
    repairname text,
    totalrepairpricevnd integer,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.partnumber::text,
        m.oemnumber::text,
        m.partname::text,
        m.brand::text,
        m.category::text,
        m.make::text,
        m.model::text,
        m.enginecapacitycc,
        m.region::text,
        m.retailpricevnd,
        m.costpricevnd,
        m.repaircode::text,
        m.repairname::text,
        m.totalrepairpricevnd,
        1 - (m.embedding <=> query_embedding) as similarity
    FROM motorcycle_spareparts m
    WHERE 
        m.embedding IS NOT NULL  -- Only search rows with embeddings (required for vector search)
        AND (brand_filter IS NULL OR m.brand = brand_filter)
        AND (make_filter IS NULL OR m.make = make_filter)
        AND (region_filter IS NULL OR m.region = region_filter)
        AND (max_price IS NULL OR m.retailpricevnd <= max_price)
    ORDER BY m.embedding <=> query_embedding
    LIMIT match_limit;
END;
$$;
