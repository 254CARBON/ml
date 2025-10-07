-- Initialize database for ML platform
-- This script sets up the required extensions and initial tables

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create embeddings table for vector storage
-- Note: Vector dimension should match ML_VECTOR_DIMENSION environment variable
-- Default is 384 for sentence-transformers/all-MiniLM-L6-v2
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    vector vector(384), -- Default dimension, update via migration if needed
    meta JSONB,
    model_version VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(100) DEFAULT 'default',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_type, entity_id, model_version, tenant_id)
);

-- Create search_items table for metadata
CREATE TABLE IF NOT EXISTS search_items (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    text TEXT NOT NULL,
    meta JSONB,
    tags JSONB,
    tenant_id VARCHAR(100) DEFAULT 'default',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_type, entity_id, tenant_id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_tenant ON embeddings(tenant_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_version);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING ivfflat (vector vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_search_items_entity ON search_items(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_search_items_tenant ON search_items(tenant_id);
CREATE INDEX IF NOT EXISTS idx_search_items_text ON search_items USING gin(to_tsvector('english', text));

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_embeddings_updated_at BEFORE UPDATE ON embeddings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_search_items_updated_at BEFORE UPDATE ON search_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlflow;
