-- Migration script to update vector dimension
-- This script updates the vector column dimension based on ML_VECTOR_DIMENSION

-- Drop existing vector index
DROP INDEX IF EXISTS idx_embeddings_vector;

-- Alter the vector column dimension
-- Note: This requires recreating the table in PostgreSQL with pgvector
-- For production, use a proper migration tool

-- Create new table with configurable dimension
CREATE TABLE IF NOT EXISTS embeddings_new (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    vector vector(${ML_VECTOR_DIMENSION}), -- Configurable dimension
    meta JSONB,
    model_version VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(100) DEFAULT 'default',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_type, entity_id, model_version, tenant_id)
);

-- Copy data from old table (if exists)
INSERT INTO embeddings_new (entity_type, entity_id, vector, meta, model_version, tenant_id, created_at, updated_at)
SELECT entity_type, entity_id, vector, meta, model_version, tenant_id, created_at, updated_at
FROM embeddings
ON CONFLICT (entity_type, entity_id, model_version, tenant_id) DO NOTHING;

-- Drop old table and rename new one
DROP TABLE IF EXISTS embeddings;
ALTER TABLE embeddings_new RENAME TO embeddings;

-- Recreate indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_tenant ON embeddings(tenant_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_version);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING ivfflat (vector vector_cosine_ops);

-- Create function to update updated_at timestamp (if not exists)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for updated_at
DROP TRIGGER IF EXISTS update_embeddings_updated_at ON embeddings;
CREATE TRIGGER update_embeddings_updated_at BEFORE UPDATE ON embeddings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
