#!/usr/bin/env python3
"""Initialize database with configurable vector dimension."""

import os
import asyncio
import asyncpg
from libs.common.config import BaseConfig


async def init_database():
    """Initialize database with configurable vector dimension."""
    config = BaseConfig()
    vector_dimension = config.ml_vector_dimension
    
    print(f"Initializing database with vector dimension: {vector_dimension}")
    
    # Connect to database
    conn = await asyncpg.connect(config.ml_vector_db_dsn)
    
    try:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("✓ pgvector extension enabled")
        
        # Create embeddings table with configurable dimension
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                entity_type VARCHAR(50) NOT NULL,
                entity_id VARCHAR(255) NOT NULL,
                vector vector({vector_dimension}),
                meta JSONB,
                model_version VARCHAR(100) NOT NULL,
                tenant_id VARCHAR(100) DEFAULT 'default',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(entity_type, entity_id, model_version, tenant_id)
            );
        """)
        print(f"✓ embeddings table created with vector dimension {vector_dimension}")
        
        # Create search_items table
        await conn.execute("""
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
        """)
        print("✓ search_items table created")
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_type, entity_id);",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_tenant ON embeddings(tenant_id);",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_version);",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING ivfflat (vector vector_cosine_ops);",
            "CREATE INDEX IF NOT EXISTS idx_search_items_entity ON search_items(entity_type, entity_id);",
            "CREATE INDEX IF NOT EXISTS idx_search_items_tenant ON search_items(tenant_id);",
            "CREATE INDEX IF NOT EXISTS idx_search_items_text ON search_items USING gin(to_tsvector('english', text));"
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
        print("✓ indexes created")
        
        # Create update trigger function
        await conn.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)
        
        # Create triggers
        await conn.execute("""
            DROP TRIGGER IF EXISTS update_embeddings_updated_at ON embeddings;
            CREATE TRIGGER update_embeddings_updated_at BEFORE UPDATE ON embeddings
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """)
        
        await conn.execute("""
            DROP TRIGGER IF EXISTS update_search_items_updated_at ON search_items;
            CREATE TRIGGER update_search_items_updated_at BEFORE UPDATE ON search_items
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """)
        print("✓ triggers created")
        
        print("Database initialization completed successfully!")
        
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(init_database())
