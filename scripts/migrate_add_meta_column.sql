-- Migration script to add meta JSONB column to search_items table
-- Run this if you have an existing database without the meta column

-- Add meta column if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'search_items' 
        AND column_name = 'meta'
    ) THEN
        ALTER TABLE search_items ADD COLUMN meta JSONB;
        RAISE NOTICE 'Added meta column to search_items table';
    ELSE
        RAISE NOTICE 'meta column already exists in search_items table';
    END IF;
END $$;

-- Update any existing rows to have empty meta object
UPDATE search_items SET meta = '{}' WHERE meta IS NULL;

-- Add comment for documentation
COMMENT ON COLUMN search_items.meta IS 'Entity metadata stored as JSONB for flexible search and filtering';
