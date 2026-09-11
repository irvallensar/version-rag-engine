import os

import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()


class DBConnection:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            user=os.getenv("DB_USER", "rag_user"),
            password=os.getenv("DB_PASSWORD", "rag_password"),
            dbname=os.getenv("DB_NAME", "version_rag"),
        )

        self.conn.autocommit = True
        self.init_db()

    def init_db(self):
        with self.conn.cursor() as cur:

            # Required extensions
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

            # ---------------------------------------------------------
            # Create the table if it does not exist.
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    manufacturer VARCHAR(100) NOT NULL,
                    model VARCHAR(100) NOT NULL,
                    model_year VARCHAR(50) NOT NULL,
                    chunk_type VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding VECTOR(768),
                    fts_vector TSVECTOR GENERATED ALWAYS AS (
                        to_tsvector('english', content)
                    ) STORED
                );
                """
            )

            # ---------------------------------------------------------
            # If an OLD version of the table existed, migrate it.
            # ---------------------------------------------------------

            # Old: library -> manufacturer
            cur.execute(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = 'document_chunks'
                        AND column_name = 'library'
                    )
                    AND NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = 'document_chunks'
                        AND column_name = 'manufacturer'
                    )
                    THEN
                        ALTER TABLE document_chunks
                        RENAME COLUMN library TO manufacturer;
                    END IF;
                END
                $$;
                """
            )

            # Old: version -> model_year
            cur.execute(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = 'document_chunks'
                        AND column_name = 'version'
                    )
                    AND NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = 'document_chunks'
                        AND column_name = 'model_year'
                    )
                    THEN
                        ALTER TABLE document_chunks
                        RENAME COLUMN version TO model_year;
                    END IF;
                END
                $$;
                """
            )

            # Add missing BMW columns if necessary.
            cur.execute(
                """
                ALTER TABLE document_chunks
                ADD COLUMN IF NOT EXISTS manufacturer VARCHAR(100);
                """
            )

            cur.execute(
                """
                ALTER TABLE document_chunks
                ADD COLUMN IF NOT EXISTS model VARCHAR(100);
                """
            )

            cur.execute(
                """
                ALTER TABLE document_chunks
                ADD COLUMN IF NOT EXISTS model_year VARCHAR(50);
                """
            )

            # ---------------------------------------------------------
            # Existing rows from an old schema may contain NULLs.
            # These values are only relevant if migrating old data.
            # ---------------------------------------------------------

            cur.execute(
                """
                UPDATE document_chunks
                SET manufacturer = 'unknown'
                WHERE manufacturer IS NULL;
                """
            )

            cur.execute(
                """
                UPDATE document_chunks
                SET model = 'unknown'
                WHERE model IS NULL;
                """
            )

            cur.execute(
                """
                UPDATE document_chunks
                SET model_year = 'unknown'
                WHERE model_year IS NULL;
                """
            )

            # Make the BMW fields mandatory.
            cur.execute(
                """
                ALTER TABLE document_chunks
                ALTER COLUMN manufacturer SET NOT NULL,
                ALTER COLUMN model SET NOT NULL,
                ALTER COLUMN model_year SET NOT NULL;
                """
            )

            # ---------------------------------------------------------
            # Indexes
            # ---------------------------------------------------------

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_manufacturer_model_year
                ON document_chunks (manufacturer, model, model_year);
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_metadata
                ON document_chunks USING GIN (metadata);
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fts
                ON document_chunks USING GIN (fts_vector);
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embedding
                ON document_chunks
                USING hnsw (embedding vector_cosine_ops);
                """
            )

    def insert_chunk(
        self,
        manufacturer: str,
        model: str,
        model_year: str,
        chunk_type: str,
        content: str,
        metadata: dict,
        embedding: list[float],
    ):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_chunks (
                    manufacturer,
                    model,
                    model_year,
                    chunk_type,
                    content,
                    metadata,
                    embedding
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s::vector
                );
                """,
                (
                    manufacturer,
                    model,
                    model_year,
                    chunk_type,
                    content,
                    Json(metadata),
                    embedding,
                ),
            )

    def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        target_manufacturer: str,
        target_model: str,
        target_model_year: str | None = None,
        top_k: int = 5,
    ):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                WITH semantic_search AS (
                    SELECT
                        id,
                        content,
                        metadata,
                        ROW_NUMBER() OVER (
                            ORDER BY embedding <=> %s::vector
                        ) AS rank
                    FROM document_chunks
                    WHERE manufacturer = %s
                      AND model = %s
                      AND (
                          %s IS NULL
                          OR model_year = %s
                      )
                    ORDER BY embedding <=> %s::vector
                    LIMIT 20
                ),

                keyword_search AS (
                    SELECT
                        id,
                        content,
                        metadata,
                        ROW_NUMBER() OVER (
                            ORDER BY ts_rank_cd(
                                fts_vector,
                                plainto_tsquery('english', %s)
                            ) DESC
                        ) AS rank
                    FROM document_chunks
                    WHERE manufacturer = %s
                      AND model = %s
                      AND (
                          %s IS NULL
                          OR model_year = %s
                      )
                      AND fts_vector @@ plainto_tsquery(
                          'english',
                          %s
                      )
                    ORDER BY ts_rank_cd(
                        fts_vector,
                        plainto_tsquery('english', %s)
                    ) DESC
                    LIMIT 20
                )

                SELECT
                    COALESCE(s.id, k.id) AS id,
                    COALESCE(s.content, k.content) AS content,
                    COALESCE(s.metadata, k.metadata) AS metadata,
                    (
                        COALESCE(1.0 / (60 + s.rank), 0.0)
                        +
                        COALESCE(1.0 / (60 + k.rank), 0.0)
                    ) AS rrf_score
                FROM semantic_search s
                FULL OUTER JOIN keyword_search k
                    ON s.id = k.id
                ORDER BY rrf_score DESC
                LIMIT %s;
                """,
                (
                    query_embedding,
                    target_manufacturer,
                    target_model,
                    target_model_year,
                    target_model_year,
                    query_embedding,
                    query,
                    target_manufacturer,
                    target_model,
                    target_model_year,
                    target_model_year,
                    query,
                    query,
                    top_k,
                ),
            )

            return cur.fetchall()