import json
from db import DBConnection

def inspect_database():
    db = DBConnection()
    
    with db.conn.cursor() as cur:
        # 1. Total chunks
        cur.execute("SELECT COUNT(*) FROM document_chunks;")
        print(f"Total chunks in database: {cur.fetchone()[0]}")
        
        # 2. Breakdown by type
        cur.execute("SELECT chunk_type, COUNT(*) FROM document_chunks GROUP BY chunk_type;")
        print("\n--- Chunks by Type ---")
        for chunk_type, count in cur.fetchall():
            print(f"{chunk_type}: {count}")
            
        # 3. View the latest code blocks
        print("\n--- Latest Code Blocks ---")
        cur.execute("""
            SELECT content, metadata 
            FROM document_chunks 
            WHERE chunk_type = 'mixed context' 
            ORDER BY id DESC LIMIT 3;
        """)
        for content, metadata in cur.fetchall():
            print(f"\nScope: {metadata.get('code_scope')}")
            print(f"Content:\n{content[:150]}...")

if __name__ == "__main__":
    inspect_database()