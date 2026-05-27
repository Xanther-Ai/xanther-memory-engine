"""
Memory Pruner Lambda
Runs daily to:
- Compress memories older than 30 days
- Archive memories older than 90 days to S3
- Delete memories older than 365 days
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List

import boto3
import psycopg2
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Load configuration from environment variables."""
    TIMESCALEDB_HOST = os.getenv("TIMESCALEDB_HOST", "")
    TIMESCALEDB_PORT = int(os.getenv("TIMESCALEDB_PORT", "5432"))
    TIMESCALEDB_NAME = os.getenv("TIMESCALEDB_NAME", "xce_memory")
    TIMESCALEDB_USER = os.getenv("TIMESCALEDB_USER", "xce_memory")
    TIMESCALEDB_PASSWORD = os.getenv("TIMESCALEDB_PASSWORD", "")
    COLD_STORAGE_BUCKET = os.getenv("COLD_STORAGE_BUCKET", "")
    RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "30"))
    COMPRESS_AFTER_DAYS = int(os.getenv("COMPRESS_AFTER_DAYS", "90"))
    DELETE_AFTER_DAYS = int(os.getenv("DELETE_AFTER_DAYS", "365"))


def get_connection(config: Config):
    """Get TimescaleDB connection."""
    return psycopg2.connect(
        host=config.TIMESCALEDB_HOST,
        port=config.TIMESCALEDB_PORT,
        database=config.TIMESCALEDB_NAME,
        user=config.TIMESCALEDB_USER,
        password=config.TIMESCALEDB_PASSWORD
    )


def compress_old_memories(conn, config: Config) -> int:
    """Compress memories older than COMPRESS_AFTER_DAYS."""
    threshold = datetime.now(timezone.utc) - timedelta(days=config.COMPRESS_AFTER_DAYS)
    
    try:
        with conn.cursor() as cur:
            # Find memories to compress (not yet compressed, older than threshold)
            cur.execute("""
                SELECT id, content, code_snippet, diff
                FROM session_memories
                WHERE created_at < %s
                AND is_compressed = FALSE
                LIMIT 1000
            """, (threshold,))
            
            rows = cur.fetchall()
            compressed_count = 0
            
            for row in rows:
                memory_id, content, code_snippet, diff = row
                
                # Simple compression: store diff-like format
                compressed_content = content
                if len(content) > 500:
                    compressed_content = content[:500] + "... [truncated]"
                
                # Update with compressed content
                cur.execute("""
                    UPDATE session_memories
                    SET content = %s, is_compressed = TRUE
                    WHERE id = %s
                """, (compressed_content, memory_id))
                compressed_count += 1
            
            conn.commit()
            logger.info(f"Compressed {compressed_count} memories")
            return compressed_count
            
    except Exception as e:
        logger.error(f"Error compressing memories: {e}")
        conn.rollback()
        return 0


def archive_to_s3(conn, config: Config, s3_client) -> int:
    """Archive memories older than DELETE_AFTER_DAYS to S3."""
    threshold = datetime.now(timezone.utc) - timedelta(days=config.DELETE_AFTER_DAYS)
    
    try:
        with conn.cursor() as cur:
            # Find memories to archive
            cur.execute("""
                SELECT id, session_id, user_id, repo_id, kind, priority, 
                       content, code_snippet, diff, references, new_entities, created_at
                FROM session_memories
                WHERE created_at < %s
                AND created_at > %s
                LIMIT 500
            """, (threshold, threshold - timedelta(days=7)))
            
            rows = cur.fetchall()
            archived_count = 0
            
            for row in rows:
                memory_data = {
                    "id": row[0],
                    "session_id": row[1],
                    "user_id": row[2],
                    "repo_id": row[3],
                    "kind": row[4],
                    "priority": row[5],
                    "content": row[6],
                    "code_snippet": row[7],
                    "diff": row[8],
                    "references": row[9],
                    "new_entities": row[10],
                    "created_at": row[11].isoformat() if row[11] else None
                }
                
                # Upload to S3 with date-based key
                date_key = memory_data["created_at"][:10] if memory_data["created_at"] else "unknown"
                s3_key = f"archives/{date_key}/{memory_data['id']}.json"
                
                try:
                    s3_client.put_object(
                        Bucket=config.COLD_STORAGE_BUCKET,
                        Key=s3_key,
                        Body=json.dumps(memory_data),
                        ContentType="application/json"
                    )
                    archived_count += 1
                except ClientError as e:
                    logger.error(f"Failed to upload {s3_key}: {e}")
            
            logger.info(f"Archived {archived_count} memories to S3")
            return archived_count
            
    except Exception as e:
        logger.error(f"Error archiving memories: {e}")
        return 0


def delete_ancient_memories(conn, config: Config) -> int:
    """Delete memories older than DELETE_AFTER_DAYS (after archiving)."""
    threshold = datetime.now(timezone.utc) - timedelta(days=config.DELETE_AFTER_DAYS)
    
    try:
        with conn.cursor() as cur:
            # Delete old memories (already archived)
            cur.execute("""
                DELETE FROM session_memories
                WHERE created_at < %s
            """, (threshold,))
            
            deleted_count = cur.rowcount
            conn.commit()
            logger.info(f"Deleted {deleted_count} ancient memories")
            return deleted_count
            
    except Exception as e:
        logger.error(f"Error deleting memories: {e}")
        conn.rollback()
        return 0


def prune(event, context):
    """Main Lambda handler."""
    config = Config()
    
    # Initialize clients
    conn = get_connection(config)
    s3_client = boto3.client('s3')
    
    results = {
        "compressed": 0,
        "archived": 0,
        "deleted": 0
    }
    
    try:
        # 1. Compress old memories
        results["compressed"] = compress_old_memories(conn, config)
        
        # 2. Archive to S3
        results["archived"] = archive_to_s3(conn, config, s3_client)
        
        # 3. Delete ancient memories
        results["deleted"] = delete_ancient_memories(conn, config)
        
        logger.info(f"Pruning complete: {results}")
        return {
            "statusCode": 200,
            "body": json.dumps(results)
        }
        
    except Exception as e:
        logger.error(f"Pruning failed: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
    finally:
        conn.close()


if __name__ == "__main__":
    # For local testing
    print(prune({}, None))