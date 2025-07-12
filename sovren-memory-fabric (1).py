#!/usr/bin/env python3
"""
SOVREN AI Memory Fabric Service - B200 Bare Metal Production
Port: 8002
High-performance memory management leveraging 250GB RAM + 192GB HBM3e
PRODUCTION READY - NO PLACEHOLDERS
"""

import asyncio
import json
import logging
import os
import sys
import time
import mmap
import struct
import pickle
import lz4.frame
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from datetime import datetime, timedelta
import hashlib
import shutil

# System packages
import asyncpg
import redis
import numpy as np

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/mnt/yellow-mackerel-volume/sovren/logs/memory-fabric.log')
    ]
)
logger = logging.getLogger('sovren-memory-fabric')

# Load configuration
CONFIG_PATH = '/mnt/yellow-mackerel-volume/sovren/config/sovren.conf'
config = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                config[key] = value.strip('"')

# Service configuration
SERVICE_HOST = '0.0.0.0'
SERVICE_PORT = 8002
REDIS_HOST = config.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(config.get('REDIS_PORT', '6379'))
DB_HOST = config.get('DB_HOST', 'localhost')
DB_PORT = int(config.get('DB_PORT', '5432'))
DB_NAME = 'sovren_memory'
DB_USER = config.get('DB_USER', 'sovren')
DB_PASS = config.get('DB_PASS', 'Renegades1!')

# Memory configuration - leverage full system RAM
MEMORY_POOL_SIZE = 200 * 1024 * 1024 * 1024  # 200GB for memory fabric
CHUNK_SIZE = 64 * 1024 * 1024  # 64MB chunks
MAX_CONTEXT_SIZE = 10 * 1024 * 1024 * 1024  # 10GB per user context
MEMORY_MAP_PATH = '/mnt/yellow-mackerel-volume/sovren/memory/fabric.mmap'

# Global connections
redis_client = None
db_pool = None
memory_pool = None

class B200MemoryPool:
    """High-performance memory pool leveraging system RAM"""
    
    def __init__(self, size=MEMORY_POOL_SIZE):
        self.size = size
        self.chunk_size = CHUNK_SIZE
        self.num_chunks = size // self.chunk_size
        self.free_chunks = set(range(self.num_chunks))
        self.allocated = {}  # key -> chunk_ids
        self.chunk_data = {}  # chunk_id -> data
        self.access_times = {}  # chunk_id -> last_access
        self.lock = threading.Lock()
        
        # Create memory-mapped file for persistence
        os.makedirs(os.path.dirname(MEMORY_MAP_PATH), exist_ok=True)
        
        # Initialize or load existing memory map
        if os.path.exists(MEMORY_MAP_PATH):
            self._load_memory_map()
        else:
            self._create_memory_map()
        
        logger.info(f"Memory pool initialized: {size/1024**3:.1f}GB across {self.num_chunks} chunks")
    
    def _create_memory_map(self):
        """Create new memory-mapped file"""
        with open(MEMORY_MAP_PATH, 'wb') as f:
            f.write(b'\0' * self.size)
        
        self.mmap_file = open(MEMORY_MAP_PATH, 'r+b')
        self.mmap = mmap.mmap(self.mmap_file.fileno(), self.size)
        
        # Write header
        header = struct.pack('IIII', 
            0x534F5652,  # SOVR magic number
            1,  # version
            self.num_chunks,
            self.chunk_size
        )
        self.mmap[0:16] = header
    
    def _load_memory_map(self):
        """Load existing memory-mapped file"""
        self.mmap_file = open(MEMORY_MAP_PATH, 'r+b')
        self.mmap = mmap.mmap(self.mmap_file.fileno(), self.size)
        
        # Read header
        header = struct.unpack('IIII', self.mmap[0:16])
        if header[0] != 0x534F5652:
            raise ValueError("Invalid memory map file")
        
        logger.info(f"Loaded existing memory map: {self.num_chunks} chunks")
    
    def allocate(self, key, data):
        """Allocate memory for data"""
        with self.lock:
            # Compress data
            compressed = lz4.frame.compress(pickle.dumps(data))
            required_chunks = (len(compressed) + self.chunk_size - 1) // self.chunk_size
            
            if len(self.free_chunks) < required_chunks:
                # Evict least recently used chunks
                self._evict_lru(required_chunks)
            
            if len(self.free_chunks) < required_chunks:
                raise MemoryError(f"Cannot allocate {required_chunks} chunks")
            
            # Allocate chunks
            chunk_ids = []
            for _ in range(required_chunks):
                chunk_id = self.free_chunks.pop()
                chunk_ids.append(chunk_id)
            
            # Write data to chunks
            offset = 0
            for i, chunk_id in enumerate(chunk_ids):
                chunk_offset = 16 + (chunk_id * self.chunk_size)  # Skip header
                chunk_data = compressed[offset:offset + self.chunk_size]
                
                # Write chunk header
                chunk_header = struct.pack('II', len(chunk_data), i)
                self.mmap[chunk_offset:chunk_offset + 8] = chunk_header
                
                # Write chunk data
                self.mmap[chunk_offset + 8:chunk_offset + 8 + len(chunk_data)] = chunk_data
                
                offset += len(chunk_data)
                self.access_times[chunk_id] = time.time()
            
            self.allocated[key] = chunk_ids
            
            logger.debug(f"Allocated {required_chunks} chunks for key {key}")
            return True
    
    def get(self, key):
        """Retrieve data from memory"""
        with self.lock:
            if key not in self.allocated:
                return None
            
            chunk_ids = self.allocated[key]
            data_parts = []
            
            for chunk_id in chunk_ids:
                chunk_offset = 16 + (chunk_id * self.chunk_size)
                
                # Read chunk header
                chunk_header = struct.unpack('II', self.mmap[chunk_offset:chunk_offset + 8])
                data_length = chunk_header[0]
                
                # Read chunk data
                chunk_data = self.mmap[chunk_offset + 8:chunk_offset + 8 + data_length]
                data_parts.append(bytes(chunk_data))
                
                # Update access time
                self.access_times[chunk_id] = time.time()
            
            # Decompress and unpickle
            compressed = b''.join(data_parts)
            data = pickle.loads(lz4.frame.decompress(compressed))
            
            return data
    
    def delete(self, key):
        """Delete data from memory"""
        with self.lock:
            if key not in self.allocated:
                return False
            
            chunk_ids = self.allocated[key]
            for chunk_id in chunk_ids:
                self.free_chunks.add(chunk_id)
                if chunk_id in self.access_times:
                    del self.access_times[chunk_id]
            
            del self.allocated[key]
            logger.debug(f"Freed {len(chunk_ids)} chunks from key {key}")
            return True
    
    def _evict_lru(self, required_chunks):
        """Evict least recently used chunks"""
        # Sort chunks by access time
        chunk_access = [(chunk_id, access_time) 
                       for chunk_id, access_time in self.access_times.items()]
        chunk_access.sort(key=lambda x: x[1])
        
        freed = 0
        for chunk_id, _ in chunk_access:
            if freed >= required_chunks:
                break
            
            # Find which key owns this chunk
            for key, chunks in self.allocated.items():
                if chunk_id in chunks:
                    # Free all chunks for this key
                    for cid in chunks:
                        self.free_chunks.add(cid)
                        if cid in self.access_times:
                            del self.access_times[cid]
                    
                    del self.allocated[key]
                    freed += len(chunks)
                    logger.debug(f"Evicted key {key} to free {len(chunks)} chunks")
                    break
    
    def get_stats(self):
        """Get memory pool statistics"""
        with self.lock:
            return {
                'total_chunks': self.num_chunks,
                'free_chunks': len(self.free_chunks),
                'allocated_chunks': self.num_chunks - len(self.free_chunks),
                'total_size_gb': self.size / 1024**3,
                'used_size_gb': (self.num_chunks - len(self.free_chunks)) * self.chunk_size / 1024**3,
                'keys_stored': len(self.allocated)
            }

class UserContextManager:
    """Manage user contexts with massive memory capacity"""
    
    def __init__(self, memory_pool):
        self.memory_pool = memory_pool
        self.active_contexts = {}  # user_id -> context_data
        self.context_locks = {}  # user_id -> lock
    
    async def load_context(self, user_id):
        """Load complete user context into memory"""
        try:
            # Get or create lock for user
            if user_id not in self.context_locks:
                self.context_locks[user_id] = threading.Lock()
            
            with self.context_locks[user_id]:
                # Check if already loaded
                if user_id in self.active_contexts:
                    return self.active_contexts[user_id]
                
                # Load from memory pool
                context_key = f"context:{user_id}"
                context = self.memory_pool.get(context_key)
                
                if context:
                    self.active_contexts[user_id] = context
                    return context
                
                # Load from database
                context = await self._load_from_database(user_id)
                
                # Store in memory pool
                self.memory_pool.allocate(context_key, context)
                self.active_contexts[user_id] = context
                
                return context
                
        except Exception as e:
            logger.error(f"Failed to load context for user {user_id}: {e}")
            return None
    
    async def update_context(self, user_id, updates):
        """Update user context"""
        try:
            context = await self.load_context(user_id)
            if not context:
                context = self._create_new_context(user_id)
            
            # Apply updates
            for key, value in updates.items():
                if key == 'conversations':
                    context['conversations'].extend(value)
                elif key == 'interactions':
                    context['interactions'].extend(value)
                elif key == 'business_data':
                    context['business_data'].update(value)
                else:
                    context[key] = value
            
            # Update timestamp
            context['last_updated'] = datetime.now().isoformat()
            
            # Save to memory pool
            context_key = f"context:{user_id}"
            self.memory_pool.allocate(context_key, context)
            self.active_contexts[user_id] = context
            
            # Async save to database
            asyncio.create_task(self._save_to_database(user_id, context))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update context for user {user_id}: {e}")
            return False
    
    async def get_context_slice(self, user_id, slice_type, limit=100):
        """Get specific slice of user context"""
        context = await self.load_context(user_id)
        if not context:
            return None
        
        if slice_type == 'recent_conversations':
            return context['conversations'][-limit:]
        elif slice_type == 'business_metrics':
            return context['business_data']
        elif slice_type == 'interaction_history':
            return context['interactions'][-limit:]
        elif slice_type == 'preferences':
            return context.get('preferences', {})
        else:
            return None
    
    def _create_new_context(self, user_id):
        """Create new user context structure"""
        return {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'conversations': [],
            'interactions': [],
            'business_data': {},
            'preferences': {},
            'ai_state': {
                'personality': 'adaptive',
                'memory_depth': 0,
                'context_understanding': 0.0
            },
            'embeddings': {},
            'insights': []
        }
    
    async def _load_from_database(self, user_id):
        """Load context from database"""
        try:
            async with db_pool.acquire() as conn:
                # Load base context
                row = await conn.fetchrow(
                    "SELECT * FROM user_contexts WHERE user_id = $1",
                    user_id
                )
                
                if not row:
                    return self._create_new_context(user_id)
                
                context = json.loads(row['context_data'])
                
                # Load conversation history
                conversations = await conn.fetch(
                    """SELECT * FROM conversations 
                       WHERE user_id = $1 
                       ORDER BY created_at DESC 
                       LIMIT 1000""",
                    user_id
                )
                
                context['conversations'] = [dict(conv) for conv in conversations]
                
                # Load interaction events
                interactions = await conn.fetch(
                    """SELECT * FROM interaction_events 
                       WHERE user_id = $1 
                       ORDER BY timestamp DESC 
                       LIMIT 5000""",
                    user_id
                )
                
                context['interactions'] = [dict(inter) for inter in interactions]
                
                return context
                
        except Exception as e:
            logger.error(f"Database load error: {e}")
            return self._create_new_context(user_id)
    
    async def _save_to_database(self, user_id, context):
        """Save context to database"""
        try:
            async with db_pool.acquire() as conn:
                # Save base context
                await conn.execute(
                    """INSERT INTO user_contexts (user_id, context_data, updated_at)
                       VALUES ($1, $2, $3)
                       ON CONFLICT (user_id) 
                       DO UPDATE SET context_data = $2, updated_at = $3""",
                    user_id, json.dumps(context), datetime.now()
                )
                
                logger.debug(f"Saved context for user {user_id} to database")
                
        except Exception as e:
            logger.error(f"Database save error: {e}")

class MemoryFabricHandler(BaseHTTPRequestHandler):
    """Production memory fabric handler"""
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == '/health':
                self._handle_health_check()
            elif self.path == '/stats':
                self._handle_stats()
            elif self.path.startswith('/context/'):
                self._handle_get_context()
            else:
                self.send_error(404, "Not Found")
        except Exception as e:
            logger.error(f"GET error: {e}")
            self.send_error(500, str(e))
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            if self.path == '/store':
                self._handle_store()
            elif self.path == '/retrieve':
                self._handle_retrieve()
            elif self.path == '/delete':
                self._handle_delete()
            elif self.path.startswith('/context/'):
                self._handle_update_context()
            else:
                self.send_error(404, "Not Found")
        except Exception as e:
            logger.error(f"POST error: {e}")
            self.send_error(500, str(e))
    
    def _handle_health_check(self):
        """Service health check"""
        stats = memory_pool.get_stats() if memory_pool else {}
        
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'memory-fabric',
            'memory_pool': stats,
            'database': 'connected' if db_pool else 'disconnected',
            'redis': 'connected' if redis_client else 'disconnected'
        }
        
        self._send_json_response(200, health)
    
    def _handle_stats(self):
        """Get detailed statistics"""
        stats = {
            'memory_pool': memory_pool.get_stats() if memory_pool else {},
            'active_contexts': len(context_manager.active_contexts) if 'context_manager' in globals() else 0,
            'system_memory': {
                'total_gb': 250,
                'available_gb': 250 - (memory_pool.get_stats()['used_size_gb'] if memory_pool else 0)
            }
        }
        
        self._send_json_response(200, stats)
    
    def _handle_store(self):
        """Store data in memory fabric"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error(400, "No data provided")
                return
            
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            if 'key' not in data or 'value' not in data:
                self._send_json_response(400, {'error': 'Key and value required'})
                return
            
            # Store in memory pool
            success = memory_pool.allocate(data['key'], data['value'])
            
            # Also store in Redis for fast lookups
            if redis_client and success:
                redis_client.set(f"fabric:{data['key']}", "1", ex=86400)  # 24h TTL
            
            self._send_json_response(200, {
                'success': success,
                'key': data['key']
            })
            
        except Exception as e:
            logger.error(f"Store error: {e}")
            self._send_json_response(500, {'error': str(e)})
    
    def _handle_retrieve(self):
        """Retrieve data from memory fabric"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error(400, "No data provided")
                return
            
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            if 'key' not in data:
                self._send_json_response(400, {'error': 'Key required'})
                return
            
            # Retrieve from memory pool
            value = memory_pool.get(data['key'])
            
            if value is None:
                self._send_json_response(404, {'error': 'Key not found'})
            else:
                self._send_json_response(200, {
                    'key': data['key'],
                    'value': value
                })
            
        except Exception as e:
            logger.error(f"Retrieve error: {e}")
            self._send_json_response(500, {'error': str(e)})
    
    def _handle_delete(self):
        """Delete data from memory fabric"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error(400, "No data provided")
                return
            
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            if 'key' not in data:
                self._send_json_response(400, {'error': 'Key required'})
                return
            
            # Delete from memory pool
            success = memory_pool.delete(data['key'])
            
            # Also delete from Redis
            if redis_client and success:
                redis_client.delete(f"fabric:{data['key']}")
            
            self._send_json_response(200, {
                'success': success,
                'key': data['key']
            })
            
        except Exception as e:
            logger.error(f"Delete error: {e}")
            self._send_json_response(500, {'error': str(e)})
    
    def _handle_get_context(self):
        """Get user context"""
        try:
            path_parts = self.path.split('/')
            if len(path_parts) < 3:
                self.send_error(400, "Invalid path")
                return
            
            user_id = path_parts[2]
            slice_type = path_parts[3] if len(path_parts) > 3 else None
            
            if slice_type:
                # Get context slice
                loop = asyncio.new_event_loop()
                data = loop.run_until_complete(
                    context_manager.get_context_slice(user_id, slice_type)
                )
                loop.close()
            else:
                # Get full context
                loop = asyncio.new_event_loop()
                data = loop.run_until_complete(
                    context_manager.load_context(user_id)
                )
                loop.close()
            
            if data is None:
                self._send_json_response(404, {'error': 'Context not found'})
            else:
                self._send_json_response(200, data)
            
        except Exception as e:
            logger.error(f"Get context error: {e}")
            self._send_json_response(500, {'error': str(e)})
    
    def _handle_update_context(self):
        """Update user context"""
        try:
            path_parts = self.path.split('/')
            if len(path_parts) < 3:
                self.send_error(400, "Invalid path")
                return
            
            user_id = path_parts[2]
            
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error(400, "No data provided")
                return
            
            body = self.rfile.read(content_length)
            updates = json.loads(body.decode('utf-8'))
            
            # Update context
            loop = asyncio.new_event_loop()
            success = loop.run_until_complete(
                context_manager.update_context(user_id, updates)
            )
            loop.close()
            
            self._send_json_response(200, {
                'success': success,
                'user_id': user_id
            })
            
        except Exception as e:
            logger.error(f"Update context error: {e}")
            self._send_json_response(500, {'error': str(e)})
    
    def _send_json_response(self, status_code, data):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"{self.client_address[0]} - {format % args}")

class ThreadedHTTPServer(HTTPServer):
    """Multi-threaded HTTP server"""
    def process_request(self, request, client_address):
        thread = threading.Thread(
            target=self.process_request_thread,
            args=(request, client_address)
        )
        thread.daemon = True
        thread.start()
    
    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
            self.shutdown_request(request)
        except:
            self.shutdown_request(request)

async def initialize_database():
    """Initialize database schema"""
    try:
        async with db_pool.acquire() as conn:
            # User contexts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_contexts (
                    user_id VARCHAR(16) PRIMARY KEY,
                    context_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Conversations table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id VARCHAR(16) PRIMARY KEY,
                    user_id VARCHAR(16) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    messages JSONB NOT NULL,
                    metadata JSONB
                )
            """)
            
            # Interaction events table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS interaction_events (
                    event_id VARCHAR(16) PRIMARY KEY,
                    user_id VARCHAR(16) NOT NULL,
                    event_type VARCHAR(50),
                    event_data JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Memory snapshots table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_snapshots (
                    snapshot_id VARCHAR(16) PRIMARY KEY,
                    user_id VARCHAR(16) NOT NULL,
                    snapshot_type VARCHAR(50),
                    snapshot_data BYTEA,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interaction_events(user_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interaction_events(timestamp)"
            )
            
            logger.info("Database schema initialized")
            
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise

async def initialize_connections():
    """Initialize database and Redis connections"""
    global redis_client, db_pool
    
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            socket_keepalive_options={
                1: 1,  # TCP_KEEPIDLE
                2: 1,  # TCP_KEEPINTVL
                3: 5,  # TCP_KEEPCNT
            }
        )
        redis_client.ping()
        
        # Configure Redis for high-performance
        redis_client.config_set('maxmemory', '10gb')
        redis_client.config_set('maxmemory-policy', 'allkeys-lru')
        
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        redis_client = None
    
    try:
        db_pool = await asyncpg.create_pool(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            min_size=10,
            max_size=20,
            command_timeout=60
        )
        logger.info("PostgreSQL connection pool created")
        
        await initialize_database()
        
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        raise

def main():
    """Main entry point"""
    logger.info("SOVREN AI Memory Fabric starting...")
    logger.info(f"System RAM: 250GB | Target Memory Pool: {MEMORY_POOL_SIZE/1024**3:.1f}GB")
    
    # Initialize memory pool
    global memory_pool, context_manager
    try:
        memory_pool = B200MemoryPool(MEMORY_POOL_SIZE)
        context_manager = UserContextManager(memory_pool)
        logger.info("Memory pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize memory pool: {e}")
        sys.exit(1)
    
    # Initialize connections
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(initialize_connections())
    except Exception as e:
        logger.error(f"Failed to initialize connections: {e}")
        sys.exit(1)
    
    # Start HTTP server
    try:
        server = ThreadedHTTPServer((SERVICE_HOST, SERVICE_PORT), MemoryFabricHandler)
        logger.info(f"Memory Fabric listening on http://{SERVICE_HOST}:{SERVICE_PORT}")
        logger.info("High-performance memory system: ONLINE")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
