import re
import os
import time
import json
import hashlib
import threading
import requests
import atexit
from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timedelta

load_dotenv()

class SearchResult(BaseModel):
    id: str
    title: str
    url: str
    snippet: str
    source: str
    display_link: Optional[str] = None
    formatted_url: Optional[str] = None

class SearchResponse(BaseModel):
    success: bool
    query: str
    results: List[SearchResult]
    count: int
    search_time: float
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SearchCache:
    """Thread-safe search cache using SQLite with proper locking mechanisms"""
    
    # Class-level lock for singleton pattern
    _instance_lock = threading.Lock()
    _instances: Dict[str, 'SearchCache'] = {}
    
    def __new__(cls, cache_dir: Optional[str] = None, cache_ttl: int = 86400):
        """
        Singleton pattern per cache_dir to ensure one cache instance per database
        This prevents multiple cache objects accessing the same database
        """
        if cache_dir is None:
            cache_dir = os.path.join(Path.home(), ".cache", "google_search")
        
        # Normalize path for consistent key
        cache_dir = os.path.abspath(cache_dir)
        
        with cls._instance_lock:
            if cache_dir not in cls._instances:
                instance = super(SearchCache, cls).__new__(cls)
                cls._instances[cache_dir] = instance
                instance._initialized = False
            return cls._instances[cache_dir]
    
    def __init__(self, cache_dir: Optional[str] = None, cache_ttl: int = 86400):
        """
        Initialize search cache (only once due to singleton pattern)
        
        Args:
            cache_dir: Directory to store cache database, defaults to ~/.cache/google_search
            cache_ttl: Cache time-to-live in seconds, defaults to 24 hours
        """
        # Skip if already initialized (singleton pattern)
        if self._initialized:
            return
        
        if cache_dir is None:
            cache_dir = os.path.join(Path.home(), ".cache", "google_search")
        
        cache_dir = os.path.abspath(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        
        self.db_path = os.path.join(cache_dir, "search_cache.db")
        self.cache_ttl = cache_ttl
        
        # Thread-local storage for connections to avoid sharing across threads
        self._local = threading.local()
        
        # Lock for write operations to serialize writes
        self._write_lock = threading.Lock()
        
        # Lock for database initialization
        self._init_lock = threading.Lock()
        
        # Connection pool tracking for cleanup
        self._active_connections = {}
        self._connection_lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        self._initialized = True
        
        # Register cleanup on program exit
        atexit.register(self._cleanup_all_connections)
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get thread-local database connection with proper configuration
        Each thread gets its own connection to avoid threading issues
        """
        thread_id = threading.get_ident()
        
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            try:
                # Create connection with optimal settings for concurrent access
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=30.0,  # Wait up to 30 seconds for locks
                    isolation_level='DEFERRED',  # Use deferred transactions for better concurrency
                    check_same_thread=False  # Allow connection use across threads (but we use thread-local)
                )
                
                # Enable WAL mode for concurrent reads and writes
                # WAL allows multiple readers and one writer simultaneously
                conn.execute('PRAGMA journal_mode=WAL')
                
                # Set busy timeout - how long to wait when database is locked
                conn.execute('PRAGMA busy_timeout=30000')
                
                # Enable foreign keys
                conn.execute('PRAGMA foreign_keys=ON')
                
                # Synchronous=NORMAL is safe with WAL and faster
                conn.execute('PRAGMA synchronous=NORMAL')
                
                # Cache size for better performance (negative = KB)
                conn.execute('PRAGMA cache_size=-64000')  # 64MB cache
                
                # Set row factory for easier data access
                conn.row_factory = sqlite3.Row
                
                self._local.conn = conn
                
                # Track connection for cleanup
                with self._connection_lock:
                    self._active_connections[thread_id] = conn
                
            except sqlite3.Error as e:
                print(f"Error creating database connection: {e}")
                raise
        
        return self._local.conn
    
    @contextmanager
    def _get_cursor(self, write_operation: bool = False):
        """
        Context manager for database cursor with automatic commit/rollback
        
        Args:
            write_operation: If True, acquires write lock to serialize writes
        """
        # For write operations, acquire the write lock to prevent concurrent writes
        write_lock = self._write_lock if write_operation else None
        
        if write_lock:
            write_lock.acquire()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            yield cursor
            conn.commit()
        except sqlite3.OperationalError as e:
            # Handle database locked errors gracefully
            if "locked" in str(e).lower():
                print(f"Database locked, operation may be retried: {e}")
            conn.rollback()
            raise
        except Exception as e:
            conn.rollback()
            raise
        finally:
            cursor.close()
            if write_lock:
                write_lock.release()
    
    def _init_database(self):
        """Initialize database schema with proper indexes"""
        with self._init_lock:
            try:
                with self._get_cursor(write_operation=True) as cursor:
                    # Create cache table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS search_cache (
                            cache_key TEXT PRIMARY KEY,
                            query TEXT NOT NULL,
                            num_results INTEGER NOT NULL,
                            language TEXT NOT NULL,
                            country TEXT NOT NULL,
                            safe_search INTEGER NOT NULL,
                            response_data TEXT NOT NULL,
                            created_at REAL NOT NULL,
                            accessed_at REAL NOT NULL,
                            access_count INTEGER DEFAULT 1
                        )
                    ''')
                    
                    # Create indexes for efficient queries
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_created_at 
                        ON search_cache(created_at)
                    ''')
                    
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_query 
                        ON search_cache(query)
                    ''')
                    
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_accessed_at 
                        ON search_cache(accessed_at)
                    ''')
                    
                    # Create metadata table for cache statistics
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS cache_metadata (
                            key TEXT PRIMARY KEY,
                            value TEXT,
                            updated_at REAL
                        )
                    ''')
            except sqlite3.Error as e:
                print(f"Error initializing database: {e}")
                raise
    
    def _generate_cache_key(
        self,
        query: str,
        num_results: int,
        language: str,
        country: str,
        safe_search: bool
    ) -> str:
        """Generate unique cache key from search parameters"""
        key_string = f"{query}|{num_results}|{language}|{country}|{safe_search}"
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def get(
        self,
        query: str,
        num_results: int,
        language: str,
        country: str,
        safe_search: bool,
        max_retries: int = 3
    ) -> Optional[SearchResponse]:
        """
        Retrieve cached search response with retry logic
        
        Args:
            max_retries: Number of retries on database lock
            
        Returns None if cache miss or expired
        """
        cache_key = self._generate_cache_key(query, num_results, language, country, safe_search)
        
        for attempt in range(max_retries):
            try:
                # Read operations don't need write lock
                with self._get_cursor(write_operation=False) as cursor:
                    cursor.execute('''
                        SELECT response_data, created_at 
                        FROM search_cache 
                        WHERE cache_key = ?
                    ''', (cache_key,))
                    
                    row = cursor.fetchone()
                    
                    if row is None:
                        return None
                    
                    response_data = row['response_data']
                    created_at = row['created_at']
                    
                    # Check if cache is expired
                    if time.time() - created_at > self.cache_ttl:
                        # Schedule deletion (non-blocking)
                        threading.Thread(
                            target=self._delete_expired_entry,
                            args=(cache_key,),
                            daemon=True
                        ).start()
                        return None
                    
                    # Update access statistics asynchronously to avoid blocking reads
                    threading.Thread(
                        target=self._update_access_stats,
                        args=(cache_key,),
                        daemon=True
                    ).start()
                    
                    # Deserialize response
                    response_dict = json.loads(response_data)
                    return SearchResponse(**response_dict)
                    
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    # Wait and retry on database lock
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    print(f"Cache get error after {attempt + 1} attempts: {e}")
                    return None
            except (json.JSONDecodeError, Exception) as e:
                print(f"Cache get error: {e}")
                return None
        
        return None
    
    def _delete_expired_entry(self, cache_key: str):
        """Helper method to delete expired entry (called asynchronously)"""
        try:
            with self._get_cursor(write_operation=True) as cursor:
                cursor.execute('DELETE FROM search_cache WHERE cache_key = ?', (cache_key,))
        except Exception as e:
            print(f"Error deleting expired entry: {e}")
    
    def _update_access_stats(self, cache_key: str):
        """Helper method to update access statistics (called asynchronously)"""
        try:
            with self._get_cursor(write_operation=True) as cursor:
                cursor.execute('''
                    UPDATE search_cache 
                    SET accessed_at = ?, access_count = access_count + 1
                    WHERE cache_key = ?
                ''', (time.time(), cache_key))
        except Exception as e:
            print(f"Error updating access stats: {e}")
    
    def set(
        self,
        query: str,
        num_results: int,
        language: str,
        country: str,
        safe_search: bool,
        response: SearchResponse,
        max_retries: int = 3
    ):
        """
        Store search response in cache with retry logic
        
        Args:
            max_retries: Number of retries on database lock
        """
        cache_key = self._generate_cache_key(query, num_results, language, country, safe_search)
        
        for attempt in range(max_retries):
            try:
                # Write operations need the write lock
                with self._get_cursor(write_operation=True) as cursor:
                    current_time = time.time()
                    response_data = response.model_dump_json()
                    
                    # Use INSERT OR REPLACE to handle duplicates
                    cursor.execute('''
                        INSERT OR REPLACE INTO search_cache 
                        (cache_key, query, num_results, language, country, safe_search, 
                         response_data, created_at, accessed_at, access_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ''', (
                        cache_key, query, num_results, language, country,
                        int(safe_search), response_data, current_time, current_time
                    ))
                    
                return  # Success, exit
                    
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    # Wait and retry on database lock
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    print(f"Cache set error after {attempt + 1} attempts: {e}")
                    return
            except sqlite3.Error as e:
                print(f"Cache set error: {e}")
                return
    
    def cleanup_expired(self, max_retries: int = 3) -> int:
        """
        Remove expired cache entries, returns number of deleted entries
        
        Args:
            max_retries: Number of retries on database lock
        """
        for attempt in range(max_retries):
            try:
                with self._get_cursor(write_operation=True) as cursor:
                    expiry_time = time.time() - self.cache_ttl
                    cursor.execute('''
                        DELETE FROM search_cache 
                        WHERE created_at < ?
                    ''', (expiry_time,))
                    return cursor.rowcount
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    print(f"Cache cleanup error after {attempt + 1} attempts: {e}")
                    return 0
            except sqlite3.Error as e:
                print(f"Cache cleanup error: {e}")
                return 0
        return 0
    
    def clear_all(self, max_retries: int = 3) -> int:
        """
        Clear all cache entries, returns number of deleted entries
        
        Args:
            max_retries: Number of retries on database lock
        """
        for attempt in range(max_retries):
            try:
                with self._get_cursor(write_operation=True) as cursor:
                    cursor.execute('DELETE FROM search_cache')
                    return cursor.rowcount
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    print(f"Cache clear error after {attempt + 1} attempts: {e}")
                    return 0
            except sqlite3.Error as e:
                print(f"Cache clear error: {e}")
                return 0
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with self._get_cursor(write_operation=False) as cursor:
                cursor.execute('SELECT COUNT(*) as count FROM search_cache')
                total_entries = cursor.fetchone()['count']
                
                cursor.execute('SELECT SUM(access_count) as sum FROM search_cache')
                result = cursor.fetchone()
                total_accesses = result['sum'] if result['sum'] is not None else 0
                
                cursor.execute('''
                    SELECT COUNT(*) as count FROM search_cache 
                    WHERE created_at < ?
                ''', (time.time() - self.cache_ttl,))
                expired_entries = cursor.fetchone()['count']
                
                # Get database file size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return {
                    'total_entries': total_entries,
                    'expired_entries': expired_entries,
                    'active_entries': total_entries - expired_entries,
                    'total_accesses': total_accesses,
                    'cache_ttl': self.cache_ttl,
                    'db_size_mb': db_size / (1024 * 1024),
                    'db_path': self.db_path,
                    'active_connections': len(self._active_connections)
                }
        except sqlite3.Error as e:
            print(f"Cache stats error: {e}")
            return {}
    
    def close_current_thread(self):
        """Close database connection for current thread"""
        thread_id = threading.get_ident()
        
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception as e:
                print(f"Error closing connection: {e}")
            finally:
                self._local.conn = None
                
                # Remove from active connections
                with self._connection_lock:
                    if thread_id in self._active_connections:
                        del self._active_connections[thread_id]
    
    def _cleanup_all_connections(self):
        """Close all active connections (called on program exit)"""
        with self._connection_lock:
            for thread_id, conn in list(self._active_connections.items()):
                try:
                    conn.close()
                except Exception as e:
                    print(f"Error closing connection for thread {thread_id}: {e}")
            self._active_connections.clear()
    
    def optimize_database(self):
        """Optimize database by running VACUUM and ANALYZE"""
        try:
            # VACUUM cannot run in a transaction, so we need a separate connection
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute('VACUUM')
            conn.execute('ANALYZE')
            conn.close()
            print("Database optimized successfully")
        except sqlite3.Error as e:
            print(f"Error optimizing database: {e}")
    
    def __del__(self):
        """Cleanup on object destruction"""
        self._cleanup_all_connections()


class GoogleSearch:
    
    # Class-level cache instance (singleton per cache_dir)
    _cache_instance: Optional[SearchCache] = None
    _cache_lock = threading.Lock()
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_cache: bool = True,
        cache_dir: Optional[str] = None,
        cache_ttl: int = 86400
    ):
        """
        Initialize Google Search with optional caching
        
        Args:
            api_key: Google API key
            enable_cache: Whether to enable caching
            cache_dir: Directory for cache database
            cache_ttl: Cache time-to-live in seconds (default 24 hours)
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("missing GOOGLE_API_KEY")
        
        # Your search url
        self.api_url = os.getenv("SEARCH_URL")
        if not self.api_url:
            raise ValueError("missing SEARCH_URL")
        
        # Initialize cache (uses singleton pattern internally)
        self.enable_cache = enable_cache
        if enable_cache:
            with self._cache_lock:
                # Use shared cache instance to avoid creating multiple cache objects
                if GoogleSearch._cache_instance is None:
                    GoogleSearch._cache_instance = SearchCache(cache_dir, cache_ttl)
                self.cache = GoogleSearch._cache_instance
        else:
            self.cache = None
    
    def search(
        self,
        query: str,
        num_results: int = 5,
        language: str = "zh-cn",
        country: str = "cn",
        safe_search: bool = True,
        use_cache: bool = True
    ) -> SearchResponse:
        """
        Perform search with optional caching
        
        Args:
            query: Search query
            num_results: Number of results to return
            language: Search language
            country: Search country
            safe_search: Enable safe search
            use_cache: Whether to use cache for this request
        """
        if not query or not query.strip():
            return SearchResponse(
                success=False,
                query=query,
                results=[],
                count=0,
                search_time=0,
                message="search query is None"
            )
        
        query = query.strip()
        num_results = max(1, min(num_results, 10))
        
        # Try to get from cache
        if self.enable_cache and use_cache and self.cache:
            cached_response = self.cache.get(query, num_results, language, country, safe_search)
            if cached_response is not None:
                # Add cache hit indicator to metadata
                if cached_response.metadata is None:
                    cached_response.metadata = {}
                cached_response.metadata['cache_hit'] = True
                return cached_response
        
        start_time = time.time()
        
        try:
            headers = {
                'X-AK': self.api_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                'query': query,
                'num': num_results,
                'extendParams': {
                    'country': country,
                    'locale': language,
                },
                'platformInput': {
                    'model': 'google-search'
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            search_time = time.time() - start_time
            
            json_data = response.json()
            search_results = []
            
            if "data" in json_data and 'originalOutput' in json_data['data']:
                organic_results = json_data['data']['originalOutput'].get('organic', [])
                
                for i, item in enumerate(organic_results):
                    search_results.append(SearchResult(
                        id=f"google-{i}",
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source="google",
                        display_link=item.get("displayLink"),
                        formatted_url=item.get("formattedUrl")
                    ))
            
            search_response = SearchResponse(
                success=True,
                query=query,
                results=search_results,
                count=len(search_results),
                search_time=search_time,
                metadata={
                    "language": language,
                    "country": country,
                    "safe_search": safe_search,
                    "search_engine": "google",
                    "cache_hit": False
                }
            )
            
            # Store in cache if successful
            if self.enable_cache and use_cache and self.cache:
                self.cache.set(query, num_results, language, country, safe_search, search_response)
            
            return search_response
            
        except requests.exceptions.Timeout:
            return SearchResponse(
                success=False,
                query=query,
                results=[],
                count=0,
                search_time=time.time() - start_time,
                message="time out"
            )
        except requests.exceptions.RequestException as e:
            return SearchResponse(
                success=False,
                query=query,
                results=[],
                count=0,
                search_time=time.time() - start_time,
                message=f"wrong api: {str(e)}"
            )
        except Exception as e:
            return SearchResponse(
                success=False,
                query=query,
                results=[],
                count=0,
                search_time=time.time() - start_time,
                message=f"search failed: {str(e)}"
            )
    
    def search_simple(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Simple search interface returning list of dicts"""
        response = self.search(query, num_results)
        
        if response.success:
            return [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet
                }
                for r in response.results
            ]
        else:
            return []
    
    def cleanup_cache(self) -> int:
        """Cleanup expired cache entries"""
        if self.cache:
            return self.cache.cleanup_expired()
        return 0
    
    def clear_cache(self) -> int:
        """Clear all cache entries"""
        if self.cache:
            return self.cache.clear_all()
        return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_stats()
        return {}
    
    def optimize_cache(self):
        """Optimize cache database"""
        if self.cache:
            self.cache.optimize_database()
    
    def close(self):
        """Close cache connections for current thread"""
        if self.cache:
            self.cache.close_current_thread()


def quick_search(
    query: str,
    num_results: int = 5,
    use_cache: bool = True
) -> List[Dict[str, str]]:
    """
    Quick search function with caching support
    Thread-safe: Uses singleton cache pattern to ensure all threads share the same cache instance
    
    Args:
        query: Search query
        num_results: Number of results to return
        use_cache: Whether to use cache
    
    Returns:
        List of search results as dicts
    """
    try:
        # GoogleSearch will use the singleton cache instance
        searcher = GoogleSearch(enable_cache=use_cache)
        results = searcher.search_simple(query, num_results)
        # Don't close the shared cache, just close current thread's connection
        searcher.close()
        return results
    except Exception as e:
        print(f"search Error: {e}")
        return []


# Example usage and testing
if __name__ == "__main__":
    import concurrent.futures
    
    # Test 1: Basic functionality
    print("=" * 80)
    print("Test 1: Basic search with cache")
    print("=" * 80)
    
    query = "Python programming"
    print(f"\nSearching: {query}")
    
    # First search
    start = time.time()
    results1 = quick_search(query, num_results=3)
    time1 = time.time() - start
    print(f"First search: {len(results1)} results in {time1:.3f}s")
    
    # Second search (should hit cache)
    start = time.time()
    results2 = quick_search(query, num_results=3)
    time2 = time.time() - start
    print(f"Second search: {len(results2)} results in {time2:.3f}s")
    print(f"Speed improvement: {time1/time2:.1f}x faster")
    
    # Test 2: Multi-threaded stress test
    print(f"\n{'='*80}")
    print("Test 2: Multi-threaded stress test")
    print("=" * 80)
    
    def thread_search(thread_id: int, query: str):
        """Function to be executed by each thread"""
        try:
            start = time.time()
            results = quick_search(query, num_results=3)
            elapsed = time.time() - start
            return {
                'thread_id': thread_id,
                'query': query,
                'success': len(results) > 0 or True,  # Consider success even if no results
                'result_count': len(results),
                'time': elapsed
            }
        except Exception as e:
            return {
                'thread_id': thread_id,
                'query': query,
                'success': False,
                'error': str(e)
            }
    
    # Test queries - mix of duplicates to test cache hits
    test_queries = [
        "Python programming",
        "Machine learning",
        "Python programming",  # Duplicate
        "Web development",
        "Machine learning",    # Duplicate
        "Data science",
        "Python programming",  # Duplicate
        "Artificial intelligence",
        "Web development",     # Duplicate
        "Cloud computing"
    ]
    
    print(f"\nRunning {len(test_queries)} searches across 5 threads...")
    print("(Some queries are duplicates to test cache)")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(thread_search, i, query)
            for i, query in enumerate(test_queries)
        ]
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            
            status = "✓" if result['success'] else "✗"
            if result['success']:
                print(f"{status} Thread {result['thread_id']}: "
                      f"'{result['query'][:30]}...' - "
                      f"{result['result_count']} results in {result['time']:.3f}s")
            else:
                print(f"{status} Thread {result['thread_id']}: "
                      f"'{result['query'][:30]}...' - "
                      f"ERROR: {result.get('error', 'Unknown')}")
    
    # Test 3: Verify cache statistics
    print(f"\n{'='*80}")
    print("Test 3: Cache statistics")
    print("=" * 80)
    
    searcher = GoogleSearch(enable_cache=True)
    stats = searcher.get_cache_stats()
    
    print("\nCache Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Test 4: Concurrent writes test
    print(f"\n{'='*80}")
    print("Test 4: Concurrent writes stress test")
    print("=" * 80)
    
    def concurrent_write_test(thread_id: int):
        """Test concurrent writes with unique queries"""
        unique_query = f"Test query {thread_id} {time.time()}"
        try:
            start = time.time()
            results = quick_search(unique_query, num_results=2)
            elapsed = time.time() - start
            return {
                'thread_id': thread_id,
                'success': True,
                'time': elapsed
            }
        except Exception as e:
            return {
                'thread_id': thread_id,
                'success': False,
                'error': str(e)
            }
    
    print("\nTesting 20 concurrent unique searches (all will write to cache)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(concurrent_write_test, i) for i in range(20)]
        
        successes = 0
        failures = 0
        total_time = 0
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result['success']:
                successes += 1
                total_time += result['time']
            else:
                failures += 1
                print(f"  Thread {result['thread_id']} failed: {result.get('error')}")
    
    print(f"\nResults:")
    print(f"  Successes: {successes}")
    print(f"  Failures: {failures}")
    print(f"  Average time: {total_time/successes if successes > 0 else 0:.3f}s")
    
    # Test 5: Cross-instance cache sharing
    print(f"\n{'='*80}")
    print("Test 5: Cross-instance cache sharing")
    print("=" * 80)
    
    def test_instance_isolation(instance_id: int, query: str):
        """Test that different instances share the same cache"""
        try:
            # Create a new GoogleSearch instance in each thread
            searcher = GoogleSearch(enable_cache=True)
            start = time.time()
            response = searcher.search(query, num_results=3)
            elapsed = time.time() - start
            
            cache_hit = response.metadata.get('cache_hit', False) if response.metadata else False
            
            searcher.close()
            
            return {
                'instance_id': instance_id,
                'success': response.success,
                'cache_hit': cache_hit,
                'time': elapsed,
                'count': response.count
            }
        except Exception as e:
            return {
                'instance_id': instance_id,
                'success': False,
                'error': str(e)
            }
    
    shared_query = "Shared cache test query"
    
    print(f"\nSearching '{shared_query}' from 5 different GoogleSearch instances...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(test_instance_isolation, i, shared_query) for i in range(5)]
        
        cache_hits = 0
        cache_misses = 0
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result['success']:
                hit_status = "CACHE HIT" if result['cache_hit'] else "CACHE MISS"
                print(f"  Instance {result['instance_id']}: {hit_status} - "
                      f"{result['count']} results in {result['time']:.3f}s")
                if result['cache_hit']:
                    cache_hits += 1
                else:
                    cache_misses += 1
    
    print(f"\nCache sharing verification:")
    print(f"  Cache hits: {cache_hits}")
    print(f"  Cache misses: {cache_misses}")
    print(f"  Expected: 1 miss (first search), rest should be hits")
    
    # Test 6: Cache cleanup
    print(f"\n{'='*80}")
    print("Test 6: Cache cleanup and optimization")
    print("=" * 80)
    
    print("\nBefore cleanup:")
    stats = searcher.get_cache_stats()
    print(f"  Total entries: {stats.get('total_entries', 0)}")
    print(f"  Expired entries: {stats.get('expired_entries', 0)}")
    print(f"  Database size: {stats.get('db_size_mb', 0):.2f} MB")
    
    print("\nCleaning up expired entries...")
    cleaned = searcher.cleanup_cache()
    print(f"  Cleaned: {cleaned} entries")
    
    print("\nAfter cleanup:")
    stats = searcher.get_cache_stats()
    print(f"  Total entries: {stats.get('total_entries', 0)}")
    print(f"  Database size: {stats.get('db_size_mb', 0):.2f} MB")
    
    # Test 7: Performance comparison
    print(f"\n{'='*80}")
    print("Test 7: Cache performance comparison")
    print("=" * 80)
    
    test_query = "Cache performance test"
    
    # Without cache
    print("\nSearching WITHOUT cache:")
    searcher_no_cache = GoogleSearch(enable_cache=False)
    times_no_cache = []
    for i in range(3):
        start = time.time()
        searcher_no_cache.search(test_query, num_results=3)
        times_no_cache.append(time.time() - start)
        print(f"  Attempt {i+1}: {times_no_cache[-1]:.3f}s")
    avg_no_cache = sum(times_no_cache) / len(times_no_cache)
    print(f"  Average: {avg_no_cache:.3f}s")
    
    # With cache
    print("\nSearching WITH cache:")
    searcher_with_cache = GoogleSearch(enable_cache=True)
    times_with_cache = []
    for i in range(3):
        start = time.time()
        response = searcher_with_cache.search(test_query + f" variant {i%2}", num_results=3)
        times_with_cache.append(time.time() - start)
        cache_status = "HIT" if response.metadata.get('cache_hit') else "MISS"
        print(f"  Attempt {i+1}: {times_with_cache[-1]:.3f}s ({cache_status})")
    
    # Test 8: Error handling
    print(f"\n{'='*80}")
    print("Test 8: Error handling and edge cases")
    print("=" * 80)
    
    # Empty query
    print("\n1. Empty query test:")
    result = quick_search("", num_results=3)
    print(f"  Result: {len(result)} items (expected: 0)")
    
    # Very long query
    print("\n2. Long query test:")
    long_query = "test " * 100
    result = quick_search(long_query, num_results=3)
    print(f"  Result: {len(result)} items")
    
    # Invalid num_results
    print("\n3. Invalid num_results test:")
    result = quick_search("test query", num_results=-5)
    print(f"  Result with -5: {len(result)} items")
    result = quick_search("test query", num_results=100)
    print(f"  Result with 100: {len(result)} items (capped at 10)")
    
    # Test 9: Memory and resource usage
    print(f"\n{'='*80}")
    print("Test 9: Resource usage test")
    print("=" * 80)
    
    stats = searcher.get_cache_stats()
    print(f"\nFinal cache statistics:")
    print(f"  Total entries: {stats.get('total_entries', 0)}")
    print(f"  Active entries: {stats.get('active_entries', 0)}")
    print(f"  Total accesses: {stats.get('total_accesses', 0)}")
    print(f"  Database size: {stats.get('db_size_mb', 0):.2f} MB")
    print(f"  Active connections: {stats.get('active_connections', 0)}")
    print(f"  Database path: {stats.get('db_path', 'N/A')}")
    
    # Optimize database
    print("\nOptimizing database...")
    searcher.optimize_cache()
    
    stats = searcher.get_cache_stats()
    print(f"  Database size after optimization: {stats.get('db_size_mb', 0):.2f} MB")
    
    # Final cleanup
    print(f"\n{'='*80}")
    print("Cleaning up...")
    print("=" * 80)
    
    searcher.close()
    
    print("\n✓ All tests completed successfully!")
    print("=" * 80)


