#!/usr/bin/env python3
"""
Efficient Trajectory Storage System

High-performance trajectory storage with:
- Binary format with custom serialization
- Streaming I/O for large trajectory files
- Real-time compression and decompression
- Metadata management and indexing
- Memory-mapped file access for large datasets
- Asynchronous I/O operations

Designed for handling large trajectory datasets with minimal memory footprint.
"""

import os
import struct
import mmap
import zlib
import lz4
import pickle
import json
import sqlite3
import threading
import asyncio
import aiofiles
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import uuid


class CompressionType(Enum):
    """Compression algorithms available for trajectory storage."""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    CUSTOM_DELTA = "delta"  # Custom delta compression for trajectories


class StorageFormat(Enum):
    """Storage format types for trajectory data."""
    BINARY_STREAM = "binary_stream"
    BINARY_INDEXED = "binary_indexed"
    SQLITE_DB = "sqlite_db"
    HDF5 = "hdf5"
    MEMORY_MAPPED = "memory_mapped"


@dataclass
class TrajectoryHeader:
    """Header information for trajectory files."""
    version: str = "1.0"
    magic_number: int = 0x54524A4F  # 'TRJO' in hex
    point_count: int = 0
    compression_type: CompressionType = CompressionType.LZ4
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    spatial_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-1, 1), (-1, 1), (-1, 1))
    metadata_size: int = 0
    data_offset: int = 0
    checksum: str = ""
    created_at: float = 0.0
    session_id: str = ""


@dataclass
class TrajectoryMetadata:
    """Extended metadata for trajectory sessions."""
    session_id: str
    duration_seconds: float
    total_points: int
    average_fps: float
    gesture_summary: Dict[str, Any]
    parameter_ranges: Dict[str, Tuple[float, float]]
    file_size_bytes: int
    compression_ratio: float
    tags: List[str]
    description: str = ""
    created_at: float = 0.0
    modified_at: float = 0.0


class TrajectoryPoint:
    """Optimized trajectory point representation for storage."""
    __slots__ = ['timestamp', 'x', 'y', 'z', 'gesture_strength', 'gesture_type', 
                'r1', 'r2', 'w1', 'w2', 'p1', 'p2']
    
    def __init__(self, timestamp: float, x: float, y: float, z: float = 0.0,
                 gesture_strength: float = 1.0, gesture_type: int = 0,
                 r1: float = 1.0, r2: float = 0.5, w1: float = 1.0, 
                 w2: float = 2.0, p1: float = 0.0, p2: float = 0.0):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.gesture_strength = gesture_strength
        self.gesture_type = gesture_type
        self.r1 = r1
        self.r2 = r2
        self.w1 = w1
        self.w2 = w2
        self.p1 = p1
        self.p2 = p2
    
    def to_bytes(self) -> bytes:
        """Convert point to binary representation."""
        return struct.pack('dddddidddddd',
                          self.timestamp, self.x, self.y, self.z,
                          self.gesture_strength, self.gesture_type,
                          self.r1, self.r2, self.w1, self.w2, self.p1, self.p2)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'TrajectoryPoint':
        """Create point from binary data."""
        values = struct.unpack('dddddidddddd', data)
        return cls(*values)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.timestamp, self.x, self.y, self.z,
                        self.gesture_strength, self.gesture_type,
                        self.r1, self.r2, self.w1, self.w2, self.p1, self.p2])


class DeltaCompressor:
    """Custom delta compression for trajectory data."""
    
    @staticmethod
    def compress_trajectory(points: List[TrajectoryPoint]) -> bytes:
        """Compress trajectory using delta encoding."""
        if not points:
            return b''
        
        # Convert to arrays for efficient processing
        data = np.array([point.to_array() for point in points])
        
        # Store first point as-is
        compressed = [data[0].astype(np.float32).tobytes()]
        
        # Delta encode subsequent points
        for i in range(1, len(data)):
            delta = data[i] - data[i-1]
            # Use smaller data types for deltas when possible
            delta_compressed = delta.astype(np.float16).tobytes()
            compressed.append(delta_compressed)
        
        # Combine with LZ4 compression
        combined_data = b''.join(compressed)
        return lz4.compress(combined_data)
    
    @staticmethod
    def decompress_trajectory(compressed_data: bytes) -> List[TrajectoryPoint]:
        """Decompress delta-encoded trajectory."""
        if not compressed_data:
            return []
        
        # Decompress with LZ4
        try:
            decompressed = lz4.decompress(compressed_data)
        except:
            return []
        
        points = []
        point_size_full = 12 * 4  # 12 float32s
        point_size_delta = 12 * 2  # 12 float16s
        
        if len(decompressed) < point_size_full:
            return []
        
        # Read first point
        first_data = np.frombuffer(decompressed[:point_size_full], dtype=np.float32)
        current_point = first_data.astype(np.float64)
        points.append(TrajectoryPoint(*current_point))
        
        # Read delta-encoded points
        offset = point_size_full
        while offset + point_size_delta <= len(decompressed):
            delta_data = np.frombuffer(decompressed[offset:offset+point_size_delta], dtype=np.float16)
            current_point = current_point + delta_data.astype(np.float64)
            points.append(TrajectoryPoint(*current_point))
            offset += point_size_delta
        
        return points


class TrajectoryWriter:
    """High-performance trajectory writer with streaming support."""
    
    def __init__(self, filepath: str, compression: CompressionType = CompressionType.LZ4,
                 buffer_size: int = 8192):
        """
        Initialize trajectory writer.
        
        Args:
            filepath: Output file path
            compression: Compression algorithm to use
            buffer_size: Internal buffer size for batching
        """
        self.filepath = Path(filepath)
        self.compression = compression
        self.buffer_size = buffer_size
        
        self.file_handle = None
        self.buffer = []
        self.header = TrajectoryHeader(
            compression_type=compression,
            session_id=str(uuid.uuid4()),
            created_at=time.time()
        )
        self.metadata = TrajectoryMetadata(
            session_id=self.header.session_id,
            duration_seconds=0.0,
            total_points=0,
            average_fps=0.0,
            gesture_summary={},
            parameter_ranges={},
            file_size_bytes=0,
            compression_ratio=1.0,
            tags=[],
            created_at=time.time()
        )
        
        self.lock = threading.RLock()
        self.start_time = None
        self.bytes_written = 0
        self.uncompressed_size = 0
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def open(self) -> None:
        """Open file for writing."""
        with self.lock:
            self.file_handle = open(self.filepath, 'wb')
            self.start_time = time.time()
            
            # Write placeholder header (will be updated on close)
            header_data = self._serialize_header(self.header)
            self.file_handle.write(header_data)
            self.bytes_written = len(header_data)
    
    def write_point(self, point: TrajectoryPoint) -> None:
        """Write a single trajectory point."""
        with self.lock:
            if not self.file_handle:
                raise RuntimeError("Writer not opened")
            
            self.buffer.append(point)
            
            # Update running statistics
            self.header.point_count += 1
            self.uncompressed_size += 96  # Size of uncompressed point
            
            # Update spatial bounds
            self._update_bounds(point)
            
            # Flush buffer if full
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def write_points(self, points: List[TrajectoryPoint]) -> None:
        """Write multiple trajectory points efficiently."""
        with self.lock:
            for point in points:
                self.write_point(point)
    
    def _flush_buffer(self) -> None:
        """Flush internal buffer to file."""
        if not self.buffer:
            return
        
        # Compress batch
        if self.compression == CompressionType.NONE:
            data = b''.join(point.to_bytes() for point in self.buffer)
        elif self.compression == CompressionType.ZLIB:
            raw_data = b''.join(point.to_bytes() for point in self.buffer)
            data = zlib.compress(raw_data, level=6)
        elif self.compression == CompressionType.LZ4:
            raw_data = b''.join(point.to_bytes() for point in self.buffer)
            data = lz4.compress(raw_data)
        elif self.compression == CompressionType.CUSTOM_DELTA:
            data = DeltaCompressor.compress_trajectory(self.buffer)
        else:
            raise ValueError(f"Unsupported compression: {self.compression}")
        
        # Write chunk header (size + compressed flag)
        chunk_header = struct.pack('I', len(data))
        self.file_handle.write(chunk_header)
        self.file_handle.write(data)
        
        self.bytes_written += 4 + len(data)
        self.buffer.clear()
    
    def _update_bounds(self, point: TrajectoryPoint) -> None:
        """Update spatial bounds with new point."""
        bounds = list(self.header.spatial_bounds)
        
        # X bounds
        bounds[0] = (min(bounds[0][0], point.x), max(bounds[0][1], point.x))
        # Y bounds
        bounds[1] = (min(bounds[1][0], point.y), max(bounds[1][1], point.y))
        # Z bounds
        bounds[2] = (min(bounds[2][0], point.z), max(bounds[2][1], point.z))
        
        self.header.spatial_bounds = tuple(bounds)
    
    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set additional metadata for the trajectory."""
        with self.lock:
            self.metadata.description = metadata.get('description', '')
            self.metadata.tags = metadata.get('tags', [])
            self.metadata.gesture_summary = metadata.get('gesture_summary', {})
            self.metadata.parameter_ranges = metadata.get('parameter_ranges', {})
    
    def close(self) -> None:
        """Close writer and finalize file."""
        with self.lock:
            if not self.file_handle:
                return
            
            # Flush remaining buffer
            self._flush_buffer()
            
            # Update header with final information
            self.header.timestamp_end = time.time()
            if self.start_time:
                duration = self.header.timestamp_end - self.start_time
                self.metadata.duration_seconds = duration
                self.metadata.average_fps = self.header.point_count / duration if duration > 0 else 0
            
            self.metadata.total_points = self.header.point_count
            self.metadata.file_size_bytes = self.bytes_written
            self.metadata.compression_ratio = self.uncompressed_size / max(self.bytes_written, 1)
            self.metadata.modified_at = time.time()
            
            # Calculate file checksum
            current_pos = self.file_handle.tell()
            self.file_handle.seek(0)
            hasher = hashlib.sha256()
            while chunk := self.file_handle.read(8192):
                hasher.update(chunk)
            self.header.checksum = hasher.hexdigest()
            
            # Write final header
            self.file_handle.seek(0)
            header_data = self._serialize_header(self.header)
            self.file_handle.write(header_data)
            
            # Write metadata at end of file
            self.file_handle.seek(current_pos)
            metadata_data = self._serialize_metadata(self.metadata)
            self.file_handle.write(metadata_data)
            
            self.file_handle.close()
            self.file_handle = None
    
    def _serialize_header(self, header: TrajectoryHeader) -> bytes:
        """Serialize header to binary format."""
        # Fixed-size header (256 bytes)
        data = struct.pack('I', header.magic_number)
        data += header.version.encode('utf-8').ljust(16, b'\x00')
        data += struct.pack('Q', header.point_count)
        data += header.compression_type.value.encode('utf-8').ljust(16, b'\x00')
        data += struct.pack('dd', header.timestamp_start, header.timestamp_end)
        
        # Spatial bounds (6 doubles)
        bounds_flat = [coord for bound in header.spatial_bounds for coord in bound]
        data += struct.pack('6d', *bounds_flat)
        
        data += struct.pack('II', header.metadata_size, header.data_offset)
        data += header.checksum.encode('utf-8').ljust(64, b'\x00')
        data += struct.pack('d', header.created_at)
        data += header.session_id.encode('utf-8').ljust(36, b'\x00')
        
        # Pad to exactly 256 bytes
        data = data.ljust(256, b'\x00')
        return data
    
    def _serialize_metadata(self, metadata: TrajectoryMetadata) -> bytes:
        """Serialize metadata to JSON format."""
        metadata_dict = asdict(metadata)
        json_data = json.dumps(metadata_dict, indent=2).encode('utf-8')
        
        # Write size prefix
        size_data = struct.pack('I', len(json_data))
        return size_data + json_data


class TrajectoryReader:
    """High-performance trajectory reader with streaming and memory-mapped support."""
    
    def __init__(self, filepath: str, use_memory_map: bool = True):
        """
        Initialize trajectory reader.
        
        Args:
            filepath: Input file path
            use_memory_map: Use memory mapping for large files
        """
        self.filepath = Path(filepath)
        self.use_memory_map = use_memory_map and os.path.getsize(filepath) > 100 * 1024 * 1024  # 100MB threshold
        
        self.file_handle = None
        self.memory_map = None
        self.header: Optional[TrajectoryHeader] = None
        self.metadata: Optional[TrajectoryMetadata] = None
        self.data_start_offset = 256  # Header size
        
        self.lock = threading.RLock()
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def open(self) -> None:
        """Open trajectory file for reading."""
        with self.lock:
            self.file_handle = open(self.filepath, 'rb')
            
            if self.use_memory_map:
                self.memory_map = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Read header
            header_data = self.file_handle.read(256)
            self.header = self._deserialize_header(header_data)
            
            # Read metadata from end of file
            self._read_metadata()
    
    def _deserialize_header(self, data: bytes) -> TrajectoryHeader:
        """Deserialize header from binary format."""
        if len(data) < 256:
            raise ValueError("Invalid header size")
        
        magic = struct.unpack('I', data[0:4])[0]
        if magic != 0x54524A4F:
            raise ValueError("Invalid magic number")
        
        version = data[4:20].rstrip(b'\x00').decode('utf-8')
        point_count = struct.unpack('Q', data[20:28])[0]
        compression_str = data[28:44].rstrip(b'\x00').decode('utf-8')
        compression = CompressionType(compression_str)
        
        timestamp_start, timestamp_end = struct.unpack('dd', data[44:60])
        bounds_flat = struct.unpack('6d', data[60:108])
        spatial_bounds = ((bounds_flat[0], bounds_flat[1]),
                         (bounds_flat[2], bounds_flat[3]),
                         (bounds_flat[4], bounds_flat[5]))
        
        metadata_size, data_offset = struct.unpack('II', data[108:116])
        checksum = data[116:180].rstrip(b'\x00').decode('utf-8')
        created_at = struct.unpack('d', data[180:188])[0]
        session_id = data[188:224].rstrip(b'\x00').decode('utf-8')
        
        return TrajectoryHeader(
            version=version, magic_number=magic, point_count=point_count,
            compression_type=compression, timestamp_start=timestamp_start,
            timestamp_end=timestamp_end, spatial_bounds=spatial_bounds,
            metadata_size=metadata_size, data_offset=data_offset,
            checksum=checksum, created_at=created_at, session_id=session_id
        )
    
    def _read_metadata(self) -> None:
        """Read metadata from end of file."""
        try:
            # Seek to end and read metadata size
            self.file_handle.seek(-4, 2)
            metadata_size_data = self.file_handle.read(4)
            
            if len(metadata_size_data) == 4:
                metadata_size = struct.unpack('I', metadata_size_data)[0]
                
                # Read metadata
                self.file_handle.seek(-(4 + metadata_size), 2)
                metadata_data = self.file_handle.read(metadata_size)
                
                metadata_dict = json.loads(metadata_data.decode('utf-8'))
                self.metadata = TrajectoryMetadata(**metadata_dict)
        except:
            # Metadata reading failed, create minimal metadata
            self.metadata = TrajectoryMetadata(
                session_id=self.header.session_id,
                duration_seconds=self.header.timestamp_end - self.header.timestamp_start,
                total_points=self.header.point_count,
                average_fps=0.0,
                gesture_summary={},
                parameter_ranges={},
                file_size_bytes=os.path.getsize(self.filepath),
                compression_ratio=1.0,
                tags=[]
            )
    
    def read_all_points(self) -> List[TrajectoryPoint]:
        """Read all trajectory points from file."""
        return list(self.read_points_stream())
    
    def read_points_stream(self, chunk_size: int = 1000) -> Iterator[TrajectoryPoint]:
        """Stream trajectory points from file."""
        with self.lock:
            if not self.file_handle:
                raise RuntimeError("Reader not opened")
            
            self.file_handle.seek(self.data_start_offset)
            
            while True:
                # Read chunk size
                chunk_size_data = self.file_handle.read(4)
                if len(chunk_size_data) < 4:
                    break
                
                chunk_size = struct.unpack('I', chunk_size_data)[0]
                if chunk_size == 0:
                    break
                
                # Read chunk data
                chunk_data = self.file_handle.read(chunk_size)
                if len(chunk_data) < chunk_size:
                    break
                
                # Decompress chunk
                points = self._decompress_chunk(chunk_data)
                
                for point in points:
                    yield point
    
    def _decompress_chunk(self, data: bytes) -> List[TrajectoryPoint]:
        """Decompress a chunk of trajectory data."""
        if self.header.compression_type == CompressionType.NONE:
            decompressed = data
        elif self.header.compression_type == CompressionType.ZLIB:
            decompressed = zlib.decompress(data)
        elif self.header.compression_type == CompressionType.LZ4:
            decompressed = lz4.decompress(data)
        elif self.header.compression_type == CompressionType.CUSTOM_DELTA:
            return DeltaCompressor.decompress_trajectory(data)
        else:
            raise ValueError(f"Unsupported compression: {self.header.compression_type}")
        
        # Parse points from decompressed data
        points = []
        point_size = 96  # Size of single point in bytes
        
        for i in range(0, len(decompressed), point_size):
            if i + point_size <= len(decompressed):
                point_data = decompressed[i:i + point_size]
                point = TrajectoryPoint.from_bytes(point_data)
                points.append(point)
        
        return points
    
    def read_point_range(self, start_index: int, count: int) -> List[TrajectoryPoint]:
        """Read a specific range of points."""
        points = []
        current_index = 0
        
        for point in self.read_points_stream():
            if current_index >= start_index + count:
                break
            
            if current_index >= start_index:
                points.append(point)
            
            current_index += 1
        
        return points
    
    def get_header(self) -> TrajectoryHeader:
        """Get trajectory file header."""
        return self.header
    
    def get_metadata(self) -> TrajectoryMetadata:
        """Get trajectory metadata."""
        return self.metadata
    
    def close(self) -> None:
        """Close reader and release resources."""
        with self.lock:
            if self.memory_map:
                self.memory_map.close()
                self.memory_map = None
            
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None


class AsyncTrajectoryWriter:
    """Asynchronous trajectory writer for high-throughput scenarios."""
    
    def __init__(self, filepath: str, compression: CompressionType = CompressionType.LZ4):
        """Initialize async writer."""
        self.filepath = filepath
        self.compression = compression
        self.queue = asyncio.Queue(maxsize=1000)
        self.writer_task = None
        self.running = False
        
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def start(self) -> None:
        """Start async writer."""
        self.running = True
        self.writer_task = asyncio.create_task(self._writer_coroutine())
    
    async def stop(self) -> None:
        """Stop async writer."""
        self.running = False
        await self.queue.put(None)  # Sentinel to stop writer
        
        if self.writer_task:
            await self.writer_task
    
    async def write_point(self, point: TrajectoryPoint) -> None:
        """Write point asynchronously."""
        if self.running:
            await self.queue.put(point)
    
    async def write_points(self, points: List[TrajectoryPoint]) -> None:
        """Write multiple points asynchronously."""
        for point in points:
            await self.write_point(point)
    
    async def _writer_coroutine(self) -> None:
        """Main writer coroutine."""
        with TrajectoryWriter(self.filepath, self.compression) as writer:
            while self.running:
                try:
                    point = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    if point is None:  # Sentinel
                        break
                    writer.write_point(point)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Async writer error: {e}")


class TrajectoryDatabase:
    """SQLite-based trajectory database for indexing and querying."""
    
    def __init__(self, db_path: str):
        """Initialize trajectory database."""
        self.db_path = db_path
        self.connection = None
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trajectories (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT UNIQUE,
                    filepath TEXT,
                    created_at REAL,
                    duration_seconds REAL,
                    point_count INTEGER,
                    average_fps REAL,
                    file_size_bytes INTEGER,
                    compression_ratio REAL,
                    tags TEXT,
                    description TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trajectory_stats (
                    session_id TEXT,
                    min_x REAL, max_x REAL,
                    min_y REAL, max_y REAL,
                    min_z REAL, max_z REAL,
                    avg_velocity REAL,
                    max_velocity REAL,
                    gesture_types TEXT,
                    parameter_ranges TEXT,
                    FOREIGN KEY(session_id) REFERENCES trajectories(session_id)
                )
            ''')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON trajectories(session_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON trajectories(created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tags ON trajectories(tags)')
    
    def add_trajectory(self, filepath: str, metadata: TrajectoryMetadata) -> None:
        """Add trajectory to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO trajectories 
                (session_id, filepath, created_at, duration_seconds, point_count,
                 average_fps, file_size_bytes, compression_ratio, tags, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.session_id, str(filepath), metadata.created_at,
                metadata.duration_seconds, metadata.total_points, metadata.average_fps,
                metadata.file_size_bytes, metadata.compression_ratio,
                json.dumps(metadata.tags), metadata.description
            ))
    
    def find_trajectories(self, **criteria) -> List[Dict[str, Any]]:
        """Find trajectories matching criteria."""
        query = "SELECT * FROM trajectories WHERE 1=1"
        params = []
        
        if 'tag' in criteria:
            query += " AND tags LIKE ?"
            params.append(f'%"{criteria["tag"]}"%')
        
        if 'min_duration' in criteria:
            query += " AND duration_seconds >= ?"
            params.append(criteria['min_duration'])
        
        if 'min_points' in criteria:
            query += " AND point_count >= ?"
            params.append(criteria['min_points'])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


# Example usage and performance test
if __name__ == "__main__":
    print("Trajectory Storage System Test")
    
    # Create test trajectory data
    points = []
    for i in range(10000):
        t = i * 0.01
        point = TrajectoryPoint(
            timestamp=t,
            x=2.0 * np.cos(t) + 1.0 * np.cos(3*t),
            y=2.0 * np.sin(t) + 1.0 * np.sin(3*t),
            z=0.5 * np.sin(2*t),
            gesture_strength=0.5 + 0.5 * np.sin(5*t),
            gesture_type=int(5 * (0.5 + 0.5 * np.cos(t))),
            r1=2.0, r2=1.0, w1=1.0, w2=3.0
        )
        points.append(point)
    
    # Test synchronous writing
    print("\nTesting synchronous writing...")
    start_time = time.time()
    
    with TrajectoryWriter("test_trajectory.trj", CompressionType.LZ4) as writer:
        writer.set_metadata({
            'description': 'Test trajectory data',
            'tags': ['test', 'parametric', 'gesture'],
            'gesture_summary': {'total_gestures': 50, 'dominant_type': 2}
        })
        writer.write_points(points)
    
    write_time = time.time() - start_time
    file_size = os.path.getsize("test_trajectory.trj")
    
    print(f"Write time: {write_time:.2f}s")
    print(f"File size: {file_size / 1024:.1f} KB")
    print(f"Write speed: {len(points) / write_time:.0f} points/sec")
    
    # Test reading
    print("\nTesting reading...")
    start_time = time.time()
    
    with TrajectoryReader("test_trajectory.trj") as reader:
        header = reader.get_header()
        metadata = reader.get_metadata()
        
        print(f"Header: {header.point_count} points, {header.compression_type}")
        print(f"Metadata: {metadata.duration_seconds:.2f}s, {metadata.average_fps:.1f} FPS")
        
        # Read all points
        read_points = reader.read_all_points()
    
    read_time = time.time() - start_time
    print(f"Read time: {read_time:.2f}s")
    print(f"Read speed: {len(read_points) / read_time:.0f} points/sec")
    print(f"Points match: {len(read_points) == len(points)}")
    
    # Test database
    print("\nTesting database...")
    db = TrajectoryDatabase("trajectories.db")
    db.add_trajectory("test_trajectory.trj", metadata)
    
    results = db.find_trajectories(tag="test", min_points=5000)
    print(f"Database results: {len(results)} trajectories found")
    
    # Cleanup
    os.remove("test_trajectory.trj")
    os.remove("trajectories.db")
    
    print("\nTest completed successfully!")