#!/usr/bin/env python3
"""
Trajectory Analysis and Export Tools

Comprehensive analysis and export utilities for trajectory data:
- Statistical analysis (velocity, acceleration, curvature)
- Pattern recognition and gesture classification
- Trajectory comparison and similarity metrics
- Data export in multiple formats (JSON, CSV, KML, SVG)
- Interactive visualization and playback
- Batch processing capabilities

Designed for post-processing and analysis of recorded trajectory sessions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy import signal, interpolate, spatial, stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import time
import warnings
from collections import Counter

from trajectory_storage import TrajectoryReader, TrajectoryPoint, TrajectoryMetadata


@dataclass
class TrajectoryStatistics:
    """Statistical analysis results for trajectory data."""
    total_points: int
    duration_seconds: float
    total_distance: float
    average_velocity: float
    max_velocity: float
    average_acceleration: float
    max_acceleration: float
    average_curvature: float
    max_curvature: float
    bounding_box: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]  # (x_min, x_max), (y_min, y_max), (z_min, z_max)
    velocity_distribution: Dict[str, float]  # percentiles
    direction_changes: int
    loops_detected: int
    gesture_type_distribution: Dict[int, int]
    gesture_strength_stats: Dict[str, float]


@dataclass
class PatternAnalysisResult:
    """Results from pattern analysis."""
    pattern_type: str
    confidence: float
    parameters: Dict[str, float]
    frequency_components: List[Tuple[float, float]]  # (frequency, amplitude)
    repeating_segments: List[Tuple[int, int]]  # (start_index, end_index)
    anomalies: List[int]  # indices of anomalous points


@dataclass
class TrajectoryComparison:
    """Results from comparing two trajectories."""
    similarity_score: float  # 0-1, higher is more similar
    dtw_distance: float  # Dynamic Time Warping distance
    correlation_coefficient: float
    shape_similarity: float
    velocity_similarity: float
    gesture_similarity: float
    aligned_trajectories: Tuple[np.ndarray, np.ndarray]
    time_alignment: np.ndarray


class TrajectoryAnalyzer:
    """Advanced trajectory analysis and pattern recognition."""
    
    def __init__(self):
        """Initialize trajectory analyzer."""
        self.smoothing_window = 5
        self.velocity_threshold = 0.1
        self.curvature_threshold = 0.5
        
    def analyze_trajectory(self, points: List[TrajectoryPoint]) -> TrajectoryStatistics:
        """
        Perform comprehensive statistical analysis of trajectory.
        
        Args:
            points: List of trajectory points
            
        Returns:
            Statistical analysis results
        """
        if not points:
            return TrajectoryStatistics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      ((0, 0), (0, 0), (0, 0)), {}, 0, 0, {}, {})
        
        # Extract data arrays
        positions = np.array([[p.x, p.y, p.z] for p in points])
        timestamps = np.array([p.timestamp for p in points])
        gesture_types = np.array([p.gesture_type for p in points])
        gesture_strengths = np.array([p.gesture_strength for p in points])
        
        # Basic statistics
        total_points = len(points)
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        
        # Distance and velocity calculations
        distances = self._calculate_distances(positions)
        total_distance = np.sum(distances)
        velocities = self._calculate_velocities(positions, timestamps)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # Acceleration calculations
        accelerations = self._calculate_accelerations(velocities, timestamps)
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # Curvature calculations
        curvatures = self._calculate_curvature(positions)
        
        # Bounding box
        bounding_box = (
            (positions[:, 0].min(), positions[:, 0].max()),
            (positions[:, 1].min(), positions[:, 1].max()),
            (positions[:, 2].min(), positions[:, 2].max())
        )
        
        # Velocity distribution
        velocity_distribution = {
            'mean': np.mean(velocity_magnitudes),
            'std': np.std(velocity_magnitudes),
            'p10': np.percentile(velocity_magnitudes, 10),
            'p25': np.percentile(velocity_magnitudes, 25),
            'p50': np.percentile(velocity_magnitudes, 50),
            'p75': np.percentile(velocity_magnitudes, 75),
            'p90': np.percentile(velocity_magnitudes, 90)
        }
        
        # Direction changes and loops
        direction_changes = self._count_direction_changes(velocities)
        loops_detected = self._detect_loops(positions)
        
        # Gesture statistics
        gesture_type_dist = Counter(gesture_types)
        gesture_strength_stats = {
            'mean': np.mean(gesture_strengths),
            'std': np.std(gesture_strengths),
            'min': np.min(gesture_strengths),
            'max': np.max(gesture_strengths)
        }
        
        return TrajectoryStatistics(
            total_points=total_points,
            duration_seconds=duration,
            total_distance=total_distance,
            average_velocity=np.mean(velocity_magnitudes) if len(velocity_magnitudes) > 0 else 0.0,
            max_velocity=np.max(velocity_magnitudes) if len(velocity_magnitudes) > 0 else 0.0,
            average_acceleration=np.mean(acceleration_magnitudes) if len(acceleration_magnitudes) > 0 else 0.0,
            max_acceleration=np.max(acceleration_magnitudes) if len(acceleration_magnitudes) > 0 else 0.0,
            average_curvature=np.mean(curvatures) if len(curvatures) > 0 else 0.0,
            max_curvature=np.max(curvatures) if len(curvatures) > 0 else 0.0,
            bounding_box=bounding_box,
            velocity_distribution=velocity_distribution,
            direction_changes=direction_changes,
            loops_detected=loops_detected,
            gesture_type_distribution=dict(gesture_type_dist),
            gesture_strength_stats=gesture_strength_stats
        )
    
    def detect_patterns(self, points: List[TrajectoryPoint]) -> List[PatternAnalysisResult]:
        """
        Detect patterns in trajectory data.
        
        Args:
            points: List of trajectory points
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        if len(points) < 10:
            return patterns
        
        positions = np.array([[p.x, p.y, p.z] for p in points])
        timestamps = np.array([p.timestamp for p in points])
        
        # Detect circular patterns
        circular_pattern = self._detect_circular_pattern(positions)
        if circular_pattern:
            patterns.append(circular_pattern)
        
        # Detect periodic patterns
        periodic_patterns = self._detect_periodic_patterns(positions, timestamps)
        patterns.extend(periodic_patterns)
        
        # Detect spiral patterns
        spiral_pattern = self._detect_spiral_pattern(positions)
        if spiral_pattern:
            patterns.append(spiral_pattern)
        
        # Detect linear patterns
        linear_pattern = self._detect_linear_pattern(positions)
        if linear_pattern:
            patterns.append(linear_pattern)
        
        return patterns
    
    def compare_trajectories(self, points1: List[TrajectoryPoint], 
                           points2: List[TrajectoryPoint]) -> TrajectoryComparison:
        """
        Compare two trajectories for similarity.
        
        Args:
            points1, points2: Trajectory points to compare
            
        Returns:
            Comparison results
        """
        if not points1 or not points2:
            return TrajectoryComparison(0.0, float('inf'), 0.0, 0.0, 0.0, 0.0, 
                                      (np.array([]), np.array([])), np.array([]))
        
        # Extract positions
        pos1 = np.array([[p.x, p.y, p.z] for p in points1])
        pos2 = np.array([[p.x, p.y, p.z] for p in points2])
        
        # Calculate various similarity metrics
        shape_similarity = self._calculate_shape_similarity(pos1, pos2)
        velocity_similarity = self._calculate_velocity_similarity(points1, points2)
        gesture_similarity = self._calculate_gesture_similarity(points1, points2)
        
        # Dynamic Time Warping
        dtw_distance, aligned_pos1, aligned_pos2, time_alignment = self._dynamic_time_warping(pos1, pos2)
        
        # Correlation coefficient
        if len(aligned_pos1) > 1 and len(aligned_pos2) > 1:
            correlation = np.corrcoef(aligned_pos1.flatten(), aligned_pos2.flatten())[0, 1]
            correlation = 0.0 if np.isnan(correlation) else correlation
        else:
            correlation = 0.0
        
        # Overall similarity score (weighted average)
        similarity_score = (
            0.4 * shape_similarity +
            0.3 * velocity_similarity +
            0.2 * abs(correlation) +
            0.1 * gesture_similarity
        )
        
        return TrajectoryComparison(
            similarity_score=similarity_score,
            dtw_distance=dtw_distance,
            correlation_coefficient=correlation,
            shape_similarity=shape_similarity,
            velocity_similarity=velocity_similarity,
            gesture_similarity=gesture_similarity,
            aligned_trajectories=(aligned_pos1, aligned_pos2),
            time_alignment=time_alignment
        )
    
    def _calculate_distances(self, positions: np.ndarray) -> np.ndarray:
        """Calculate distances between consecutive points."""
        if len(positions) < 2:
            return np.array([])
        return np.linalg.norm(np.diff(positions, axis=0), axis=1)
    
    def _calculate_velocities(self, positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Calculate velocity vectors."""
        if len(positions) < 2:
            return np.zeros_like(positions)
        
        dt = np.diff(timestamps)
        dt = np.where(dt > 0, dt, 1e-6)  # Avoid division by zero
        
        dp = np.diff(positions, axis=0)
        velocities = dp / dt.reshape(-1, 1)
        
        # Extend to match position array length
        velocities = np.vstack([velocities[0:1], velocities])
        
        return velocities
    
    def _calculate_accelerations(self, velocities: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Calculate acceleration vectors."""
        if len(velocities) < 2:
            return np.zeros_like(velocities)
        
        dt = np.diff(timestamps)
        dt = np.where(dt > 0, dt, 1e-6)
        
        dv = np.diff(velocities, axis=0)
        accelerations = dv / dt.reshape(-1, 1)
        
        # Extend to match velocity array length
        accelerations = np.vstack([accelerations[0:1], accelerations])
        
        return accelerations
    
    def _calculate_curvature(self, positions: np.ndarray) -> np.ndarray:
        """Calculate curvature at each point."""
        if len(positions) < 3:
            return np.zeros(len(positions))
        
        # Use finite differences to compute first and second derivatives
        first_deriv = np.gradient(positions, axis=0)
        second_deriv = np.gradient(first_deriv, axis=0)
        
        # Calculate curvature using cross product formula
        cross_product = np.cross(first_deriv[:, :2], second_deriv[:, :2])  # 2D cross product
        first_deriv_mag = np.linalg.norm(first_deriv, axis=1)
        
        curvature = np.abs(cross_product) / np.maximum(first_deriv_mag**3, 1e-6)
        
        return curvature
    
    def _count_direction_changes(self, velocities: np.ndarray) -> int:
        """Count significant direction changes in trajectory."""
        if len(velocities) < 3:
            return 0
        
        # Calculate angle between consecutive velocity vectors
        angles = []
        for i in range(1, len(velocities)):
            v1 = velocities[i-1]
            v2 = velocities[i]
            
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)
            
            if mag1 > 1e-6 and mag2 > 1e-6:
                cos_angle = np.dot(v1, v2) / (mag1 * mag2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        # Count significant direction changes (> 45 degrees)
        direction_changes = np.sum(np.array(angles) > np.pi/4)
        
        return int(direction_changes)
    
    def _detect_loops(self, positions: np.ndarray, threshold: float = 0.1) -> int:
        """Detect loops in trajectory."""
        if len(positions) < 10:
            return 0
        
        loops = 0
        
        # Use a simple approach: check if trajectory returns close to previous positions
        for i in range(5, len(positions) - 5):
            current_pos = positions[i]
            
            # Check against previous positions (not too close in time)
            for j in range(max(0, i - 50), i - 5):
                if np.linalg.norm(current_pos - positions[j]) < threshold:
                    loops += 1
                    break  # Count each loop only once
        
        return loops
    
    def _detect_circular_pattern(self, positions: np.ndarray) -> Optional[PatternAnalysisResult]:
        """Detect circular/elliptical patterns."""
        if len(positions) < 20:
            return None
        
        # Use 2D positions for circle detection
        pos_2d = positions[:, :2]
        
        # Fit circle using least squares
        try:
            # Center the data
            center = np.mean(pos_2d, axis=0)
            centered_pos = pos_2d - center
            
            # Calculate radial distances
            radii = np.linalg.norm(centered_pos, axis=1)
            mean_radius = np.mean(radii)
            radius_std = np.std(radii)
            
            # Check if points are roughly circular
            circularity = 1.0 - (radius_std / mean_radius) if mean_radius > 0 else 0.0
            
            if circularity > 0.7:  # Threshold for circular pattern
                return PatternAnalysisResult(
                    pattern_type="circular",
                    confidence=circularity,
                    parameters={
                        'center_x': center[0],
                        'center_y': center[1],
                        'radius': mean_radius,
                        'eccentricity': radius_std / mean_radius
                    },
                    frequency_components=[],
                    repeating_segments=[],
                    anomalies=[]
                )
        except:
            pass
        
        return None
    
    def _detect_periodic_patterns(self, positions: np.ndarray, timestamps: np.ndarray) -> List[PatternAnalysisResult]:
        """Detect periodic patterns using FFT."""
        patterns = []
        
        if len(positions) < 50:
            return patterns
        
        try:
            # Interpolate to uniform time grid
            t_uniform = np.linspace(timestamps[0], timestamps[-1], len(timestamps))
            pos_interp = np.zeros((len(t_uniform), 3))
            
            for dim in range(3):
                pos_interp[:, dim] = np.interp(t_uniform, timestamps, positions[:, dim])
            
            # Apply FFT to each dimension
            for dim in range(3):
                signal_data = pos_interp[:, dim]
                signal_data = signal_data - np.mean(signal_data)  # Remove DC component
                
                # Apply window to reduce spectral leakage
                windowed_signal = signal_data * np.hanning(len(signal_data))
                
                fft = np.fft.fft(windowed_signal)
                freqs = np.fft.fftfreq(len(fft), d=(t_uniform[1] - t_uniform[0]))
                
                # Find dominant frequencies
                magnitude = np.abs(fft)
                positive_freqs = freqs[:len(freqs)//2]
                positive_magnitude = magnitude[:len(magnitude)//2]
                
                # Find peaks
                peaks, _ = find_peaks(positive_magnitude, height=np.max(positive_magnitude) * 0.1)
                
                if len(peaks) > 0:
                    # Get dominant frequency components
                    peak_freqs = positive_freqs[peaks]
                    peak_mags = positive_magnitude[peaks]
                    
                    # Sort by magnitude
                    sorted_indices = np.argsort(peak_mags)[::-1]
                    
                    frequency_components = [(peak_freqs[i], peak_mags[i]) for i in sorted_indices[:5]]
                    
                    if len(frequency_components) > 0 and frequency_components[0][1] > np.std(positive_magnitude) * 3:
                        patterns.append(PatternAnalysisResult(
                            pattern_type=f"periodic_dim_{dim}",
                            confidence=min(1.0, frequency_components[0][1] / np.max(positive_magnitude)),
                            parameters={
                                'dominant_frequency': frequency_components[0][0],
                                'dimension': dim
                            },
                            frequency_components=frequency_components,
                            repeating_segments=[],
                            anomalies=[]
                        ))
        except:
            pass
        
        return patterns
    
    def _detect_spiral_pattern(self, positions: np.ndarray) -> Optional[PatternAnalysisResult]:
        """Detect spiral patterns."""
        if len(positions) < 30:
            return None
        
        try:
            # Convert to polar coordinates
            pos_2d = positions[:, :2]
            center = np.mean(pos_2d, axis=0)
            centered_pos = pos_2d - center
            
            angles = np.arctan2(centered_pos[:, 1], centered_pos[:, 0])
            radii = np.linalg.norm(centered_pos, axis=1)
            
            # Unwrap angles to detect spiral motion
            angles_unwrapped = np.unwrap(angles)
            total_rotation = abs(angles_unwrapped[-1] - angles_unwrapped[0])
            
            # Check for spiral: radius should correlate with angle
            if total_rotation > 2 * np.pi:  # At least one full rotation
                correlation = np.corrcoef(angles_unwrapped, radii)[0, 1]
                if abs(correlation) > 0.7:
                    spiral_type = "expanding" if correlation > 0 else "contracting"
                    
                    return PatternAnalysisResult(
                        pattern_type=f"spiral_{spiral_type}",
                        confidence=abs(correlation),
                        parameters={
                            'center_x': center[0],
                            'center_y': center[1],
                            'total_rotation': total_rotation,
                            'correlation': correlation
                        },
                        frequency_components=[],
                        repeating_segments=[],
                        anomalies=[]
                    )
        except:
            pass
        
        return None
    
    def _detect_linear_pattern(self, positions: np.ndarray) -> Optional[PatternAnalysisResult]:
        """Detect linear patterns."""
        if len(positions) < 10:
            return None
        
        try:
            # Fit line using PCA
            centered_pos = positions - np.mean(positions, axis=0)
            _, _, V = np.linalg.svd(centered_pos)
            
            # First principal component gives line direction
            line_direction = V[0]
            
            # Project points onto line
            projections = np.dot(centered_pos, line_direction)
            
            # Calculate how well points fit the line
            projected_points = np.outer(projections, line_direction)
            residuals = centered_pos - projected_points
            residual_variance = np.var(np.linalg.norm(residuals, axis=1))
            total_variance = np.var(np.linalg.norm(centered_pos, axis=1))
            
            linearity = 1.0 - (residual_variance / total_variance) if total_variance > 0 else 0.0
            
            if linearity > 0.8:  # Strong linear pattern
                return PatternAnalysisResult(
                    pattern_type="linear",
                    confidence=linearity,
                    parameters={
                        'direction_x': line_direction[0],
                        'direction_y': line_direction[1],
                        'direction_z': line_direction[2],
                        'linearity_score': linearity
                    },
                    frequency_components=[],
                    repeating_segments=[],
                    anomalies=[]
                )
        except:
            pass
        
        return None
    
    def _calculate_shape_similarity(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate shape similarity between trajectories."""
        if len(pos1) < 2 or len(pos2) < 2:
            return 0.0
        
        try:
            # Normalize trajectories (center and scale)
            pos1_norm = self._normalize_trajectory(pos1)
            pos2_norm = self._normalize_trajectory(pos2)
            
            # Resample to same length using interpolation
            min_len = min(len(pos1_norm), len(pos2_norm))
            target_len = max(10, min_len // 2)  # Reasonable resampling size
            
            pos1_resampled = self._resample_trajectory(pos1_norm, target_len)
            pos2_resampled = self._resample_trajectory(pos2_norm, target_len)
            
            # Calculate point-wise distances
            distances = np.linalg.norm(pos1_resampled - pos2_resampled, axis=1)
            mean_distance = np.mean(distances)
            
            # Convert to similarity score (0-1, higher is more similar)
            similarity = np.exp(-mean_distance)
            
            return similarity
        except:
            return 0.0
    
    def _calculate_velocity_similarity(self, points1: List[TrajectoryPoint], 
                                     points2: List[TrajectoryPoint]) -> float:
        """Calculate velocity pattern similarity."""
        try:
            pos1 = np.array([[p.x, p.y, p.z] for p in points1])
            pos2 = np.array([[p.x, p.y, p.z] for p in points2])
            times1 = np.array([p.timestamp for p in points1])
            times2 = np.array([p.timestamp for p in points2])
            
            vel1 = self._calculate_velocities(pos1, times1)
            vel2 = self._calculate_velocities(pos2, times2)
            
            vel_mag1 = np.linalg.norm(vel1, axis=1)
            vel_mag2 = np.linalg.norm(vel2, axis=1)
            
            # Compare velocity distributions
            if len(vel_mag1) > 0 and len(vel_mag2) > 0:
                # Use statistical comparison
                stat, p_value = stats.ks_2samp(vel_mag1, vel_mag2)
                similarity = np.exp(-stat)  # Convert KS statistic to similarity
                return similarity
            
        except:
            pass
        
        return 0.0
    
    def _calculate_gesture_similarity(self, points1: List[TrajectoryPoint], 
                                    points2: List[TrajectoryPoint]) -> float:
        """Calculate gesture pattern similarity."""
        try:
            # Compare gesture type distributions
            types1 = [p.gesture_type for p in points1]
            types2 = [p.gesture_type for p in points2]
            
            # Compare gesture strength distributions  
            strengths1 = [p.gesture_strength for p in points1]
            strengths2 = [p.gesture_strength for p in points2]
            
            # Type similarity (Jaccard index)
            set1, set2 = set(types1), set(types2)
            type_similarity = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0.0
            
            # Strength similarity (correlation)
            if len(strengths1) > 1 and len(strengths2) > 1:
                # Resample to same length for correlation
                min_len = min(len(strengths1), len(strengths2))
                s1_resampled = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(strengths1)), strengths1)
                s2_resampled = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(strengths2)), strengths2)
                
                strength_similarity = abs(np.corrcoef(s1_resampled, s2_resampled)[0, 1])
                strength_similarity = 0.0 if np.isnan(strength_similarity) else strength_similarity
            else:
                strength_similarity = 0.0
            
            return (type_similarity + strength_similarity) / 2.0
            
        except:
            pass
        
        return 0.0
    
    def _dynamic_time_warping(self, seq1: np.ndarray, seq2: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Compute Dynamic Time Warping distance and alignment."""
        try:
            n, m = len(seq1), len(seq2)
            
            # Compute distance matrix
            dist_matrix = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    dist_matrix[i, j] = np.linalg.norm(seq1[i] - seq2[j])
            
            # Dynamic programming matrix
            dtw_matrix = np.full((n + 1, m + 1), np.inf)
            dtw_matrix[0, 0] = 0
            
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = dist_matrix[i-1, j-1]
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],      # insertion
                        dtw_matrix[i, j-1],      # deletion
                        dtw_matrix[i-1, j-1]     # match
                    )
            
            # Backtrack to find alignment path
            path = []
            i, j = n, m
            while i > 0 and j > 0:
                path.append((i-1, j-1))
                
                # Choose direction with minimum cost
                if dtw_matrix[i-1, j-1] <= dtw_matrix[i-1, j] and dtw_matrix[i-1, j-1] <= dtw_matrix[i, j-1]:
                    i, j = i-1, j-1
                elif dtw_matrix[i-1, j] <= dtw_matrix[i, j-1]:
                    i = i-1
                else:
                    j = j-1
            
            path.reverse()
            
            # Extract aligned sequences
            aligned_seq1 = np.array([seq1[i] for i, j in path])
            aligned_seq2 = np.array([seq2[j] for i, j in path])
            time_alignment = np.array(path)
            
            return dtw_matrix[n, m], aligned_seq1, aligned_seq2, time_alignment
            
        except:
            return float('inf'), seq1[:min(len(seq1), len(seq2))], seq2[:min(len(seq1), len(seq2))], np.array([])
    
    def _normalize_trajectory(self, positions: np.ndarray) -> np.ndarray:
        """Normalize trajectory (center and scale)."""
        centered = positions - np.mean(positions, axis=0)
        scale = np.max(np.linalg.norm(centered, axis=1))
        return centered / scale if scale > 0 else centered
    
    def _resample_trajectory(self, positions: np.ndarray, target_length: int) -> np.ndarray:
        """Resample trajectory to target length using interpolation."""
        if len(positions) <= 1:
            return positions
        
        original_indices = np.arange(len(positions))
        target_indices = np.linspace(0, len(positions) - 1, target_length)
        
        resampled = np.zeros((target_length, positions.shape[1]))
        for dim in range(positions.shape[1]):
            resampled[:, dim] = np.interp(target_indices, original_indices, positions[:, dim])
        
        return resampled


class TrajectoryExporter:
    """Export trajectory data in multiple formats."""
    
    @staticmethod
    def export_to_json(points: List[TrajectoryPoint], filepath: str, 
                      metadata: Optional[TrajectoryMetadata] = None) -> bool:
        """Export trajectory to JSON format."""
        try:
            data = {
                'metadata': asdict(metadata) if metadata else {},
                'trajectory': [
                    {
                        'timestamp': p.timestamp,
                        'position': {'x': p.x, 'y': p.y, 'z': p.z},
                        'gesture': {
                            'strength': p.gesture_strength,
                            'type': p.gesture_type
                        },
                        'parameters': {
                            'r1': p.r1, 'r2': p.r2,
                            'w1': p.w1, 'w2': p.w2,
                            'p1': p.p1, 'p2': p.p2
                        }
                    }
                    for p in points
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"JSON export failed: {e}")
            return False
    
    @staticmethod
    def export_to_csv(points: List[TrajectoryPoint], filepath: str) -> bool:
        """Export trajectory to CSV format."""
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'timestamp', 'x', 'y', 'z', 'gesture_strength', 'gesture_type',
                    'r1', 'r2', 'w1', 'w2', 'p1', 'p2'
                ])
                
                # Write data
                for p in points:
                    writer.writerow([
                        p.timestamp, p.x, p.y, p.z, p.gesture_strength, p.gesture_type,
                        p.r1, p.r2, p.w1, p.w2, p.p1, p.p2
                    ])
            
            return True
        except Exception as e:
            print(f"CSV export failed: {e}")
            return False
    
    @staticmethod
    def export_to_kml(points: List[TrajectoryPoint], filepath: str, 
                     name: str = "Trajectory") -> bool:
        """Export trajectory to KML format for Google Earth."""
        try:
            kml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
    <Document>
        <name>{name}</name>
        <Style id="trajectoryStyle">
            <LineStyle>
                <color>ff0000ff</color>
                <width>2</width>
            </LineStyle>
            <PointStyle>
                <color>ff00ff00</color>
                <scale>0.5</scale>
            </PointStyle>
        </Style>
        
        <Placemark>
            <name>Trajectory Path</name>
            <styleUrl>#trajectoryStyle</styleUrl>
            <LineString>
                <coordinates>
'''
            
            # Add coordinates (longitude, latitude, altitude)
            for p in points:
                kml_content += f"                    {p.x},{p.y},{p.z}\n"
            
            kml_content += '''                </coordinates>
            </LineString>
        </Placemark>
        
        <!-- Start Point -->
        <Placemark>
            <name>Start</name>
            <Point>
                <coordinates>''' + f"{points[0].x},{points[0].y},{points[0].z}" + '''</coordinates>
            </Point>
        </Placemark>
        
        <!-- End Point -->
        <Placemark>
            <name>End</name>
            <Point>
                <coordinates>''' + f"{points[-1].x},{points[-1].y},{points[-1].z}" + '''</coordinates>
            </Point>
        </Placemark>
        
    </Document>
</kml>'''
            
            with open(filepath, 'w') as f:
                f.write(kml_content)
            
            return True
        except Exception as e:
            print(f"KML export failed: {e}")
            return False
    
    @staticmethod
    def export_to_svg(points: List[TrajectoryPoint], filepath: str, 
                     width: int = 800, height: int = 600) -> bool:
        """Export trajectory to SVG format."""
        try:
            positions = np.array([[p.x, p.y] for p in points])
            
            # Scale to fit SVG canvas
            min_x, max_x = positions[:, 0].min(), positions[:, 0].max()
            min_y, max_y = positions[:, 1].min(), positions[:, 1].max()
            
            margin = 50
            scale_x = (width - 2 * margin) / (max_x - min_x) if max_x != min_x else 1
            scale_y = (height - 2 * margin) / (max_y - min_y) if max_y != min_y else 1
            scale = min(scale_x, scale_y)
            
            # Transform coordinates
            svg_points = []
            for p in positions:
                svg_x = margin + (p[0] - min_x) * scale
                svg_y = height - (margin + (p[1] - min_y) * scale)  # Flip Y axis
                svg_points.append(f"{svg_x:.2f},{svg_y:.2f}")
            
            path_data = "M " + " L ".join(svg_points)
            
            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <style>
            .trajectory {{ fill: none; stroke: blue; stroke-width: 2; }}
            .start-point {{ fill: green; }}
            .end-point {{ fill: red; }}
        </style>
    </defs>
    
    <!-- Trajectory path -->
    <path d="{path_data}" class="trajectory"/>
    
    <!-- Start point -->
    <circle cx="{svg_points[0].split(',')[0]}" cy="{svg_points[0].split(',')[1]}" 
            r="5" class="start-point"/>
    
    <!-- End point -->
    <circle cx="{svg_points[-1].split(',')[0]}" cy="{svg_points[-1].split(',')[1]}" 
            r="5" class="end-point"/>
    
</svg>'''
            
            with open(filepath, 'w') as f:
                f.write(svg_content)
            
            return True
        except Exception as e:
            print(f"SVG export failed: {e}")
            return False


class BatchTrajectoryProcessor:
    """Batch processing of multiple trajectory files."""
    
    def __init__(self):
        """Initialize batch processor."""
        self.analyzer = TrajectoryAnalyzer()
        self.exporter = TrajectoryExporter()
    
    def process_directory(self, directory_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Process all trajectory files in a directory.
        
        Args:
            directory_path: Directory containing trajectory files
            output_dir: Output directory for results
            
        Returns:
            Summary of processing results
        """
        directory = Path(directory_path)
        if output_dir:
            output_directory = Path(output_dir)
            output_directory.mkdir(exist_ok=True)
        else:
            output_directory = directory / "analysis_results"
            output_directory.mkdir(exist_ok=True)
        
        results = {
            'processed_files': [],
            'failed_files': [],
            'statistics_summary': {},
            'patterns_summary': {},
            'comparisons': []
        }
        
        # Find all trajectory files
        trajectory_files = list(directory.glob("*.trj"))
        
        print(f"Processing {len(trajectory_files)} trajectory files...")
        
        trajectory_data = []
        
        for file_path in trajectory_files:
            try:
                print(f"Processing: {file_path.name}")
                
                # Load trajectory
                with TrajectoryReader(str(file_path)) as reader:
                    points = reader.read_all_points()
                    metadata = reader.get_metadata()
                
                if points:
                    # Analyze trajectory
                    stats = self.analyzer.analyze_trajectory(points)
                    patterns = self.analyzer.detect_patterns(points)
                    
                    # Store data for comparisons
                    trajectory_data.append({
                        'filename': file_path.name,
                        'points': points,
                        'stats': stats,
                        'patterns': patterns,
                        'metadata': metadata
                    })
                    
                    # Export analysis results
                    base_name = file_path.stem
                    
                    # Export statistics
                    stats_file = output_directory / f"{base_name}_stats.json"
                    with open(stats_file, 'w') as f:
                        json.dump(asdict(stats), f, indent=2)
                    
                    # Export patterns
                    patterns_file = output_directory / f"{base_name}_patterns.json"
                    with open(patterns_file, 'w') as f:
                        json.dump([asdict(p) for p in patterns], f, indent=2)
                    
                    # Export in multiple formats
                    self.exporter.export_to_csv(points, str(output_directory / f"{base_name}.csv"))
                    self.exporter.export_to_json(points, str(output_directory / f"{base_name}.json"), metadata)
                    
                    results['processed_files'].append(file_path.name)
                    
            except Exception as e:
                print(f"Failed to process {file_path.name}: {e}")
                results['failed_files'].append({'filename': file_path.name, 'error': str(e)})
        
        # Generate comparison matrix
        print("Computing trajectory comparisons...")
        if len(trajectory_data) > 1:
            for i in range(len(trajectory_data)):
                for j in range(i + 1, len(trajectory_data)):
                    try:
                        comparison = self.analyzer.compare_trajectories(
                            trajectory_data[i]['points'],
                            trajectory_data[j]['points']
                        )
                        
                        results['comparisons'].append({
                            'file1': trajectory_data[i]['filename'],
                            'file2': trajectory_data[j]['filename'],
                            'similarity_score': comparison.similarity_score,
                            'dtw_distance': comparison.dtw_distance,
                            'correlation': comparison.correlation_coefficient
                        })
                    except Exception as e:
                        print(f"Comparison failed between {trajectory_data[i]['filename']} and {trajectory_data[j]['filename']}: {e}")
        
        # Generate summary statistics
        if trajectory_data:
            all_stats = [td['stats'] for td in trajectory_data]
            results['statistics_summary'] = {
                'total_trajectories': len(all_stats),
                'avg_duration': np.mean([s.duration_seconds for s in all_stats]),
                'avg_points': np.mean([s.total_points for s in all_stats]),
                'avg_distance': np.mean([s.total_distance for s in all_stats]),
                'avg_velocity': np.mean([s.average_velocity for s in all_stats])
            }
            
            # Pattern summary
            all_patterns = []
            for td in trajectory_data:
                all_patterns.extend(td['patterns'])
            
            pattern_types = [p.pattern_type for p in all_patterns]
            results['patterns_summary'] = dict(Counter(pattern_types))
        
        # Save overall results
        results_file = output_directory / "batch_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Batch processing completed. Results saved to: {output_directory}")
        return results


# Example usage and testing
if __name__ == "__main__":
    print("Trajectory Analysis Tools Test")
    
    # Create sample trajectory data
    points = []
    for i in range(100):
        t = i * 0.1
        point = TrajectoryPoint(
            timestamp=t,
            x=2.0 * np.cos(t) + 0.5 * np.cos(3*t),
            y=2.0 * np.sin(t) + 0.5 * np.sin(3*t),
            z=0.2 * np.sin(2*t),
            gesture_strength=0.5 + 0.3 * np.sin(5*t),
            gesture_type=int(3 * (0.5 + 0.5 * np.cos(t))),
            r1=2.0, r2=0.5, w1=1.0, w2=3.0
        )
        points.append(point)
    
    # Test analysis
    analyzer = TrajectoryAnalyzer()
    
    print("\nAnalyzing trajectory...")
    stats = analyzer.analyze_trajectory(points)
    print(f"Statistics:")
    print(f"  Duration: {stats.duration_seconds:.2f}s")
    print(f"  Total distance: {stats.total_distance:.2f}")
    print(f"  Average velocity: {stats.average_velocity:.2f}")
    print(f"  Direction changes: {stats.direction_changes}")
    print(f"  Loops detected: {stats.loops_detected}")
    
    print("\nDetecting patterns...")
    patterns = analyzer.detect_patterns(points)
    print(f"Found {len(patterns)} patterns:")
    for pattern in patterns:
        print(f"  {pattern.pattern_type}: confidence={pattern.confidence:.2f}")
    
    # Test exports
    print("\nTesting exports...")
    exporter = TrajectoryExporter()
    
    success_json = exporter.export_to_json(points, "test_trajectory.json")
    success_csv = exporter.export_to_csv(points, "test_trajectory.csv")
    success_svg = exporter.export_to_svg(points, "test_trajectory.svg")
    success_kml = exporter.export_to_kml(points, "test_trajectory.kml")
    
    print(f"Export results: JSON={success_json}, CSV={success_csv}, SVG={success_svg}, KML={success_kml}")
    
    # Cleanup test files
    import os
    for filename in ["test_trajectory.json", "test_trajectory.csv", "test_trajectory.svg", "test_trajectory.kml"]:
        if os.path.exists(filename):
            os.remove(filename)
    
    print("\nTrajectory analysis tools test completed!")