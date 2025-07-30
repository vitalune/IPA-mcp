"""
Advanced Analytics Engine for Personal Data Insights

This module provides comprehensive statistical analysis and machine learning capabilities:
- Time series analysis and trend detection
- Statistical analysis and correlation discovery
- Anomaly detection and pattern recognition
- Clustering and segmentation analysis
- Recommendation engines
- Privacy-preserving analytics
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import structlog

from ..models.data_models import TimeRange, CommunicationPattern, MetricEntry
from ..config.settings import get_settings

logger = structlog.get_logger(__name__)

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)


class TrendDirection(str, Enum):
    """Trend direction classifications"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected"""
    OUTLIER = "outlier"
    PATTERN_BREAK = "pattern_break"
    TREND_CHANGE = "trend_change"
    SEASONAL_ANOMALY = "seasonal_anomaly"


@dataclass
class TimeSeriesPoint:
    """Represents a time series data point"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Results of trend analysis"""
    direction: TrendDirection
    strength: float  # 0-1, how strong the trend is
    confidence: float  # 0-1, confidence in the trend
    slope: float
    r_squared: float
    periods: int
    start_date: datetime
    end_date: datetime


@dataclass
class SeasonalityAnalysis:
    """Results of seasonality analysis"""
    has_seasonality: bool
    period: Optional[int]  # in days
    strength: float  # 0-1
    seasonal_component: Optional[List[float]]
    residual_component: Optional[List[float]]


@dataclass
class AnomalyDetection:
    """Results of anomaly detection"""
    anomalies: List[Dict[str, Any]]
    threshold: float
    method: str
    confidence_interval: Tuple[float, float]


@dataclass
class ClusterAnalysis:
    """Results of clustering analysis"""
    cluster_labels: List[int]
    n_clusters: int
    silhouette_score: float
    cluster_centers: Optional[List[List[float]]]
    cluster_metadata: Dict[int, Dict[str, Any]]


@dataclass
class CorrelationAnalysis:
    """Results of correlation analysis"""
    correlations: Dict[str, Dict[str, float]]
    significant_correlations: List[Tuple[str, str, float, float]]  # var1, var2, correlation, p_value
    correlation_matrix: Optional[np.ndarray]


@dataclass
class Recommendation:
    """A personalized recommendation"""
    type: str
    title: str
    description: str
    confidence: float
    priority: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class AnalyticsEngine:
    """Main analytics engine for personal data insights"""
    
    def __init__(self):
        self.settings = get_settings()
        self._scaler = StandardScaler()
        self._min_max_scaler = MinMaxScaler()
        self._anomaly_detectors = {}
        self._cached_analyses = {}
        
        # Privacy settings
        self._enable_differential_privacy = self.settings.privacy.anonymize_logs
        self._noise_scale = 0.1  # For differential privacy
    
    async def analyze_time_series(
        self,
        data: List[TimeSeriesPoint],
        detect_trends: bool = True,
        detect_seasonality: bool = True,
        detect_anomalies: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive time series analysis"""
        if len(data) < 3:
            logger.warning("Insufficient data for time series analysis")
            return {"error": "Insufficient data points"}
        
        # Convert to pandas DataFrame
        df = pd.DataFrame([
            {
                'timestamp': point.timestamp,
                'value': point.value,
                **point.metadata
            }
            for point in data
        ])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        results = {
            'data_points': len(data),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            }
        }
        
        # Trend analysis
        if detect_trends:
            results['trend'] = await self._analyze_trend(df)
        
        # Seasonality analysis
        if detect_seasonality and len(data) >= 14:  # Need at least 2 weeks
            results['seasonality'] = await self._analyze_seasonality(df)
        
        # Anomaly detection
        if detect_anomalies:
            results['anomalies'] = await self._detect_time_series_anomalies(df)
        
        # Basic statistics
        results['statistics'] = {
            'mean': float(df['value'].mean()),
            'median': float(df['value'].median()),
            'std': float(df['value'].std()),
            'min': float(df['value'].min()),
            'max': float(df['value'].max()),
            'skewness': float(stats.skew(df['value'])),
            'kurtosis': float(stats.kurtosis(df['value']))
        }
        
        return results
    
    async def _analyze_trend(self, df: pd.DataFrame) -> TrendAnalysis:
        """Analyze trend in time series data"""
        # Convert timestamps to numeric for regression
        df['timestamp_numeric'] = pd.to_datetime(df['timestamp']).astype(np.int64) // 10**9
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df['timestamp_numeric'], df['value']
        )
        
        # Determine trend direction
        if abs(slope) < std_err * 2:  # Not statistically significant
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING
        
        # Calculate trend strength (based on R-squared)
        r_squared = r_value ** 2
        
        # Check for volatility
        if df['value'].std() > df['value'].mean() * 0.5:  # High coefficient of variation
            if direction == TrendDirection.STABLE:
                direction = TrendDirection.VOLATILE
        
        return TrendAnalysis(
            direction=direction,
            strength=float(abs(slope) / (df['value'].std() + 1e-8)),
            confidence=float(1 - p_value) if p_value < 1 else 0.0,
            slope=float(slope),
            r_squared=float(r_squared),
            periods=len(df),
            start_date=df['timestamp'].min(),
            end_date=df['timestamp'].max()
        )
    
    async def _analyze_seasonality(self, df: pd.DataFrame) -> SeasonalityAnalysis:
        """Analyze seasonality in time series data"""
        try:
            # Simple seasonality detection using autocorrelation
            values = df['value'].values
            n = len(values)
            
            # Test for weekly seasonality (7 days)
            if n >= 14:
                autocorr_7 = np.corrcoef(values[:-7], values[7:])[0, 1]
                if not np.isnan(autocorr_7) and autocorr_7 > 0.3:
                    return SeasonalityAnalysis(
                        has_seasonality=True,
                        period=7,
                        strength=float(abs(autocorr_7)),
                        seasonal_component=None,
                        residual_component=None
                    )
            
            # Test for monthly seasonality (30 days)
            if n >= 60:
                autocorr_30 = np.corrcoef(values[:-30], values[30:])[0, 1]
                if not np.isnan(autocorr_30) and autocorr_30 > 0.3:
                    return SeasonalityAnalysis(
                        has_seasonality=True,
                        period=30,
                        strength=float(abs(autocorr_30)),
                        seasonal_component=None,
                        residual_component=None
                    )
            
            return SeasonalityAnalysis(
                has_seasonality=False,
                period=None,
                strength=0.0,
                seasonal_component=None,
                residual_component=None
            )
            
        except Exception as e:
            logger.warning(f"Seasonality analysis failed: {e}")
            return SeasonalityAnalysis(
                has_seasonality=False,
                period=None,
                strength=0.0,
                seasonal_component=None,
                residual_component=None
            )
    
    async def _detect_time_series_anomalies(self, df: pd.DataFrame) -> AnomalyDetection:
        """Detect anomalies in time series data"""
        values = df['value'].values.reshape(-1, 1)
        
        # Use Isolation Forest for anomaly detection
        detector = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = detector.fit_predict(values)
        
        # Find anomaly points
        anomalies = []
        for i, label in enumerate(anomaly_labels):
            if label == -1:  # Anomaly
                anomalies.append({
                    'timestamp': df.iloc[i]['timestamp'].isoformat(),
                    'value': float(df.iloc[i]['value']),
                    'type': AnomalyType.OUTLIER.value,
                    'index': i,
                    'anomaly_score': float(detector.score_samples(values[i].reshape(1, -1))[0])
                })
        
        # Calculate confidence interval
        mean_val = df['value'].mean()
        std_val = df['value'].std()
        confidence_interval = (
            float(mean_val - 2 * std_val),
            float(mean_val + 2 * std_val)
        )
        
        return AnomalyDetection(
            anomalies=anomalies,
            threshold=0.1,
            method="isolation_forest",
            confidence_interval=confidence_interval
        )
    
    async def analyze_correlations(
        self,
        data: Dict[str, List[float]],
        significance_level: float = 0.05
    ) -> CorrelationAnalysis:
        """Analyze correlations between multiple variables"""
        if len(data) < 2:
            logger.warning("Need at least 2 variables for correlation analysis")
            return CorrelationAnalysis({}, [], None)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Find statistically significant correlations
        significant_correlations = []
        variables = list(data.keys())
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables[i+1:], i+1):
                corr_coef = corr_matrix.loc[var1, var2]
                if not np.isnan(corr_coef):
                    # Calculate p-value
                    n = len(data[var1])
                    if n > 2:
                        t_stat = corr_coef * np.sqrt((n - 2) / (1 - corr_coef**2))
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                        
                        if p_value < significance_level and abs(corr_coef) > 0.1:
                            significant_correlations.append((
                                var1, var2, float(corr_coef), float(p_value)
                            ))
        
        # Convert correlation matrix to dict
        correlations = {}
        for var1 in variables:
            correlations[var1] = {}
            for var2 in variables:
                if not np.isnan(corr_matrix.loc[var1, var2]):
                    correlations[var1][var2] = float(corr_matrix.loc[var1, var2])
        
        return CorrelationAnalysis(
            correlations=correlations,
            significant_correlations=significant_correlations,
            correlation_matrix=corr_matrix.values
        )
    
    async def cluster_data(
        self,
        data: List[Dict[str, Any]],
        features: List[str],
        n_clusters: Optional[int] = None,
        method: str = 'kmeans'
    ) -> ClusterAnalysis:
        """Perform clustering analysis on data"""
        if len(data) < 2:
            logger.warning("Insufficient data for clustering")
            return ClusterAnalysis([], 0, 0.0, None, {})
        
        # Extract feature matrix
        feature_matrix = []
        for item in data:
            row = []
            for feature in features:
                value = item.get(feature, 0)
                if isinstance(value, (int, float)):
                    row.append(value)
                else:
                    row.append(0)  # Default for non-numeric
            feature_matrix.append(row)
        
        X = np.array(feature_matrix)
        
        # Standardize features
        X_scaled = self._scaler.fit_transform(X)
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X_scaled, max_k=min(10, len(data)//2))
        
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(X_scaled)
            cluster_centers = clusterer.cluster_centers_.tolist()
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            cluster_labels = clusterer.fit_predict(X_scaled)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            cluster_centers = None
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Calculate silhouette score
        if n_clusters > 1 and len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        else:
            silhouette_avg = 0.0
        
        # Generate cluster metadata
        cluster_metadata = {}
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise points in DBSCAN
                continue
            
            cluster_points = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_data = [data[i] for i in cluster_points]
            
            cluster_metadata[cluster_id] = {
                'size': len(cluster_points),
                'indices': cluster_points,
                'statistics': self._calculate_cluster_statistics(cluster_data, features)
            }
        
        return ClusterAnalysis(
            cluster_labels=cluster_labels.tolist(),
            n_clusters=n_clusters,
            silhouette_score=float(silhouette_avg),
            cluster_centers=cluster_centers,
            cluster_metadata=cluster_metadata
        )
    
    def _find_optimal_clusters(self, X: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method"""
        if max_k < 2:
            return 1
        
        inertias = []
        k_range = range(1, min(max_k + 1, len(X)))
        
        for k in k_range:
            if k == 1:
                inertias.append(np.sum((X - X.mean(axis=0))**2))
            else:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
        
        # Find elbow using second derivative
        if len(inertias) < 3:
            return min(2, len(X) // 2)
        
        # Calculate second differences
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        
        # Find the point with maximum second difference
        elbow_idx = np.argmax(second_diffs) + 2  # +2 because of double diff
        return min(elbow_idx, max_k)
    
    def _calculate_cluster_statistics(
        self,
        cluster_data: List[Dict[str, Any]],
        features: List[str]
    ) -> Dict[str, Any]:
        """Calculate statistics for a cluster"""
        stats_dict = {}
        
        for feature in features:
            values = [item.get(feature, 0) for item in cluster_data if isinstance(item.get(feature), (int, float))]
            if values:
                stats_dict[feature] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        return stats_dict
    
    async def detect_patterns(
        self,
        data: List[Dict[str, Any]],
        pattern_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect various patterns in data"""
        if not data:
            return {}
        
        patterns = {}
        
        # Temporal patterns
        if 'temporal' in (pattern_types or ['temporal']):
            patterns['temporal'] = await self._detect_temporal_patterns(data)
        
        # Frequency patterns
        if 'frequency' in (pattern_types or ['frequency']):
            patterns['frequency'] = await self._detect_frequency_patterns(data)
        
        # Behavioral patterns
        if 'behavioral' in (pattern_types or ['behavioral']):
            patterns['behavioral'] = await self._detect_behavioral_patterns(data)
        
        return patterns
    
    async def _detect_temporal_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect temporal patterns in data"""
        timestamps = []
        for item in data:
            if 'timestamp' in item:
                if isinstance(item['timestamp'], str):
                    timestamps.append(pd.to_datetime(item['timestamp']))
                elif isinstance(item['timestamp'], datetime):
                    timestamps.append(item['timestamp'])
        
        if not timestamps:
            return {}
        
        df = pd.DataFrame({'timestamp': timestamps})
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        
        patterns = {
            'peak_hours': df['hour'].value_counts().head(3).to_dict(),
            'peak_days': df['day_of_week'].value_counts().head(3).to_dict(),
            'activity_distribution': {
                'morning': len(df[(df['hour'] >= 6) & (df['hour'] < 12)]),
                'afternoon': len(df[(df['hour'] >= 12) & (df['hour'] < 18)]),
                'evening': len(df[(df['hour'] >= 18) & (df['hour'] < 24)]),
                'night': len(df[(df['hour'] >= 0) & (df['hour'] < 6)])
            }
        }
        
        return patterns
    
    async def _detect_frequency_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect frequency patterns in categorical data"""
        patterns = {}
        
        # Find categorical fields
        categorical_fields = []
        if data:
            for key, value in data[0].items():
                if isinstance(value, str) and key != 'timestamp':
                    categorical_fields.append(key)
        
        for field in categorical_fields:
            values = [item.get(field) for item in data if item.get(field)]
            if values:
                value_counts = pd.Series(values).value_counts()
                patterns[field] = {
                    'top_values': value_counts.head(5).to_dict(),
                    'unique_count': len(value_counts),
                    'entropy': float(stats.entropy(value_counts.values))
                }
        
        return patterns
    
    async def _detect_behavioral_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect behavioral patterns in user data"""
        patterns = {}
        
        # Response time patterns (if available)
        response_times = [item.get('response_time') for item in data if item.get('response_time')]
        if response_times:
            patterns['response_time'] = {
                'mean': float(np.mean(response_times)),
                'median': float(np.median(response_times)),
                'std': float(np.std(response_times)),
                'percentiles': {
                    '25th': float(np.percentile(response_times, 25)),
                    '75th': float(np.percentile(response_times, 75)),
                    '95th': float(np.percentile(response_times, 95))
                }
            }
        
        # Session duration patterns
        session_durations = [item.get('duration') for item in data if item.get('duration')]
        if session_durations:
            patterns['session_duration'] = {
                'mean': float(np.mean(session_durations)),
                'median': float(np.median(session_durations)),
                'distribution': {
                    'short': len([d for d in session_durations if d < 300]),  # < 5 min
                    'medium': len([d for d in session_durations if 300 <= d < 1800]),  # 5-30 min
                    'long': len([d for d in session_durations if d >= 1800])  # > 30 min
                }
            }
        
        return patterns
    
    async def generate_recommendations(
        self,
        user_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Recommendation]:
        """Generate personalized recommendations based on data analysis"""
        recommendations = []
        
        # Productivity recommendations
        productivity_recs = await self._generate_productivity_recommendations(user_data)
        recommendations.extend(productivity_recs)
        
        # Communication recommendations
        comm_recs = await self._generate_communication_recommendations(user_data)
        recommendations.extend(comm_recs)
        
        # Content recommendations
        content_recs = await self._generate_content_recommendations(user_data)
        recommendations.extend(content_recs)
        
        # Sort by priority and confidence
        recommendations.sort(key=lambda x: (x.priority, -x.confidence))
        
        return recommendations[:10]  # Return top 10
    
    async def _generate_productivity_recommendations(
        self,
        user_data: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate productivity-focused recommendations"""
        recommendations = []
        
        # Analyze work patterns
        if 'work_hours' in user_data:
            work_hours = user_data['work_hours']
            if isinstance(work_hours, list) and len(work_hours) > 0:
                avg_hours = np.mean(work_hours)
                if avg_hours > 50:
                    recommendations.append(Recommendation(
                        type="productivity",
                        title="Consider Work-Life Balance",
                        description=f"You're averaging {avg_hours:.1f} hours per week. Consider setting boundaries to prevent burnout.",
                        confidence=0.8,
                        priority=1,
                        metadata={"current_hours": avg_hours}
                    ))
        
        # Analyze peak productivity times
        if 'activity_by_hour' in user_data:
            activity = user_data['activity_by_hour']
            if isinstance(activity, dict):
                peak_hour = max(activity.keys(), key=lambda x: activity[x])
                recommendations.append(Recommendation(
                    type="productivity",
                    title="Optimize Your Schedule",
                    description=f"Your peak activity is around {peak_hour}:00. Schedule important tasks during this time.",
                    confidence=0.7,
                    priority=2,
                    metadata={"peak_hour": peak_hour}
                ))
        
        return recommendations
    
    async def _generate_communication_recommendations(
        self,
        user_data: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate communication-focused recommendations"""
        recommendations = []
        
        # Analyze response patterns
        if 'avg_response_time' in user_data:
            response_time = user_data['avg_response_time']
            if response_time > 24:  # More than 24 hours
                recommendations.append(Recommendation(
                    type="communication",
                    title="Improve Response Time",
                    description=f"Your average response time is {response_time:.1f} hours. Consider setting up email filters or scheduled check-ins.",
                    confidence=0.6,
                    priority=3,
                    metadata={"current_response_time": response_time}
                ))
        
        # Analyze communication volume
        if 'daily_emails' in user_data:
            daily_emails = user_data['daily_emails']
            if isinstance(daily_emails, list):
                avg_emails = np.mean(daily_emails)
                if avg_emails > 100:
                    recommendations.append(Recommendation(
                        type="communication",
                        title="Email Overload Management",
                        description=f"You receive {avg_emails:.0f} emails per day. Consider unsubscribing from newsletters or using filters.",
                        confidence=0.7,
                        priority=2,
                        metadata={"daily_average": avg_emails}
                    ))
        
        return recommendations
    
    async def _generate_content_recommendations(
        self,
        user_data: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate content-focused recommendations"""
        recommendations = []
        
        # Analyze content engagement
        if 'content_performance' in user_data:
            performance = user_data['content_performance']
            if isinstance(performance, dict):
                best_type = max(performance.keys(), key=lambda x: performance[x].get('engagement', 0))
                recommendations.append(Recommendation(
                    type="content",
                    title="Focus on High-Performing Content",
                    description=f"Your {best_type} content performs best. Consider creating more similar content.",
                    confidence=0.8,
                    priority=1,
                    metadata={"best_content_type": best_type}
                ))
        
        return recommendations
    
    async def calculate_privacy_metrics(
        self,
        data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate privacy-preserving metrics"""
        if not data or not self._enable_differential_privacy:
            return {}
        
        # Add noise to protect privacy
        def add_noise(value: float) -> float:
            return value + np.random.laplace(0, self._noise_scale)
        
        metrics = {}
        
        # Basic statistics with noise
        numeric_fields = []
        if data:
            for key, value in data[0].items():
                if isinstance(value, (int, float)):
                    numeric_fields.append(key)
        
        for field in numeric_fields:
            values = [item.get(field, 0) for item in data if isinstance(item.get(field), (int, float))]
            if values:
                metrics[field] = {
                    'count': len(values),
                    'mean': add_noise(float(np.mean(values))),
                    'std': add_noise(float(np.std(values))),
                    'min': add_noise(float(np.min(values))),
                    'max': add_noise(float(np.max(values)))
                }
        
        return metrics
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get analytics engine status"""
        return {
            'cached_analyses': len(self._cached_analyses),
            'anomaly_detectors': len(self._anomaly_detectors),
            'differential_privacy_enabled': self._enable_differential_privacy,
            'noise_scale': self._noise_scale,
            'scaler_fitted': hasattr(self._scaler, 'mean_'),
        }


# Global analytics engine instance
_analytics_engine: Optional[AnalyticsEngine] = None


def get_analytics_engine() -> AnalyticsEngine:
    """Get the global analytics engine instance"""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = AnalyticsEngine()
    return _analytics_engine


async def quick_trend_analysis(
    timestamps: List[datetime],
    values: List[float]
) -> TrendAnalysis:
    """Quick trend analysis for time series data"""
    engine = get_analytics_engine()
    data_points = [
        TimeSeriesPoint(timestamp=ts, value=val)
        for ts, val in zip(timestamps, values)
    ]
    result = await engine.analyze_time_series(data_points, detect_seasonality=False, detect_anomalies=False)
    return result.get('trend')


async def quick_correlation_analysis(
    data: Dict[str, List[float]]
) -> CorrelationAnalysis:
    """Quick correlation analysis for multiple variables"""
    engine = get_analytics_engine()
    return await engine.analyze_correlations(data)