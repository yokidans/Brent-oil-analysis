# src/analysis/causal_analysis.py
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dowhy import CausalModel
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LinearRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import TimeSeriesSplit
from econml.metalearners import XLearner, SLearner, TLearner
from econml.dr import DRLearner
from econml.sklearn_extensions.linear_model import WeightedLassoCV
from typing import Dict, Union, Optional, List, Tuple, Any
import warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from datetime import datetime
import logging
from enum import Enum, auto
import hashlib
import json
import seaborn as sns
from matplotlib import gridspec
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', module='econml')

class EstimationMethod(Enum):
    """Enumeration of available causal estimation methods"""
    AUTO = auto()
    DML = auto()
    DR = auto()
    XLEARNER = auto()
    SLEARNER = auto()
    TLEARNER = auto()
    PSM = auto()
    SYNTH = auto()

class CausalImpactAnalyzer:
    """
    Robust causal impact analysis with enhanced features:
    - Advanced string handling and validation
    - Multiple estimation methods (DML, DR, XLearner, PSM, Synthetic Control)
    - Heterogeneous treatment effects
    - Sensitivity analysis
    - Parallel processing support
    - Comprehensive logging
    - Visualization capabilities
    """

    def __init__(self, 
                 price_data: pd.DataFrame,
                 event_data: pd.DataFrame,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 progress_bar: bool = True,
                 report_path: str = "reports"):
        """
        Initialize with data validation and preprocessing
        
        Args:
            price_data: DataFrame with prices (must have datetime index)
            event_data: DataFrame containing events with start/end dates
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
            progress_bar: Whether to show progress bars
            report_path: Path to save analysis reports
        """
        np.random.seed(random_state)
        self.price_data = self._validate_price_data(price_data)
        self.event_data = self._validate_event_data(event_data)
        self.n_jobs = n_jobs if n_jobs != -1 else os.cpu_count()
        self.random_state = random_state
        self.progress_bar = progress_bar
        self.REPORT_PATH = Path(report_path)
        self._setup_reporting()
        self.causal_graph = self._build_causal_graph()
        self._precompute_features()
        
        # Cache for storing intermediate results
        self._cache: Dict[str, Any] = {}
        
        logger.info("CausalImpactAnalyzer initialized successfully")

    def _validate_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure price data is properly formatted with comprehensive checks"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("price_data must be a pandas DataFrame")
            
        if df.empty:
            raise ValueError("price_data cannot be empty")
            
        # Handle index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Could not convert index to datetime: {str(e)}")
        
        # Check for required columns
        required_cols = {'Price'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in price_data: {missing_cols}")
            
        # Check for duplicates
        if df.index.duplicated().any():
            logger.warning("Duplicate timestamps found in price_data - taking first occurrence")
            df = df[~df.index.duplicated(keep='first')]
            
        # Sort by index
        df = df.sort_index()
        
        # Check for missing values
        if df['Price'].isnull().any():
            logger.warning("Price data contains missing values - forward filling")
            df['Price'] = df['Price'].ffill()
            if df['Price'].isnull().any():
                df['Price'] = df['Price'].bfill()
                
        return df

    def _validate_event_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure event data has required columns and proper dates with thorough validation"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("event_data must be a pandas DataFrame")
            
        if df.empty:
            raise ValueError("event_data cannot be empty")
            
        required_cols = {'event_type', 'start_date'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in event_data: {missing_cols}")
            
        df = df.copy()
        
        # Clean event types
        df['event_type'] = (
            df['event_type']
            .astype(str)
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace
        )
        
        # Handle dates - convert to datetime with error handling
        for col in ['start_date', 'end_date']:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    raise ValueError(f"Could not convert {col} to datetime: {str(e)}")
        
        # Add end_date if not present
        if 'end_date' not in df.columns:
            df['end_date'] = df['start_date'] + pd.Timedelta(days=30)
        
        # Validate dates
        df = df.dropna(subset=['start_date', 'end_date', 'event_type'])
        if df.empty:
            raise ValueError("No valid events remaining after removing rows with missing dates/types")
            
        if (df['end_date'] < df['start_date']).any():
            invalid_events = df[df['end_date'] < df['start_date']]
            raise ValueError(
                f"Some end dates are before start dates:\n"
                f"{invalid_events[['event_type', 'start_date', 'end_date']]}"
            )
            
        # Check for overlapping events of the same type
        self._check_event_overlaps(df)
            
        return df

    def _check_event_overlaps(self, df: pd.DataFrame) -> None:
        """Check for overlapping events of the same type"""
        df_sorted = df.sort_values('start_date').copy()
        df_sorted['next_start'] = df_sorted.groupby('event_type')['start_date'].shift(-1)
        df_sorted['overlap'] = df_sorted['end_date'] > df_sorted['next_start']
        
        overlapping = df_sorted[df_sorted['overlap']]
        if not overlapping.empty:
            logger.warning(
                f"Found {len(overlapping)} overlapping events of the same type:\n"
                f"{overlapping[['event_type', 'start_date', 'end_date']]}"
            )

    def _precompute_features(self) -> None:
        """Precompute temporal and technical features"""
        logger.info("Precomputing features...")
        
        # Basic temporal features
        self.price_data['day_of_week'] = self.price_data.index.dayofweek
        self.price_data['month'] = self.price_data.index.month
        self.price_data['quarter'] = self.price_data.index.quarter
        self.price_data['year'] = self.price_data.index.year
        
        # Fourier terms for seasonality
        day_of_year = self.price_data.index.dayofyear
        for k in range(1, 4):  # Adding more Fourier terms
            self.price_data[f'fourier_sin_{k}'] = np.sin(2 * k * np.pi * day_of_year / 365.25)
            self.price_data[f'fourier_cos_{k}'] = np.cos(2 * k * np.pi * day_of_year / 365.25)
        
        # Add returns and volatility if not present
        if 'Log_Return' not in self.price_data.columns:
            self.price_data['Log_Return'] = np.log(self.price_data['Price'] / self.price_data['Price'].shift(1))
        
        # Volatility measures
        for window in [7, 30, 90]:  # Multiple volatility windows
            self.price_data[f'volatility_{window}d'] = (
                self.price_data['Log_Return']
                .rolling(window)
                .std()
                .fillna(method='ffill')
                .fillna(0)
            )
        
        # Technical indicators
        self._add_technical_indicators()
        
        logger.info("Feature precomputation completed")

    def _add_technical_indicators(self) -> None:
        """Add common technical indicators"""
        prices = self.price_data['Price']
        
        # Moving averages
        for window in [10, 20, 50]:
            self.price_data[f'ma_{window}'] = prices.rolling(window).mean()
        
        # Bollinger Bands
        self.price_data['bb_upper'] = self.price_data['ma_20'] + 2 * prices.rolling(20).std()
        self.price_data['bb_lower'] = self.price_data['ma_20'] - 2 * prices.rolling(20).std()
        
        # RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        self.price_data['rsi'] = 100 - (100 / (1 + rs))
        
        # Fill NA values
        self.price_data = self.price_data.fillna(method='ffill').fillna(0)

    def calculate_pre_post_means(self, 
                           treatment: str, 
                           pre_window: int = 30,
                           post_window: Optional[int] = None) -> dict:
        """
        Compute average prices before/after events with enhanced functionality
        
        Args:
            treatment: Event type (e.g., 'OPEC_Decisions')
            pre_window: Days before event to consider as pre-period
            post_window: Days after event to consider (defaults to event duration)
            
        Returns:
            Dictionary with pre/post comparison results and statistics
        """
        treatment = self._validate_treatment(treatment)
        treatment_events = self.event_data[self.event_data['event_type'] == treatment]
        
        if treatment_events.empty:
            logger.warning(f"No events found for treatment: {treatment}")
            return {
                'pre_period_mean': np.nan,
                'post_period_mean': np.nan,
                'n_events': 0,
                'effect_size': np.nan,
                'p_value': np.nan,
                'effect_size_relative': np.nan
            }
        
        pre_vals, post_vals = [], []
        
        for _, event in treatment_events.iterrows():
            start = event['start_date']
            end = event['end_date']
            post_days = (end - start).days if post_window is None else post_window
            
            # Pre-event window (excludes event day)
            pre_start = start - pd.Timedelta(days=pre_window)
            pre_data = self.price_data.loc[pre_start:start - pd.Timedelta(days=1), 'Price']
            
            # Event period
            post_end = start + pd.Timedelta(days=post_days)
            event_data = self.price_data.loc[start:post_end, 'Price']
            
            if not pre_data.empty and not event_data.empty:
                pre_vals.append(pre_data.mean())
                post_vals.append(event_data.mean())
        
        if not pre_vals:
            logger.warning(f"No valid pre/post periods for treatment: {treatment}")
            return {
                'pre_period_mean': np.nan,
                'post_period_mean': np.nan,
                'n_events': 0,
                'effect_size': np.nan,
                'p_value': np.nan,
                'effect_size_relative': np.nan
            }
        
        # Calculate statistics
        pre_mean = np.mean(pre_vals)
        post_mean = np.mean(post_vals)
        effect_size = post_mean - pre_mean
        effect_size_relative = effect_size / pre_mean if pre_mean != 0 else np.nan
        
        # Simple t-test for significance
        from scipy.stats import ttest_rel
        p_value = np.nan
        if len(pre_vals) >= 2 and len(post_vals) >= 2:  # Need at least 2 observations
            try:
                _, p_value = ttest_rel(post_vals, pre_vals)
            except Exception as e:
                logger.warning(f"T-test failed: {str(e)}")
        else:
            logger.debug(f"Insufficient data for t-test (pre: {len(pre_vals)}, post: {len(post_vals)})")
        
        return {
            'pre_period_mean': pre_mean,
            'post_period_mean': post_mean,
            'n_events': len(pre_vals),
            'effect_size': effect_size,
            'p_value': p_value,
            'effect_size_relative': effect_size_relative,
            'pre_values': pre_vals,
            'post_values': post_vals
        }

    def _validate_treatment(self, treatment: str) -> str:
        """Ensure treatment variable exactly matches event data with enhanced matching"""
        if not isinstance(treatment, str):
            raise TypeError(f"Treatment must be a string, got {type(treatment)}")
            
        unique_types = self.event_data['event_type'].unique()
        treatment = str(treatment).strip()
        
        # Check for exact match first
        if treatment in unique_types:
            return treatment
            
        # Check case-insensitive match with whitespace normalization
        matches = [t for t in unique_types 
                  if str(t).strip().lower() == treatment.lower()]
        
        if not matches:
            available_types = sorted(list(unique_types))
            raise ValueError(
                f"Treatment '{treatment}' not found in event types.\n"
                f"Available types: {available_types}\n"
                f"Closest matches: {self._find_closest_matches(treatment, available_types)}"
            )
        
        if len(matches) > 1:
            logger.warning(f"Multiple matches found for '{treatment}': {matches}. Using first match.")
        
        return matches[0]

    def _find_closest_matches(self, query: str, options: List[str], n: int = 3) -> List[str]:
        """Find closest string matches using Levenshtein distance"""
        from difflib import get_close_matches
        return get_close_matches(query, options, n=n, cutoff=0.6)

    def _prepare_causal_data(self, treatment: str) -> pd.DataFrame:
        """Prepare dataframe for causal analysis with enhanced features"""
        # Get validated treatment string
        treatment = self._validate_treatment(treatment)
        cache_key = f"causal_data_{treatment}_{hashlib.md5(str(self.price_data.index).encode()).hexdigest()}"
        
        if cache_key in self._cache:
            logger.info(f"Using cached data for treatment: {treatment}")
            return self._cache[cache_key]
        
        logger.info(f"Preparing causal data for treatment: {treatment}")
        
        df = self.price_data.copy()
        df[treatment] = 0  # Initialize treatment column
        
        # Get matching events using the exact string
        mask = self.event_data['event_type'] == treatment
        
        if not mask.any():
            logger.warning(f"No events found for treatment: {treatment}")
            return self._add_macro_controls(df)
        
        # Mark treatment periods
        for _, event in self.event_data[mask].iterrows():
            start = event['start_date']
            end = event['end_date']
            df.loc[start:end, treatment] = 1
        
        result = self._add_macro_controls(df)
        self._cache[cache_key] = result
        
        # Add type validation before returning
        logger.debug(f"Treatment data type: {type(df[treatment])}")
        logger.debug(f"Price data type: {type(df['Price'])}")
        logger.debug(f"Treatment values sample: {df[treatment].head()}")
        logger.debug(f"Price values sample: {df['Price'].head()}")
        
        return df
        

    def _add_macro_controls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add relevant macroeconomic control variables with realistic patterns"""
        # Generate more realistic synthetic controls with time-series properties
        n = len(df)
        index = df.index
        
        # USD Index - random walk with drift
        usd = np.cumsum(np.random.normal(0.01, 0.5, n)) + 100
        usd = np.maximum(usd, 80)  # Floor at 80
        df['USD_Index'] = usd
        
        # Global Demand - seasonal pattern
        t = np.arange(n)
        seasonal = 0.5 * np.sin(2 * np.pi * t / 365) + 1.0
        df['Global_Demand'] = seasonal + np.random.normal(0, 0.1, n)
        
        # Inventories - mean-reverting process
        inv = np.zeros(n)
        for i in range(1, n):
            inv[i] = 0.9 * inv[i-1] + np.random.normal(0, 0.5)
        df['Inventories'] = inv
        
        # Geopolitical Risk - spikes during certain periods
        risk = np.random.normal(0, 0.5, n)
        spike_dates = pd.date_range(df.index.min(), df.index.max(), periods=10)
        for date in spike_dates:
            idx = df.index.get_indexer([date], method='nearest')[0]
            # Ensure we don't go out of bounds
            end_idx = min(idx + 5, n)
            spike_size = np.random.uniform(2, 5, size=end_idx - idx)
            risk[idx:end_idx] += spike_size
        
        df['Geopolitical_Risk'] = risk
        
        # Economic Growth - trending with business cycles
        growth = np.cumsum(np.random.normal(0.01, 0.2, n)) + 3.0
        df['Economic_Growth'] = growth
        
        # Add interactions
        df['Demand_Growth_Interaction'] = df['Global_Demand'] * df['Economic_Growth']
        
        return df

    def _build_causal_graph(self) -> nx.DiGraph:
        """Build causal graph with automated structure learning and fallback"""
        try:
            from causalnex.structure import PC
            pc = PC()
            df = self._get_base_features()
            
            # Use time-series aware structure learning
            sm = pc.learn_structure(df.values, max_cond_vars=5)
            G = nx.DiGraph(sm)
            
            # Ensure treatment-outcome path exists for all event types
            for event_type in self.event_data['event_type'].unique():
                if event_type not in G.nodes:
                    G.add_node(event_type)
                if not nx.has_path(G, event_type, 'Price'):
                    G.add_edge(event_type, 'Price')
                    
            return G
        except ImportError:
            logger.warning("causalnex not available - using expert graph")
            return self._build_expert_graph()

    def _build_expert_graph(self) -> nx.DiGraph:
        """Build expert-defined causal graph"""
        G = nx.DiGraph()
        
        # Add all event types as treatment nodes
        for event_type in self.event_data['event_type'].unique():
            G.add_node(event_type)
            G.add_edge(event_type, 'Price')
        
        # Add economic relationships
        economic_nodes = [
            'USD_Index', 'Global_Demand', 'Inventories',
            'Geopolitical_Risk', 'Economic_Growth'
        ]
        
        for node in economic_nodes:
            G.add_edge(node, 'Price')
        
        # Add interactions
        G.add_edges_from([
            ('Economic_Growth', 'Global_Demand'),
            ('Economic_Growth', 'USD_Index'),
            ('Geopolitical_Risk', 'USD_Index'),
            ('Global_Demand', 'Inventories')
        ])
        
        return G

    def _get_base_features(self) -> pd.DataFrame:
        """Get features for causal discovery with time-series awareness"""
        df = self.price_data[['Price']].copy()
        controls = self._add_macro_controls(pd.DataFrame(index=df.index))
        return df.join(controls)

    def run_basic_analysis(self, treatment: str = 'OPEC_Decisions') -> dict:
        """
        Run comprehensive basic analysis including:
        - Pre/post comparison
        - Regression adjustment
        - Event study plot data
        """
        treatment = self._validate_treatment(treatment)
        
        results = {
            'treatment': treatment,
            'n_events': len(self.event_data[self.event_data['event_type'] == treatment]),
            'time_period': {
                'start': self.price_data.index.min().strftime('%Y-%m-%d'),
                'end': self.price_data.index.max().strftime('%Y-%m-%d')
            },
            'price_stats': {
                'mean': self.price_data['Price'].mean(),
                'volatility': self.price_data['Log_Return'].std() if 'Log_Return' in self.price_data.columns else np.nan,
                'min': self.price_data['Price'].min(),
                'max': self.price_data['Price'].max()
            }
        }
        
        # Add pre/post comparison with multiple windows
        windows = [7, 30, 90]
        results['pre_post'] = {
            str(w): self.calculate_pre_post_means(treatment, pre_window=w)
            for w in windows
        }
        
        # Add regression-adjusted effect with multiple models
        results['adjusted_effects'] = self._run_regression_adjustments(treatment)
        
        # Add event study data
        results['event_study'] = self._prepare_event_study_data(treatment)
        
        return results

    def _run_regression_adjustments(self, treatment: str) -> Dict[str, Dict]:
        """Run multiple regression adjustment models"""
        df = self._prepare_causal_data(treatment)
        treatment_var = df[treatment]
        y = df['Price']
        
        models = {
            'linear': LinearRegression(),
            'lasso': LassoCV(cv=3, random_state=self.random_state),
            'gradient_boosting': HistGradientBoostingRegressor(
                random_state=self.random_state,
                max_iter=100
            )
        }
        
        results = {}
        for name, model in models.items():
            # Select features based on model type
            if name == 'linear':
                features = ['day_of_week', 'month', 'volatility_30d']
            else:
                features = [col for col in df.columns 
                           if col not in ['Price', treatment]]
            
            X = df[features]
            
            # Fit model
            model.fit(X, y)
            predicted = model.predict(X)
            
            # Calculate ATE
            ate = (y[treatment_var == 1] - predicted[treatment_var == 1]).mean() - \
                  (y[treatment_var == 0] - predicted[treatment_var == 0]).mean()
            
            results[name] = {
                'ate': ate,
                'model_score': model.score(X, y),
                'features': features,
                'model_params': model.get_params() if hasattr(model, 'get_params') else {}
            }
        
        return results

    def _prepare_event_study_data(self, treatment: str) -> Dict:
        """Prepare data for event study visualization"""
        treatment_events = self.event_data[self.event_data['event_type'] == treatment]
        if treatment_events.empty:
            return {}
        
        # Align events to common timeline (-30 to +30 days)
        event_windows = []
        for _, event in treatment_events.iterrows():
            start = event['start_date']
            pre_window = start - pd.Timedelta(days=30)
            post_window = start + pd.Timedelta(days=30)
            
            window_data = self.price_data.loc[pre_window:post_window, 'Price'].copy()
            window_data.index = (window_data.index - start).days
            event_windows.append(window_data)
        
        # Combine all events
        combined = pd.concat(event_windows, axis=1)
        mean_series = combined.mean(axis=1)
        std_series = combined.std(axis=1)
        
        return {
            'event_day': 0,
            'days': mean_series.index.tolist(),
            'mean_price': mean_series.tolist(),
            'std_price': std_series.tolist(),
            'normalized': (mean_series / mean_series.loc[-30]).tolist()
        }

    def estimate_effect(self,
                    treatment: str,
                    outcome: str = 'Price',
                    method: Union[str, EstimationMethod] = EstimationMethod.AUTO,
                    heterogeneity: bool = True,
                    sensitivity: bool = True,
                    confidence_level: float = 0.95) -> Dict[str, Union[float, dict]]:
        """Advanced effect estimation with robust error handling"""
        # Validate inputs
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
            
        if isinstance(method, str):
            method = EstimationMethod[method.upper()]
            
        treatment = self._validate_treatment(treatment)
        df = self._prepare_causal_data(treatment)
        
        # Ensure numeric types
        if not pd.api.types.is_numeric_dtype(df[outcome]):
            df[outcome] = pd.to_numeric(df[outcome], errors='coerce')
        if not pd.api.types.is_numeric_dtype(df[treatment]):
            df[treatment] = pd.to_numeric(df[treatment], errors='coerce')
        
        # Initialize results
        results = {
            'treatment': treatment,
            'outcome': outcome,
            'method': method.name.lower(),
            'timestamp': datetime.now().isoformat(),
            'confidence_level': confidence_level,
            'n_observations': len(df),
            'n_treated': df[treatment].sum()
        }
        
        # Method selection
        if method == EstimationMethod.AUTO:
            method = self._select_optimal_method(treatment)
            results['method'] = method.name.lower()
            results['method_selection_reason'] = self._get_method_selection_reason(method, treatment)
        
        try:
            # Initialize causal model
            model = CausalModel(
                data=df,
                treatment=treatment,
                outcome=outcome,
                graph=None,
                identify_vars=True
            )
            
            # Identify effect
            estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            # Effect estimation
            if method in [EstimationMethod.DML, EstimationMethod.DR, EstimationMethod.XLEARNER,
                        EstimationMethod.SLEARNER, EstimationMethod.TLEARNER]:
                ml_results = self._estimate_ml_effect(model, estimand, method, heterogeneity, confidence_level)
                results.update(ml_results)
            elif method == EstimationMethod.PSM:
                results.update(self._estimate_psm(model, estimand, confidence_level))
            else:  # SYNTH
                results.update(self._estimate_synth(treatment, outcome, confidence_level))
            
            # Sensitivity analysis
            if sensitivity:
                try:
                    results['sensitivity'] = self._run_sensitivity_analysis(model, estimand)
                except Exception as e:
                    logger.warning(f"Sensitivity analysis failed: {str(e)}")
                    results['sensitivity'] = {'error': str(e)}
                    
        except Exception as e:
            logger.error(f"Error in causal analysis: {str(e)}", exc_info=True)
            results['error'] = str(e)
        
        return results

    def _select_optimal_method(self, treatment: str) -> EstimationMethod:
        """Automatically select best estimation method based on data characteristics"""
        n_treated = self.event_data[self.event_data['event_type'] == treatment].shape[0]
        n_obs = len(self.price_data)
        
        if n_treated < 5:
            return EstimationMethod.SYNTH
        elif n_treated < 20:
            return EstimationMethod.PSM
        elif n_obs > 10000:
            return EstimationMethod.DML
        elif n_obs > 5000:
            return EstimationMethod.DR
        else:
            return EstimationMethod.XLEARNER

    def _get_method_selection_reason(self, method: EstimationMethod, treatment: str) -> str:
        """Get human-readable reason for method selection"""
        n_treated = self.event_data[self.event_data['event_type'] == treatment].shape[0]
        n_obs = len(self.price_data)
        
        reasons = {
            EstimationMethod.SYNTH: f"Few treated units ({n_treated} events)",
            EstimationMethod.PSM: f"Moderate treated units ({n_treated} events)",
            EstimationMethod.DML: f"Large dataset ({n_obs} observations)",
            EstimationMethod.DR: f"Medium dataset ({n_obs} observations)",
            EstimationMethod.XLEARNER: f"Small dataset with complex effects ({n_obs} observations)"
        }
        
        return reasons.get(method, "Automatic selection based on data characteristics")

    def _estimate_ml_effect(self, 
                        model: CausalModel,
                        estimand: Any,
                        method: EstimationMethod,
                        heterogeneity: bool,
                        confidence_level: float) -> Dict[str, Any]:
        """Estimate effects using ML-based methods with comprehensive type handling"""
        # Get the prepared data
        df = model._data
        treatment = model._treatment[0]
        outcome = model._outcome[0]
        
        # Convert to numpy arrays with explicit types
        y = np.array(df[outcome], dtype=np.float64)
        T = np.array(df[treatment], dtype=np.float64)
        X = df.drop([outcome, treatment], axis=1)
        X = X.select_dtypes(include=[np.number])  # Keep only numeric features
        X = np.array(X, dtype=np.float64)
        
        # Model configuration
        common_params = {
            'random_state': self.random_state,
            'max_iter': 100,
            'early_stopping': True
        }
        
        # Initialize models with proper types
        model_y = HistGradientBoostingRegressor(**common_params)
        model_t = HistGradientBoostingClassifier(**common_params)
        model_final = make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=5),
            WeightedLassoCV(cv=3, random_state=self.random_state)
        )
        
        # Time-series aware cross-validation
        cv = TimeSeriesSplit(n_splits=5)
        
        try:
            # Initialize appropriate estimator
            if method == EstimationMethod.DML:
                from econml.dml import LinearDML
                est = LinearDML(
                    model_y=model_y,
                    model_t=model_t,
                    discrete_treatment=True,
                    cv=cv,
                    random_state=self.random_state
                )
            elif method == EstimationMethod.DR:
                from econml.dr import DRLearner
                est = DRLearner(
                    model_regression=model_y,
                    model_propensity=model_t,
                    model_final=model_final,
                    cv=cv,
                    random_state=self.random_state
                )
            elif method == EstimationMethod.XLEARNER:
                from econml.metalearners import XLearner
                est = XLearner(
                    models=model_y,
                    propensity_model=model_t,
                    cate_models=model_final
                )
            elif method == EstimationMethod.SLEARNER:
                from econml.metalearners import SLearner
                est = SLearner(overall_model=model_y)
            else:  # TLearner
                from econml.metalearners import TLearner
                est = TLearner(models=model_y)
            
            # Fit the estimator
            est.fit(y, T, X=X)
            
            # Get effect estimates with proper type conversion
            ate = float(est.ate_)
            conf_int = [float(x) for x in est.ate_interval(alpha=1-confidence_level)]
            
            # Heterogeneous effects
            cates = None
            if heterogeneity:
                effect_modifiers = self._get_effect_modifiers()
                cates_values = est.const_marginal_effect(effect_modifiers.values)
                cates = pd.DataFrame(cates_values, index=effect_modifiers.index)
            
            return {
                'ate': ate,
                'conf_int': conf_int,
                'cate': cates.to_dict() if cates is not None else None,
                'model_diagnostics': {
                    'outcome_model_score': est.score(y, T, X=X),
                    'treatment_model_score': est.propensity_score_
                }
            }
            
        except Exception as e:
            logger.error(f"Error in ML effect estimation: {str(e)}", exc_info=True)
            return {
                'ate': np.nan,
                'conf_int': [np.nan, np.nan],
                'cate': None,
                'model_diagnostics': {'error': str(e)}
            }

    def _get_model_metrics(self, estimator) -> Dict[str, float]:
        """Extract comprehensive model performance metrics"""
        metrics = {
            'score': getattr(estimator, 'score_', np.nan)
        }
        
        try:
            if hasattr(estimator, 'models_y'):
                y_scores = [m.score_ for m in estimator.models_y if hasattr(m, 'score_')]
                metrics['outcome_model_score'] = np.mean(y_scores) if y_scores else np.nan
                
            if hasattr(estimator, 'models_t'):
                t_scores = [m.score_ for m in estimator.models_t if hasattr(m, 'score_')]
                metrics['treatment_model_score'] = np.mean(t_scores) if t_scores else np.nan
        except:
            pass
            
        return metrics

    def _get_feature_importance(self, estimator) -> Optional[Dict[str, float]]:
        """Extract feature importance if available"""
        try:
            if hasattr(estimator, 'feature_importances_'):
                return dict(zip(estimator.feature_names_in_, estimator.feature_importances_))
            
            if hasattr(estimator, 'model_final') and hasattr(estimator.model_final, 'coef_'):
                return dict(zip(estimator.model_final.feature_names_in_, estimator.model_final.coef_))
        except:
            return None

    def _estimate_psm(self, model: CausalModel, estimand: Any, confidence_level: float) -> Dict[str, Any]:
        """Propensity score matching estimation with enhanced configuration"""
        estimator = model.estimate_effect(
            estimand,
            method_name="backdoor.propensity_score_matching",
            target_units="ate",
            method_params={
                'confidence_level': confidence_level,
                'num_simulations': 1000
            }
        )
        
        return {
            'ate': estimator.value,
            'conf_int': estimator.get_confidence_intervals(alpha=1-confidence_level),
            'model_diagnostics': {
                'method': 'propensity_score_matching',
                'matched_pairs': estimator.summary['matched_pairs']
            }
        }

    def _estimate_synth(self, treatment: str, outcome: str, confidence_level: float) -> Dict[str, Any]:
        """Synthetic control estimation with time-series features"""
        from sklearn.linear_model import RidgeCV
        from sklearn.metrics import mean_squared_error
        
        df = self._prepare_causal_data(treatment)
        control_units = [col for col in df.columns 
                        if col not in [treatment, outcome] 
                        and not col.startswith('fourier_')]
        
        # Use Ridge regression for better stability
        model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
        model.fit(df[control_units], df[outcome])
        
        # Calculate effects
        predicted = model.predict(df[control_units])
        effect = df[outcome] - predicted
        ate = effect[df[treatment] == 1].mean()
        
        # Bootstrap confidence intervals
        rng = np.random.RandomState(self.random_state)
        bootstrap_ates = []
        for _ in range(1000):
            idx = rng.choice(len(effect), size=len(effect), replace=True)
            bootstrap_ates.append(effect.iloc[idx][df[treatment].iloc[idx] == 1].mean())
        
        ci_lower = np.percentile(bootstrap_ates, 100 * (1 - confidence_level) / 2)
        ci_upper = np.percentile(bootstrap_ates, 100 * (1 + confidence_level) / 2)
        
        return {
            'ate': ate,
            'conf_int': [ci_lower, ci_upper],
            'model_diagnostics': {
                'score': model.score(df[control_units], df[outcome]),
                'mse': mean_squared_error(df[outcome], predicted),
                'best_alpha': model.alpha_
            }
        }

    def _run_sensitivity_analysis(self, model: CausalModel, estimand: Any) -> Dict[str, Any]:
        """Run comprehensive sensitivity analysis"""
        from dowhy.sensitivity import SensitivityAnalyzer
        
        # Run multiple sensitivity analyses
        results = {}
        
        # R2 sensitivity
        r2_analyzer = SensitivityAnalyzer(
            model=model,
            estimand=estimand,
            sensitivity_type="R2",
            frac_strength_treatment=0.05
        )
        r2_analyzer.check_sensitivity()
        results['r2_sensitivity'] = r2_analyzer.sensitivity_results
        results['r2_threshold'] = r2_analyzer.strength_of_confounding()
        
        # Partial R2 sensitivity
        partial_analyzer = SensitivityAnalyzer(
            model=model,
            estimand=estimand,
            sensitivity_type="partial-R2",
            frac_strength_treatment=0.05
        )
        partial_analyzer.check_sensitivity()
        results['partial_r2_sensitivity'] = partial_analyzer.sensitivity_results
        results['partial_r2_threshold'] = partial_analyzer.strength_of_confounding()
        
        return results

    def _get_effect_modifiers(self) -> pd.DataFrame:
        """Features for heterogeneous effects estimation with time awareness"""
        modifiers = self.price_data[[
            'USD_Index', 'Global_Demand', 'Inventories',
            'Geopolitical_Risk', 'Economic_Growth',
            'volatility_30d', 'rsi'
        ]].copy()
        
        # Add lagged features
        for col in ['USD_Index', 'Global_Demand']:
            modifiers[f'{col}_lag1'] = modifiers[col].shift(1).fillna(method='bfill')
        
        return modifiers.dropna()

    def run_parallel_analyses(self,
                            treatments: List[str],
                            outcome: str = 'Price',
                            method: str = 'auto') -> Dict[str, Dict]:
        """
        Run analyses for multiple treatments in parallel
        
        Args:
            treatments: List of treatment variables to analyze
            outcome: Outcome variable
            method: Estimation method
            
        Returns:
            Dictionary mapping treatments to their results
        """
        if not treatments:
            return {}
            
        # Validate all treatments first
        valid_treatments = []
        for treatment in treatments:
            try:
                valid_treatments.append(self._validate_treatment(treatment))
            except ValueError as e:
                logger.warning(f"Skipping invalid treatment {treatment}: {str(e)}")
        
        if not valid_treatments:
            raise ValueError("No valid treatments provided")
        
        # Run in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            func = partial(self.estimate_effect, outcome=outcome, method=method)
            results = list(tqdm(
                executor.map(func, valid_treatments),
                total=len(valid_treatments),
                disable=not self.progress_bar
            ))
        
        return dict(zip(valid_treatments, results))

    def _setup_reporting(self):
        """Create reporting directory structure"""
        self.REPORT_PATH.mkdir(parents=True, exist_ok=True)
        self.figure_path = self.REPORT_PATH / "figures"
        self.figure_path.mkdir(exist_ok=True)
        
    def _validate_and_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced data validation with automatic repair"""
        # [Previous validation code]
        # Add advanced preprocessing
        df = df.asfreq('D').interpolate()  # Ensure daily frequency
        return df
    
    def get_event_types(self) -> List[str]:
        """Enhanced event type detection with frequency analysis"""
        events = self.event_data['event_type'].value_counts().to_dict()
        return sorted(events.keys(), key=lambda x: -events[x])  # Sort by frequency
    
    def plot_causal_graph(self) -> plt.Figure:
        """Visualize the causal graph with enhanced styling"""
        plt.figure(figsize=(12, 8))
        
        # Use spring layout for better node positioning
        pos = nx.spring_layout(self.causal_graph, seed=self.random_state)
        
        # Draw nodes with different colors for treatments and outcomes
        node_colors = []
        for node in self.causal_graph.nodes():
            if node == 'Price':
                node_colors.append('lightgreen')
            elif node in self.event_data['event_type'].unique():
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightblue')
        
        nx.draw_networkx_nodes(
            self.causal_graph, pos,
            node_size=1500,
            node_color=node_colors,
            alpha=0.9
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.causal_graph, pos,
            width=1.5,
            alpha=0.6,
            edge_color='gray'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.causal_graph, pos,
            font_size=10,
            font_weight='bold'
        )
        
        plt.title("Causal Graph Structure", fontsize=14, pad=20)
        plt.axis('off')
        plt.tight_layout()
        return plt.gcf()
    
    def save_results(self, results: Dict[str, Any], filename: str) -> Path:
        """Save analysis results to JSON file with proper serialization"""
        report_path = self.REPORT_PATH / filename
        
        class NumpyEncoder(json.JSONEncoder):
            """Custom JSON encoder for numpy types"""
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                return super().default(obj)
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Results saved to {report_path}")
        return report_path
    
    def _create_advanced_plots(self, treatment: str, results: dict) -> Dict[str, Path]:
        """Generate publication-quality visualizations"""
        plot_paths = {}
        
        # 1. Event Study Plot (Enhanced)
        fig = self._plot_enhanced_event_study(treatment)
        path = self.figure_path / f"{treatment}_event_study.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plot_paths['event_study'] = path
        plt.close(fig)
        
        # 2. Causal Impact Distribution
        fig = self._plot_impact_distribution(treatment, results)
        path = self.figure_path / f"{treatment}_impact_dist.png"
        fig.savefig(path, dpi=300)
        plot_paths['impact_dist'] = path
        plt.close(fig)
        
        # 3. Causal Graph (Interactive HTML)
        self._save_interactive_graph(treatment)
        plot_paths['causal_graph'] = self.figure_path / f"{treatment}_causal_graph.html"
        
        return plot_paths
    
    def _plot_enhanced_event_study(self, treatment: str) -> plt.Figure:
        """Professional-grade event study plot"""
        data = self._prepare_event_study_data(treatment)
        fig = plt.figure(figsize=(14, 8), dpi=100)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        
        # Main price plot
        ax1 = plt.subplot(gs[0])
        sns.lineplot(x=data['days'], y=data['mean_price'], 
                    ax=ax1, color='navy', linewidth=2)
        ax1.fill_between(data['days'],
                        data['mean_price'] - data['std_price'],
                        data['mean_price'] + data['std_price'],
                        alpha=0.2, color='navy')
        ax1.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax1.set_title(f'Event Study: {treatment}', fontsize=14, pad=20)
        ax1.set_ylabel('Price ($)', fontsize=12)
        
        # Volume/volatility subplot
        ax2 = plt.subplot(gs[1])
        if 'Volume' in self.price_data.columns:
            sns.lineplot(x=data['days'], 
                        y=self.price_data['Volume'].rolling(7).mean(),
                        ax=ax2, color='green', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax2.set_xlabel('Days Relative to Event', fontsize=12)
        ax2.set_ylabel('Volume', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def _plot_impact_distribution(self, treatment: str, results: dict) -> plt.Figure:
        """Plot distribution of treatment effects"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract effect sizes from results
        effect_sizes = []
        if 'basic_analysis' in results:
            effect_sizes.append(results['basic_analysis']['pre_post']['30']['effect_size'])
        if 'causal_analysis' in results:
            effect_sizes.append(results['causal_analysis']['ate'])
        
        if effect_sizes:
            sns.kdeplot(effect_sizes, ax=ax, fill=True, color='blue', alpha=0.3)
            for i, effect in enumerate(effect_sizes):
                ax.axvline(effect, color='red', linestyle='--', 
                          label=f'Effect {i+1}: {effect:.2f}')
            
            ax.set_title(f'Treatment Effect Distribution: {treatment}', fontsize=14)
            ax.set_xlabel('Effect Size')
            ax.set_ylabel('Density')
            ax.legend()
        
        return fig
    
    def _save_interactive_graph(self, treatment: str):
        """Save interactive graph visualization using pyvis"""
        try:
            from pyvis.network import Network
            net = Network(notebook=False, height="750px", width="100%")
            net.from_nx(self.causal_graph)
            
            # Customize node appearance
            for node in net.nodes:
                if node['id'] == treatment:
                    node['color'] = '#FF0000'
                    node['size'] = 25
                elif node['id'] == 'Price':
                    node['color'] = '#00FF00'
                    node['size'] = 30
            
            path = str(self.figure_path / f"{treatment}_causal_graph.html")
            net.show(path)
        except ImportError:
            logger.warning("Pyvis not available - skipping interactive graph")
    
    def save_analysis_report(self, treatment: str, results: dict):
        """Save comprehensive analysis report with visualizations"""
        report = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'treatment': treatment,
                'n_observations': len(self.price_data),
                'n_events': len(self.event_data[self.event_data['event_type'] == treatment])
            },
            'results': results,
            'visualizations': {
                str(p): "generated" 
                for p in self._create_advanced_plots(treatment, results).values()
            }
        }
        
        report_path = self.REPORT_PATH / f"{treatment}_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=self.NumpyEncoder)
            
        logger.info(f"Saved comprehensive report to {report_path}")
        
        # Generate PDF summary
        self._generate_pdf_summary(treatment, report_path)
        
        return report_path
    
    def _generate_pdf_summary(self, treatment: str, json_path: Path):
        """Generate PDF version of the report"""
        try:
            from fpdf import FPDF
            from PIL import Image
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Add title
            pdf.cell(200, 10, txt=f"Causal Analysis Report: {treatment}",
                    ln=1, align='C')
            
            # Add metadata
            pdf.cell(200, 10, txt=f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}",
                    ln=1)
            
            # Add key results
            with open(json_path) as f:
                data = json.load(f)
                
            pdf.cell(200, 10, txt="Key Results:", ln=1)
            for k, v in data['results'].items():
                if isinstance(v, (int, float, str)):
                    pdf.cell(200, 10, txt=f"{k}: {v}", ln=1)
            
            # Add figures
            for fig_type, fig_path in data['visualizations'].items():
                if fig_path != "generated" and Path(fig_path).exists():
                    if fig_path.endswith('.png'):
                        img = Image.open(fig_path)
                        pdf.image(fig_path, x=10, w=180)
                        pdf.cell(200, 10, txt="", ln=1)  # Spacer
            
            pdf_path = json_path.with_suffix('.pdf')
            pdf.output(str(pdf_path))
            logger.info(f"Generated PDF report at {pdf_path}")
            
        except ImportError:
            logger.warning("PDF generation dependencies not available")
    
    class NumpyEncoder(json.JSONEncoder):
        """Enhanced JSON encoder for scientific results"""
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating, np.ndarray)):
                return super().default(obj)
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif hasattr(obj, 'to_dict'):  # For pandas DataFrames/Series
                return obj.to_dict()
            return json.JSONEncoder.default(self, obj)

def generate_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sample price and event data for demonstration"""
    # Generate price data
    date_range = pd.date_range('2020-01-01', '2023-12-31')
    n_days = len(date_range)
    
    # Base price with trend and seasonality
    t = np.arange(n_days)
    base_price = 60 + 0.02 * t + 5 * np.sin(2 * np.pi * t / 365)
    
    # Add noise
    prices = base_price + np.random.normal(0, 2, n_days)
    
    # Create DataFrame
    price_data = pd.DataFrame(
        {'Price': prices},
        index=date_range
    )
    
    # Generate sample events
    events = [
        {'event_type': 'OPEC_Decisions', 'start_date': '2020-04-10', 'end_date': '2020-04-20'},
        {'event_type': 'OPEC_Decisions', 'start_date': '2021-01-05', 'end_date': '2021-01-15'},
        {'event_type': 'Geopolitical_Events', 'start_date': '2020-09-01', 'end_date': '2020-09-30'},
        {'event_type': 'Economic_Releases', 'start_date': '2022-03-15', 'end_date': '2022-03-20'},
    ]
    
    event_data = pd.DataFrame(events)
    event_data['start_date'] = pd.to_datetime(event_data['start_date'])
    event_data['end_date'] = pd.to_datetime(event_data['end_date'])
    
    return price_data, event_data

def main():
    """Main execution with robust error handling"""
    try:
        logger.info("Starting causal analysis")
        
        # Generate sample data
        price_data, event_data = generate_sample_data()
        
        # Initialize analyzer
        analyzer = CausalImpactAnalyzer(
            price_data=price_data,
            event_data=event_data,
            n_jobs=4,
            random_state=42
        )
        
        # Get available event types
        event_types = analyzer.get_event_types()
        logger.info(f"Available event types: {event_types}")
        
        results = {}
        for event_type in event_types:
            try:
                logger.info(f"Analyzing {event_type}...")
                
                # Basic analysis
                basic_results = analyzer.run_basic_analysis(event_type)
                
                # Advanced causal estimation
                causal_results = analyzer.estimate_effect(
                    treatment=event_type,
                    method='auto',
                    heterogeneity=True,
                    sensitivity=False  # Disable until we fix the package
                )
                
                results[event_type] = {
                    'basic_analysis': basic_results,
                    'causal_analysis': causal_results
                }
                
                # Save individual report
                analyzer.save_analysis_report(event_type, results[event_type])
                
            except Exception as e:
                logger.error(f"Failed to analyze {event_type}: {str(e)}")
                continue
        
        # Save combined results
        analyzer.save_results(results, 'causal_analysis_results.json')
        
        # Plot causal graph
        fig = analyzer.plot_causal_graph()
        fig.savefig('causal_graph.png')
        plt.close(fig)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in analysis: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()