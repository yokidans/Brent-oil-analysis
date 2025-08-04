# src/analysis/change_point.py
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

class BayesianChangePoint:
    """
    Bayesian Change Point Detection for Brent Oil Prices
    
    Implements a model to detect structural breaks in the time series
    using PyMC3 for Bayesian inference.
    """
    
    def __init__(self, data):
        """
        Initialize with time series data
        
        Args:
            data (pd.Series): Time series data (log returns recommended)
        """
        self.data = data
        self.model = None
        self.trace = None
        
    def build_model(self, n_changepoints=1):
        """
        Build the Bayesian change point model
        
        Args:
            n_changepoints (int): Number of change points to detect
        """
        with pm.Model() as self.model:
            # Priors for the change point locations
            tau = pm.DiscreteUniform(
                "tau", 
                lower=0, 
                upper=len(self.data)-1, 
                shape=n_changepoints
            )
            
            # Priors for the means before and after change points
            mu = pm.Normal("mu", mu=0, sigma=1, shape=n_changepoints+1)
            
            # Priors for the standard deviations
            sigma = pm.HalfNormal("sigma", sigma=1, shape=n_changepoints+1)
            
            # Model the mean and sigma as switching at the change points
            mean = mu[0] * np.ones(len(self.data))
            std = sigma[0] * np.ones(len(self.data))
            
            for i in range(n_changepoints):
                mean = pm.math.switch(tau[i] < np.arange(len(self.data)), 
                                    mean, 
                                    mu[i+1] * np.ones(len(self.data)))
                std = pm.math.switch(tau[i] < np.arange(len(self.data)), 
                                   std, 
                                   sigma[i+1] * np.ones(len(self.data)))
            
            # Likelihood
            likelihood = pm.Normal(
                "likelihood", 
                mu=mean, 
                sigma=std, 
                observed=self.data.values
            )
    
    def fit(self, samples=3000, tune=1000, chains=2):
        """
        Fit the model using MCMC sampling
        
        Args:
            samples (int): Number of samples to draw
            tune (int): Number of tuning samples
            chains (int): Number of chains to run
        """
        with self.model:
            self.trace = pm.sample(
                samples, 
                tune=tune, 
                chains=chains, 
                return_inferencedata=True
            )
    
    def diagnose(self):
        """Check model convergence and performance"""
        if self.trace is None:
            raise ValueError("Model has not been fitted yet")
            
        # Summary statistics
        print(az.summary(self.trace))
        
        # Trace plots
        az.plot_trace(self.trace, compact=True)
        plt.tight_layout()
        plt.show()
        
    def plot_changepoints(self):
        """Plot the detected change points"""
        if self.trace is None:
            raise ValueError("Model has not been fitted yet")
            
        # Plot the time series
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data, label='Log Returns')
        
        # Plot change points
        tau_samples = self.trace.posterior['tau'].values.flatten()
        for tau in np.unique(tau_samples):
            cp_date = self.data.index[int(np.median(tau))]
            plt.axvline(cp_date, color='red', linestyle='--', alpha=0.5)
            
        plt.title('Detected Change Points in Brent Oil Log Returns')
        plt.xlabel('Date')
        plt.ylabel('Log Return')
        plt.legend()
        plt.grid()
        plt.show()
        
    def get_changepoint_dates(self):
        """Get the most probable change point dates"""
        if self.trace is None:
            raise ValueError("Model has not been fitted yet")
            
        tau_samples = self.trace.posterior['tau'].values.flatten()
        unique_taus, counts = np.unique(tau_samples, return_counts=True)
        sorted_indices = np.argsort(-counts)
        
        changepoints = []
        for idx in sorted_indices[:min(5, len(unique_taus))]:  # Get top 5
            date = self.data.index[int(unique_taus[idx])]
            probability = counts[idx] / len(tau_samples)
            changepoints.append({
                'date': date,
                'probability': probability,
                'index': int(unique_taus[idx])
            })
            
        return changepoints
    def analyze_changepoints(price_data, event_data):
        """
        Complete change point analysis pipeline
        
        Args:
            price_data (pd.DataFrame): Processed price data with log returns
            event_data (pd.DataFrame): Key events with dates
            
        Returns:
            dict: Analysis results including change points and event correlations
        """
        # Initialize and fit the model
        cp_model = BayesianChangePoint(price_data['Log_Return'])
        cp_model.build_model(n_changepoints=3)  # Look for 3 change points
        cp_model.fit()
        
        # Check model diagnostics
        cp_model.diagnose()
        
        # Get detected change points
        changepoints = cp_model.get_changepoint_dates()
        
        # Find nearest events to change points
        results = []
        for cp in changepoints:
            # Find events within 30 days of the change point
            time_delta = pd.Timedelta(days=30)
            nearby_events = event_data[
                (event_data.index >= cp['date'] - time_delta) & 
                (event_data.index <= cp['date'] + time_delta)
            ]
            
            # Calculate price changes around the change point
            window = 30  # days
            pre_period = price_data.loc[
                (price_data.index >= cp['date'] - pd.Timedelta(days=window)) & 
                (price_data.index < cp['date'])
            ]
            post_period = price_data.loc[
                (price_data.index >= cp['date']) & 
                (price_data.index <= cp['date'] + pd.Timedelta(days=window))
            ]
            
            mean_pre = np.exp(pre_period['Log_Return'].mean()) - 1
            mean_post = np.exp(post_period['Log_Return'].mean()) - 1
            pct_change = (mean_post - mean_pre) / mean_pre * 100 if mean_pre != 0 else 0
            
            results.append({
                'change_point_date': cp['date'],
                'probability': cp['probability'],
                'nearby_events': nearby_events.to_dict('records'),
                'price_change_pct': pct_change,
                'mean_pre': mean_pre,
                'mean_post': mean_post,
                'window_days': window
            })
        
        return {
            'changepoints': results,
            'model_diagnostics': az.summary(cp_model.trace).to_dict()
        }
    
    

class AdvancedChangePointModels:
    """
    Proposed advanced models for future work
    """
    
    @staticmethod
    def markov_switching_model(data):
        """
        Proposed Markov Switching Model for regime detection
        
        Args:
            data (pd.Series): Time series data
            
        Returns:
            dict: Model specification for future implementation
        """
        return {
            "model_type": "Markov Switching",
            "description": "Model that explicitly defines different market regimes",
            "parameters": {
                "k_regimes": 2,  # Calm and volatile regimes
                "mean_params": "Varies by regime",
                "volatility_params": "Varies by regime",
                "transition_matrix": "Probabilities of switching between regimes"
            }
        }
    
    @staticmethod
    def var_model(data, exogenous_vars):
        """
        Proposed Vector Autoregression Model for multivariate analysis
        
        Args:
            data (pd.DataFrame): Time series data
            exogenous_vars (list): List of exogenous variables to include
            
        Returns:
            dict: Model specification for future implementation
        """
        return {
            "model_type": "Vector Autoregression",
            "description": "Model dynamic relationships between oil prices and other variables",
            "proposed_variables": exogenous_vars + ["Brent_Price"],
            "lag_order": "To be determined by information criteria"
        }