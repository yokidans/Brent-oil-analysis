# src/utils/robust_sampling.py
import numpy as np
import pymc as pm  # Note: Changed from pymc3 to pymc
import arviz as az

class RobustSampler:
    """Modern PyMC v5 implementation (Windows-compatible)"""
    
    def __init__(self):
        self.trace = None
        
    def sample_normal(self, observed_data=None):
        """Robust sampling using PyMC v5"""
        if observed_data is None:
            observed_data = np.random.normal(loc=1.5, scale=0.5, size=200)
            
        with pm.Model() as model:
            # More stable priors
            mu = pm.Normal('mu', mu=0, sigma=10)
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Likelihood
            pm.Normal('obs', 
                     mu=mu, 
                     sigma=sigma, 
                     observed=observed_data)
            
            # Sampling with modern settings
            self.trace = pm.sample(
                draws=1000,
                tune=1000,
                cores=1,
                progressbar=False  # Reduces console output
            )
            
        return self
    
    def get_summary(self):
        """Get sampling statistics"""
        if self.trace is None:
            raise ValueError("Run sampling first")
        return az.summary(self.trace)

if __name__ == "__main__":
    # Test
    sampler = RobustSampler()
    sampler.sample_normal()
    print(sampler.get_summary())