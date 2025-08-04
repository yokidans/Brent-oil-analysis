# src/utils/classical_sampling.py
import numpy as np
from typing import Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassicalSampler:
    """Pure NumPy implementation for maximum compatibility with parameter validation"""
    
    def __init__(self, n_samples: int = 1000):
        """
        Initialize the classical sampler.
        
        Args:
            n_samples: Number of samples to generate (default: 1000)
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        self.n_samples = n_samples
        logger.info(f"Initialized ClassicalSampler with n_samples={n_samples}")
    
    def sample_normal(self, params: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Sample from normal distributions with parameter validation.
        
        Args:
            params: Dictionary containing:
                - mu_mean: Mean for mu distribution
                - mu_std: Standard deviation for mu distribution (must be > 0)
                - sigma_mean: Mean for sigma distribution
                - sigma_std: Standard deviation for sigma distribution (must be > 0)
        
        Returns:
            Dictionary with keys 'mu' and 'sigma' containing sampled values
        """
        # Validate parameters
        required_params = ['mu_mean', 'mu_std', 'sigma_mean', 'sigma_std']
        if not all(key in params for key in required_params):
            raise ValueError(f"Missing required parameters. Needed: {required_params}")
        
        if params['mu_std'] <= 0 or params['sigma_std'] <= 0:
            raise ValueError("Standard deviation parameters must be positive")
        
        # Generate samples
        samples = {
            'mu': np.random.normal(params['mu_mean'], params['mu_std'], self.n_samples),
            'sigma': np.random.normal(params['sigma_mean'], params['sigma_std'], self.n_samples)
        }
        
        logger.info(f"Generated {self.n_samples} samples for mu and sigma")
        return samples

    def get_sample_stats(self, samples: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
        """
        Calculate basic statistics for generated samples.
        
        Args:
            samples: Dictionary containing 'mu' and 'sigma' arrays
            
        Returns:
            Dictionary with mean and std for each parameter
        """
        return {
            'mu': (np.mean(samples['mu']), np.std(samples['mu'])),
            'sigma': (np.mean(samples['sigma']), np.std(samples['sigma']))
        }

if __name__ == "__main__":
    # Example usage
    try:
        sampler = ClassicalSampler(n_samples=2000)
        params = {
            'mu_mean': 0,
            'mu_std': 1,
            'sigma_mean': 1,
            'sigma_std': 0.1
        }
        samples = sampler.sample_normal(params)
        stats = sampler.get_sample_stats(samples)
        
        print("\nSample Statistics:")
        for param, (mean, std) in stats.items():
            print(f"{param}: mean={mean:.4f}, std={std:.4f}")
            
    except Exception as e:
        logger.error(f"Error in sampling: {str(e)}")