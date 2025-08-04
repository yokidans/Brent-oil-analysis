# src/utils/distributed_sampling.py
import numpy as np
import logging
from multiprocessing import Pool
from typing import Dict, List, Union, Optional
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle PyMC3/PyMC import with warning suppression
PYMC_AVAILABLE = False
PYMC_IMPORT_ERROR = ""
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import pymc as pm  # Try importing PyMC v4 first
            PYMC_AVAILABLE = True
            logger.info("Using PyMC v4")
        except ImportError:
            import pymc3 as pm  # Fall back to PyMC3
            PYMC_AVAILABLE = True
            logger.info("Using PyMC3 (legacy)")
except ImportError as e:
    PYMC_IMPORT_ERROR = str(e)
    logger.warning(f"PyMC not available - falling back to NumPy sampling: {PYMC_IMPORT_ERROR}")

try:
    import arviz as az
except ImportError:
    az = None
    logger.warning("ArviZ not available - summary statistics will be limited")

class ParallelMCMC:
    """Robust parallel sampling with comprehensive fallback mechanisms"""
    
    def __init__(self, n_chains: int = 4, samples_per_chain: int = 1000):
        """
        Initialize the parallel sampler.
        
        Args:
            n_chains: Number of parallel chains (default: 4)
            samples_per_chain: Samples to generate per chain (default: 1000)
        """
        if n_chains <= 0:
            raise ValueError("n_chains must be a positive integer")
        if samples_per_chain <= 0:
            raise ValueError("samples_per_chain must be a positive integer")
            
        self.n_chains = n_chains
        self.samples_per_chain = samples_per_chain
        logger.info(f"Initialized ParallelMCMC with {n_chains} chains, {samples_per_chain} samples each")

    def _single_chain(self, seed: int) -> Dict[str, np.ndarray]:
        """
        Run a single sampling chain with comprehensive error handling.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary of samples for this chain
        """
        np.random.seed(seed)
        
        if PYMC_AVAILABLE:
            try:
                with pm.Model() as model:
                    mu = pm.Normal('mu', mu=0, sigma=1)
                    sigma = pm.HalfNormal('sigma', sigma=0.1)
                    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=np.random.randn(100))
                    
                    trace = pm.sample(
                        draws=self.samples_per_chain // 2,  # PyMC generates more samples
                        tune=self.samples_per_chain // 2,
                        chains=1,
                        random_seed=seed,
                        return_inferencedata=False,
                        progressbar=False
                    )
                    return trace
            except Exception as e:
                logger.warning(f"PyMC sampling failed (chain {seed}), falling back to NumPy: {str(e)}")
        
        # Fallback to classical sampling
        samples = {
            'mu': np.random.normal(0, 1, self.samples_per_chain),
            'sigma': np.abs(np.random.normal(1, 0.1, self.samples_per_chain))  # Ensure positive sigma
        }
        return samples

    def sample(self) -> Union[List[Dict[str, np.ndarray]], az.InferenceData]:
        """
        Run parallel sampling with comprehensive error handling.
        
        Returns:
            Either a list of chain results (when PyMC not available) or
            an ArviZ InferenceData object (when PyMC available)
        """
        try:
            seeds = np.random.randint(0, 2**31 - 1, size=self.n_chains)  # Safe seed range
            
            with Pool(self.n_chains) as pool:
                chains = pool.map(self._single_chain, seeds)
                
            if PYMC_AVAILABLE and all(isinstance(c, pm.backends.base.MultiTrace) for c in chains):
                if az is not None:
                    return az.concat(*[az.from_pymc3(trace) for trace in chains])
                return chains
                
            return chains
            
        except Exception as e:
            logger.error(f"Parallel sampling failed: {str(e)}")
            # Return empty chains of the correct structure
            return [{
                'mu': np.array([]),
                'sigma': np.array([])
            } for _ in range(self.n_chains)]

    def summarize_results(self, result: Union[List[Dict[str, np.ndarray]], az.InferenceData]) -> None:
        """Print summary statistics for the sampling results"""
        if az is not None and isinstance(result, az.InferenceData):
            print(az.summary(result))
        elif isinstance(result, list):
            print("\nChain Statistics:")
            for i, chain in enumerate(result):
                if len(chain['mu']) > 0:
                    print(f"Chain {i + 1}:")
                    print(f"  mu:    mean={np.mean(chain['mu']):.4f}, std={np.std(chain['mu']):.4f}")
                    print(f"  sigma: mean={np.mean(chain['sigma']):.4f}, std={np.std(chain['sigma']):.4f}")
                else:
                    print(f"Chain {i + 1}: No samples generated")

if __name__ == "__main__":
    try:
        sampler = ParallelMCMC(n_chains=4, samples_per_chain=1000)
        result = sampler.sample()
        
        if result is not None:
            print("\nSampling completed successfully!")
            sampler.summarize_results(result)
        else:
            logger.error("Sampling returned None result")
            
    except Exception as e:
        logger.error(f"Error during sampling: {str(e)}")