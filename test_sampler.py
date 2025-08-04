# test_sampler.py
from src.utils.robust_sampling import RobustSampler
import numpy as np

# Create test data
data = np.random.normal(loc=1.5, scale=0.5, size=200)

# Run analysis
sampler = RobustSampler()
sampler.sample_normal(data)
results = sampler.get_summary()

print('\nSuccessful run! Key results:')
print(f"Estimated mean: {results['mean']['mu']:.2f} (true: 1.50)")
print(f"Estimated std: {results['mean']['sigma']:.2f} (true: 0.50)")