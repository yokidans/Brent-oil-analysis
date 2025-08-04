# run_analysis.ps1
param (
    [string]$DataPath = "data/processed/cleaned_prices.parquet",
    [string]$OutputPath = "reports/change_points.png",
    [int]$Epochs = 200,
    [float]$LearningRate = 0.05
)

try {
    # Create the Python analysis script
    $pythonScript = @"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages

import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

tfd = tfp.distributions
tfb = tfp.bijectors

class EffectiveChangePointModel:
    def __init__(self, n_changepoints: int = 3):
        self.n_changepoints = n_changepoints
        
        # Raw variational parameters
        self.raw_tau = tf.Variable(
            tf.linspace(0.2, 0.8, n_changepoints),  # Relative positions (0-1)
            name='raw_tau')
        self.raw_mu = tf.Variable(
            tf.zeros(n_changepoints+1),
            name='raw_mu')
        self.raw_sigma = tf.Variable(
            tf.ones(n_changepoints+1),
            name='raw_sigma')
        
        # Bijectors for constraints
        self.tau_bijector = tfb.Sigmoid()  # Constrain between 0 and 1
        self.sigma_bijector = tfb.Exp()

    def prepare_data(self, data):
        return tf.convert_to_tensor(data, dtype=tf.float32)

    @property
    def q_params(self):
        """Get constrained variational parameters"""
        return {
            'tau': tf.sort(self.tau_bijector(self.raw_tau)),  # Sorted relative positions
            'mu': self.raw_mu,
            'sigma': self.sigma_bijector(self.raw_sigma)
        }

    def fit(self, data, epochs=200, learning_rate=0.05):
        X = self.prepare_data(data)
        n = len(data)
        optimizer = tf.optimizers.Adam(learning_rate)
        
        @tf.function
        def loss_fn():
            params = self.q_params
            relative_tau = params['tau']
            absolute_tau = relative_tau * tf.cast(n-1, tf.float32)  # Scale to data length
            mu = params['mu']
            sigma = params['sigma']
            
            # Create segment indicators using sigmoid transitions
            time_points = tf.range(n, dtype=tf.float32)
            segment_probs = []
            prev_cutpoint = 0.0
            for cutpoint in tf.unstack(absolute_tau):
                segment_prob = tf.sigmoid(10.0 * (cutpoint - time_points))
                segment_probs.append(segment_prob - prev_cutpoint)
                prev_cutpoint = segment_prob
            segment_probs.append(1.0 - prev_cutpoint)
            
            # Compute weighted log probabilities
            total_loss = 0.0
            for i, prob in enumerate(segment_probs):
                dist = tfd.Normal(mu[i], sigma[i])
                log_prob = tf.clip_by_value(dist.log_prob(X), -10, 10)
                total_loss += tf.reduce_sum(prob * log_prob)
            
            # Regularization
            reg_loss = 0.1 * (tf.reduce_sum(tf.abs(mu)) + tf.reduce_sum(tf.abs(sigma)))
            
            return -(total_loss / tf.cast(n, tf.float32) - reg_loss)
        
        # Training loop with gradient clipping
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss = loss_fn()
            
            grads = tape.gradient(loss, [self.raw_tau, self.raw_mu, self.raw_sigma])
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, [self.raw_tau, self.raw_mu, self.raw_sigma]))
            
            if epoch % 20 == 0:
                params = self.q_params
                abs_tau = (params['tau'] * (n-1)).numpy()
                print(f"Epoch {epoch}: Loss = {loss.numpy():.4f}, Tau = {abs_tau}")
        
        return self

    def detect_changepoints(self, data):
        params = self.q_params
        abs_tau = (params['tau'] * (len(data)-1)).numpy()
        return {
            'changepoints': abs_tau,
            'uncertainty': 0.1 * np.ones_like(abs_tau),
            'parameters': {
                'mu': params['mu'].numpy(),
                'sigma': params['sigma'].numpy()
            }
        }

# Main execution
try:
    print("Loading data...")
    df = pd.read_parquet(r'$DataPath')
    returns = df['Log_Return'].values
    
    print("Initializing model...")
    model = EffectiveChangePointModel(n_changepoints=3)
    
    print(f"Training model for {$Epochs} epochs...")
    model.fit(returns, epochs=$Epochs, learning_rate=$LearningRate)
    
    print("Detecting change points...")
    results = model.detect_changepoints(returns)
    
    print("Plotting results...")
    plt.figure(figsize=(12,6))
    plt.plot(df.index, returns, label='Log Returns')
    for cp in results['changepoints']:
        plt.axvline(df.index[int(cp)], color='r', linestyle='--', alpha=0.7)
    plt.title('Detected Change Points in Brent Oil Log Returns')
    plt.legend()
    
    os.makedirs(os.path.dirname(r'$OutputPath'), exist_ok=True)
    plt.savefig(r'$OutputPath')
    plt.close()
    print(f"Analysis complete. Results saved to {r'$OutputPath'}")

except Exception as e:
    print(f"Error: {str(e)}")
    raise
"@

    # Save and run the script
    $tempScript = "temp_analysis.py"
    $pythonScript | Out-File $tempScript -Encoding utf8
    python $tempScript
    Remove-Item $tempScript
}
catch {
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}