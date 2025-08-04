# src/analysis/deep_change_point.py
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tfd = tfp.distributions
tfb = tfp.bijectors

class NeuralChangePointModel:
    """
    Hybrid neural network and Bayesian change point detection model
    with fixed shape handling for MarkovChain distribution.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 n_changepoints: int = 3,
                 latent_dim: int = 8):
        if len(input_shape) != 2:
            raise ValueError("input_shape must be a tuple of (window_size, n_features)")
            
        self.input_shape = input_shape
        self.n_changepoints = n_changepoints
        self.latent_dim = latent_dim
        
        # Build components
        self.feature_extractor = self._build_feature_extractor()
        self.model = self._build_joint_distribution()
        
        logger.info(f"Model initialized with input_shape={input_shape}, "
                   f"n_changepoints={n_changepoints}, latent_dim={latent_dim}")
    
    def _build_feature_extractor(self) -> tf.keras.Model:
        """Simplified feature extractor for stability"""
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        return tf.keras.Model(inputs=inputs, outputs=x, name="feature_extractor")
    
    def _build_joint_distribution(self) -> tfd.JointDistributionNamed:
        """Build the joint distribution with proper shape handling"""
        def changepoint_model():
            # Latent features
            features = yield tfd.JointDistributionCoroutine.Root(
                tfd.Sample(
                    tfd.Normal(loc=0., scale=1.),
                    sample_shape=self.latent_dim
                )
            )
            
            # Reshape features for the extractor
            features_reshaped = tf.reshape(features, [1, self.latent_dim, 1])
            features_reshaped = tf.tile(features_reshaped, [1, self.input_shape[0], 1])
            
            # Get transformed features
            features_transformed = self.feature_extractor(features_reshaped)
            
            # Changepoint locations with proper shape handling
            tau = yield tfd.JointDistributionCoroutine.Root(
                tfd.MarkovChain(
                    initial_state_prior=tfd.Categorical(
                        logits=tf.ones([self.n_changepoints])  # Uniform prior
                    ),
                    transition_fn=lambda t, s: tfd.Categorical(
                        probs=tf.ones([self.n_changepoints]) * 0.5  # Uniform transitions
                    ),
                    num_steps=self.input_shape[0]  # Match window size
                )
            )
            
            # Regime parameters
            mu = yield tfd.Sample(
                tfd.Normal(loc=0., scale=1.), 
                sample_shape=self.n_changepoints+1
            )
            
            sigma = yield tfd.Sample(
                tfd.HalfNormal(scale=1.),
                sample_shape=self.n_changepoints+1
            )
            
            return tau, mu, sigma
            
        return tfd.JointDistributionCoroutine(changepoint_model)
    
    def _prepare_data(self, data: np.ndarray) -> tf.Tensor:
        """Convert raw data to properly shaped sliding windows"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        window_size = self.input_shape[0]
        n_features = self.input_shape[1]
        
        # Create sliding windows
        X = np.array([
            data[i:i+window_size] 
            for i in range(len(data)-window_size)
        ])
        
        # Ensure correct shape
        if X.shape[-1] != n_features:
            X = X.reshape(*X.shape[:-1], n_features)
            
        return tf.convert_to_tensor(X, dtype=tf.float32)
    
    def fit(self, 
            data: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            verbose: bool = True):
        """Fit the model with simplified variational inference"""
        X = self._prepare_data(data)
        
        # Simplified surrogate posterior
        surrogate_posterior = tfd.JointDistributionNamed({
            'tau': tfd.Categorical(probs=tf.ones([self.input_shape[0]])/self.input_shape[0]),
            'mu': tfd.Normal(loc=tf.zeros(self.n_changepoints+1), scale=1.),
            'sigma': tfd.HalfNormal(scale=tf.ones(self.n_changepoints+1))
        })
        
        # Optimization
        optimizer = tf.optimizers.Adam(learning_rate)
        
        @tf.function
        def train_step(batch):
            with tf.GradientTape() as tape:
                loss = -self.model.log_prob(batch)
            grads = tape.gradient(loss, surrogate_posterior.trainable_variables)
            optimizer.apply_gradients(zip(grads, surrogate_posterior.trainable_variables))
            return loss
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                epoch_loss += train_step(batch)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss.numpy()/len(X):.4f}")
                
        self.posterior = surrogate_posterior
        return self
    
    def detect_changepoints(self, data: np.ndarray, n_samples: int = 1000) -> Dict:
        """Detect change points with simplified sampling"""
        X = self._prepare_data(data)
        samples = self.posterior.sample(n_samples)
        
        # Find most likely change points
        tau_samples = samples['tau'].numpy()
        unique, counts = np.unique(tau_samples, return_counts=True)
        changepoints = unique[np.argsort(-counts)][:self.n_changepoints]
        
        return {
            'changepoints': changepoints,
            'uncertainty': np.std(tau_samples, axis=0),
            'samples': samples
        }