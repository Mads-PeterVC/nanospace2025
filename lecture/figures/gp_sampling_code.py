import numpy as np
    
X = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) # Sample points
covariance = np.eye(len(X))                  # Construct convariance matrix
mean = np.zeros(len(X))                      # Mean vector for the GP

sample = np.random.multivariate_normal(mean, covariance) # Sample from the GP
