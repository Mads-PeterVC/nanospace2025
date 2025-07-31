import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from bayes_nanospace2025.lecture import set_plot_settings

set_plot_settings()

def kernel(X1, X2):
    """Simple squared exponential kernel."""
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return np.exp(-0.5 * sqdist / 1**2)

def moment_of_inertia(X):
    M = X[:, 0]
    r = X[:, 1]
    result = M * r**2
    return np.sum(result)

# Training data: 
x1 = np.array([[0.1, 2], [0.3, 2]])
x2 = np.array([[0.2, 2], [0.5, 2]])
x3 = np.array([[0.4, 2], [0.5, 2]])

X_train = np.vstack((x1, x2, x3))
y_train = [moment_of_inertia(x) for x in [x1, x2, x3]]

L_train = block_diag(*[[1, 1], [1, 1], [1, 1]])

K = L_train @ kernel(X_train, X_train) @ L_train.T
K_inv = np.linalg.inv(K + 1e-6 * np.eye(K.shape[0]))  # Adding a small value for numerical stability

# Query points:
X_query = np.zeros((20, 2))
X_query[:, 0] = np.linspace(0.1, 1.5, 20)
X_query[:, 1] = 2
L_query = block_diag(*[1]*len(X_query)) # Dumb np.eye
K_query = (L_query @ kernel(X_query, X_train) @ L_train.T)
y_query = K_query @ K_inv @ y_train
cov_query = L_query @ kernel(X_query, X_query) @ L_query.T - K_query @ K_inv @ K_query.T
std = np.sqrt(np.diag(cov_query))

X_query_2 = np.zeros((40, 2))
X_query_2[:, 0] = np.repeat(np.linspace(0.1, 1.5, 20), 2)
X_query_2[:, 1] = 2
L_query_2 = block_diag(*[[1, 1]]*20)
K_query_2 = (L_query_2 @ kernel(X_query_2, X_train) @ L_train.T)
y_query_2 = K_query_2 @ K_inv @ y_train
cov_query_2 = L_query_2 @ kernel(X_query_2, X_query_2) @ L_query_2.T - K_query_2 @ K_inv @ K_query_2.T
std_2 = np.sqrt(np.diag(cov_query_2))

n = 3
X_query_3 = np.zeros((n*20, 2))
X_query_3[:, 0] = np.repeat(np.linspace(0.1, 1.5, 20), n)
X_query_3[:, 1] = 2
L_query_3 = block_diag(*[[1]*n]*20)
K_query_3 = (L_query_3 @ kernel(X_query_3, X_train) @ L_train.T)
y_query_3 = K_query_3 @ K_inv @ y_train
cov_query_3 = L_query_3 @ kernel(X_query_3, X_query_3) @ L_query_3.T - K_query_3 @ K_inv @ K_query_3.T
std_3 = np.sqrt(np.diag(cov_query_3))


fig, ax = plt.subplots()

l1, = ax.plot(X_query[:, 0], y_query, '-o')
l2, = ax.plot(X_query[:, 0], y_query_2, '-o')
l3, = ax.plot(X_query[:, 0], y_query_3, '-o')
ax.fill_between(X_query[:, 0], y_query - 2*std, y_query + 2*std, alpha=0.2, color=l1.get_color())
ax.fill_between(X_query[:, 0], y_query_2 - 2*std_2, y_query_2 + 2*std_2, alpha=0.2, color=l2.get_color())
ax.fill_between(X_query[:, 0], y_query_3 - 2*std_3, y_query_3 + 2*std_3, alpha=0.2, color=l3.get_color())


y_query_true = np.array([moment_of_inertia(x.reshape(1, 2)) for x in X_query])
ax.plot(X_query[:, 0], y_query_true, '-', label='True Moment of Inertia', color='gray')
ax.plot(X_query[:, 0], y_query_true*2, '-', label='True Moment of Inertia', color='gray')
ax.plot(X_query[:, 0], y_query_true*3, '-', label='True Moment of Inertia', color='gray')



ax.set_xlabel('Mass')
ax.set_ylabel('Moment of Inertia')
ax.set_title('Predicted Moment of Inertia vs Mass')

plt.savefig('test.png')







