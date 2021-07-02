import numpy as np
import scipy.stats as stats

n=5
d=3
k=2
x = np.random.random((n,d))
A = np.random.random((d,k))
mu= np.random.random((d,))
Ψ = np.zeros((d,d))
np.fill_diagonal(Ψ, np.random.random(d))

# we will work with x pre normalized. Throughout E-M iterations to improve on A, mu, Psi, we always achieve
# mu = (1/n) Sum_i x_i, so we may as well just compute that once and replace x -> x - mu
# (this is the M-step of maximising mu, A, Ψ given fixed distributions z_i ~ q_i

def fa_E_step(x, A, Ψ):
    """Q_i(z_i) = p(z_i | x_i; A)
    x: (n,d)
    Ψ: (d,d)
    A: (d,k)

    :returns mu_q, Σ_q: (n,k), (k,k)
    """
    # mu_z_i|x_i (k,)
    #   M @ x_i = (k,d) @ (d,)
    # So (n,d) mu_z|x is
    #   x @ M.T = (n,d) @ (d,k) = (n,k) shape
    mu_q = x @ (A.T @ np.linalg.inv(A @ A.T + Ψ)).T

    # Σ_z_i|x_i (k,k)
    #   I - A.T(AA.T + Ψ)^{-1} A
    # So (n,k,k) Σ is just n repeats of that
    Σ_q = np.eye(k) - A.T @ np.linalg.inv(A @ A.T + Ψ) @ A

    return mu_q, Σ_q

def fa_M_step(mu_q, Σ_q, x):
    # A = (Sum_i x_i mu_q.T ) @ (Sum_i mu_q @ mu_q.T + Σ_q)^{-1}
    # This is (d,k) @ (k,k) = (d,k) as required
    # (:, d) (:, k) @  ( (:, k) (:, k) + (k, k) )^{-1}
    A = np.tensordot(x, mu_q, ([0], [0])) @ np.linalg.inv(np.tensordot(mu_q, mu_q, ([0], [0])) + Σ_q)

    # Φ = (1/n) Sum_i=1^n x_i x_i^T - x_i mu_q^T A^T - A mu_q x_i^T + A (mu_q mu_q^T + Σ_q) A^T
    Φ = (1/n) * (
        np.tensordot(x, x, ([0], [0])) -
        np.tensordot(x, mu_q @ A, ([0], [0])) -
        np.tensordot(A @ mu_q, x, ([0], [0])) +
        A @ (np.tensordot(mu_q, mu_q, ([0], [0])) + n*Σ_q) @ A.T
    )

    # We extract a diagonal matrix from this by dropping off diagonal elements to 0
    d = x.shape[1]
    Ψ = np.diag(shape=(d,d))
    np.fill_diagonal(Ψ, np.diag(Φ))

    return A, Ψ

def Q(mu_q, Σ_q, size=None):
    return np.random.multivariate_normal(mu_q, Σ_q, size=size)

def fa_ELBO_estimate(x, A, Ψ, n_samples):


def run_fa_em(x, A, Ψ, max_iter=10000, conv_eps=1e-4):
    mu = (1/n) * x.sum(axis=0)
    x = x - mu

    iteration = 0
    while iteration < max_iter:
        mu_q, Σ_q = fa_E_step(x, A, Ψ)

        # ELBO(Q, theta) <= LL(x,theta) with equality when q_i = p(z_i | x_i; theta)
        # ELBO(Q, theta) = Sum_i=1^n E_{z_i ~ q_i} [log p(x_i, z_i; theta) - log q_i(z_i)]
        # [z,x] ~ N([0, mu], [[I, A^T], [A, AA^T + Ψ]])
        # We will sample from E_{z_i ~ q_i} to estimate



        A, Ψ = fa_M_step(mu_q, Σ_q)




