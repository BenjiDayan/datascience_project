import numpy as np

n=5
d=3
k=2
x = np.random.random((n,d))
A = np.random.random((d,k))
mu= np.random.random((d,))
Psi = np.random.random((d,d))

# we will work with x pre normalized. Throughout E-M iterations to improve on A, mu, Psi, we always achieve
# mu = (1/n) Sum_i x_i, so we may as well just compute that once and replace x -> x - mu

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