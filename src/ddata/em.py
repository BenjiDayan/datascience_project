from dataclasses import dataclass
import os
import numpy as np
import math
from scipy.stats import multivariate_normal
import pathlib
from tqdm import tqdm
from itertools import islice

from ddata.parsing import ii2


# n=5
# d=3
# k=2
# x = np.random.random((n,d))
# A = np.random.random((d,k))
# mu= np.random.random((d,))
# Ψ = np.zeros((d,d))
# np.fill_diagonal(Ψ, np.random.random(d))

# we will work with x pre normalized. Throughout E-M iterations to improve on A, mu, Psi, we always achieve
# mu = (1/n) Sum_i x_i, so we may as well just compute that once and replace x -> x - mu
# (this is the M-step of maximising mu, A, Ψ given fixed distributions z_i ~ q_i

# @dataclass
# class FA_state:
#     x: np.ndarray       # (n,d)
#     A: np.ndarray       # (d,k)
#     Ψ: np.ndarray       # (d,d)
#     mu_q: np.ndarray    # (k,)
#     Σ_q: np.ndarray     # (k,k)
#
#     @property
#     def num_data(self):
#         return self.x.shape[0]
#
#     @property
#     def num_factors(self):
#         return self.A.shape[1]
#
#     @property
#     def num_dims(self):
#         return self.x.shape[1]


def fa_E_step(x, A, Ψ):
    """Q_i(z_i) = p(z_i | x_i; A)
    x: (n,d)
    Ψ: (d,d)
    A: (d,k)

    :returns mu_q, Σ_q: (n,k), (k,k)
    """
    k = A.shape[1]

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

def fa_M_step(µ_q, Σ_q, x):
    """

    :param µ_q: (n,k)
    :param Σ_q: (k,k)
    :param x: (n,d)
    :return: A, Ψ: (d,k), (d,d) optimized to maximize ELBO(Q; A,Ψ)
    """
    n, d = x.shape

    # A = (Sum_i x_i mu_q.T ) @ (Sum_i µ_q @ µ_q.T + Σ_q)^{-1}
    # This is (d,k) @ (k,k) = (d,k) as required
    # (:, d) (:, k) @  ( (:, k) (:, k) + (k, k) )^{-1}
    A = np.tensordot(x, µ_q, ([0], [0])) @ np.linalg.inv(np.tensordot(µ_q, µ_q, ([0], [0])) + n*Σ_q)

    # Φ = (1/n) Sum_i=1^n x_i x_i^T - x_i mu_q^T A^T - A µ_q x_i^T + A (µ_q µ_q^T + Σ_q) A^T
    # tensored = np.tensordot(x, µ_q @ A.T, ([0], [0]))  # x_i mu_q^T A^T
    # Φ = (1/n) * (
    #     np.tensordot(x, x, ([0], [0])) +
    #     (-tensored - tensored.T) +
    #     A @ (np.tensordot(µ_q, µ_q, ([0], [0])) + n*Σ_q) @ A.T
    # )

    # we simplify the expression above via the formula A = (Sum x_i E[z_i|x_i]) (Sum_i E[z_i z_i^T|x_i]).
    # i.e. A (µ_q µ_q^T + Σ_q) A^T = (Sum x_i E[z_i|x_i]) A^T

    Φ = (1/n) * (
        np.tensordot(x, x, ([0], [0])) -
        A @ np.tensordot(µ_q, x, ([0], [0]))
    )

    # We extract a diagonal matrix from this by dropping off diagonal elements to 0
    Ψ = np.zeros(shape=(d,d))
    np.fill_diagonal(Ψ, np.diag(Φ))

    return A, Ψ


def Q(µ_q, Σ_q, size=None):
    return np.random.multivariate_normal(µ_q, Σ_q, size=size)

# def epsilon_test(x, A, Ψ, mu_q, Σ_q):

def epsilon_test_classify(f_plus, f_minus):
    """
    :param f_plus: (float) f(x + ε) - f(x)
    :param f_minus: (float) f(x - ε) - f(x)
    :return: 0,1,2,3 for different cases
    """
    # U bend
    if f_plus * f_minus > 0:
        if f_plus > 0:
            return 0  # U
        return 1  # upside down U
    # slope
    elif f_plus * f_minus < 0:
        if f_plus - f_minus > 0:
            return 2  # /
        return 3  # \
    else:
        return 4 # unknown! At least one is 0.

def epsilon_test(my_var, f, eps=1e-3, indices=None):
    """Performs epsilon test on parameters of my_var wrt func f.
    :param my_var: (np.ndarray) our parameter for f, which we want to gradient test - is it a local optimum or not?
    :param f: f(my_var + eps) is the function we're interested in
    :param eps: (float) amount to perturb my_var by
    :param indices: Optional[List[Tuple]] optionally specify which indices of my_var we wish to test. Other indices will
        be -1
    :return:
    """
    shape = my_var.shape
    output = np.zeros(shape)
    delta_var = -np.ones(shape)
    if indices is None:
        it = np.nditer(my_var, flags=['multi_index'])
        iterator = tqdm((it.multi_index for _ in it),
                        total=math.prod(shape))
    else:
        iterator = tqdm(indices)
    for index in iterator:
        delta_var[index] += eps
        f_plus, f_minus, classification = epsilon_test_plus_minus_perturbation(f, my_var, delta_var)
        delta_var[index] = 0.
        output[index] = classification

    return output

def epsilon_test_plus_minus_perturbation(f, my_var, delta_var):
    """
    :param f: func to test displacement about
    :param my_var: base point
    :param delta_var: perturbation
    :return:
    """
    f_plus = f(my_var + delta_var)
    f_minus = f(my_var - delta_var)
    return f_plus, f_minus, epsilon_test_classify(f_plus, f_minus)

def sample_z_from_q(µ_q, Σ_q, n_samples=100):
    """
    :param µ_q: (n,k)
    :param Σ_q: (k,k)
    :param n_samples: number of samples to sample
    :return:
    """
    # list of distributions q_i(z_i)
    Q = [multivariate_normal(µ, Σ_q) for µ in µ_q]  # (n,)
    k = Σ_q.shape[0]

    # (n, n_samples, k)
    sampled_z = np.stack([q_i.rvs(size=n_samples).reshape(-1, k) for q_i in Q])
    return sampled_z

def fa_ELBO_delta_estimate(x, A, Ψ, sampled_z):
    """
    :param x: (n,d)
    :param A: (d,k)
    :param Ψ: (d,d)  NB this should be diagonal as we use that to save computation.
    :param sampled_z: (n, n_samples, k)
    :return: (float) simplified estimate of ELBO(Q(~sampled_z); A, Ψ) (much crud thrown out of formula)
    """
    # The only part of ELBO formula influenced by A, Ψ is the log p(x|z; A,Ψ) = (-1/2) [log |Ψ| + x^T Ψ^{-1} x]

    n_samples = sampled_z.shape[1]
    n = x.shape[0]

    tiled_x = np.expand_dims(x, axis=-2)  # (n,d) -> (n,1,d)
    tiled_x = np.tile(tiled_x, (1, n_samples, 1))  # (n,1,d) -> (n, n_samples, d)

    cond = (tiled_x - sampled_z @ A.T)  # (n, n_samples, d) this is the x|z conditional dist, ~ N(0, Ψ)
    a1, a2 = np.expand_dims(cond, axis=2), np.expand_dims(cond, axis=3)  # (n, n_samples, 1, d), (n, n_samples, d, 1)
    out = (
        + np.log(np.linalg.det(Ψ))
          # ((n, n_samples, 1, d) * (d,) (diagonal)) @ (n, n_samples, d, 1) = (n, n_samples, 1, 1)
        + np.squeeze((a1 * np.diag(np.linalg.inv(Ψ))) @ a2)  # squeezed, then summed over all axes
    ).sum() / (- 2 * n * n_samples)
    # normalizing to n and n_samples is good. We may as well also keep in the factor of 2
    return out

def fa_ELBO_estimate(x, A, Ψ, mu_q, Σ_q, n_samples=100):
    """Estimate ELBO by randomly sampling over z distribution
    :param x: (n,d)
    :param A: (d,k)
    :param Ψ: (d,d)
    :param mu_q: (n,k)
    :param Σ_q: (k,k)
    :param n_samples: (int) for each x_i, how many times to sample from z_i ~ N(µ_q_i, Σ_q) in order to approximate
        E_{z_i ~ Q_i}
    :return: an estimate of ELBO(Q; A,Ψ) = Sum_{i=1}^n E_{z_i ~ Q_i}[ log ( p(x_i, z_i; A,Ψ) / Q_i(z_i) ) ]
    """
    # We generate the base multivariate normal distribution of (z,x)
    k, d = mu_q.shape[1], x.shape[1]
    joint_mean = np.zeros(k+d)
    joint_cov = np.concatenate(
        [
            np.concatenate([np.eye(k),  A.T],       axis=1),
            np.concatenate([A,          A@A.T + Ψ], axis=1)
        ]
        , axis=0
    )
    joint_rv = multivariate_normal(joint_mean, joint_cov)

    # list of distributions q_i(z_i)
    Q = [multivariate_normal(mu, Σ_q) for mu in mu_q]  # (n,)

    # (n, n_samples, k)
    sampled_z = np.stack([q_i.rvs(size=n_samples) for q_i in Q])

    tiled_x = np.expand_dims(x, axis=-2)  # (n,d) -> (n,1,d)
    tiled_x = np.tile(tiled_x, (1, n_samples, 1))  # (n,1,d) -> (n, n_samples, d)

    joint_samples = np.concatenate([sampled_z, tiled_x], axis=-1)  # (n, n_samples, k+d)
    # this is log of p(z_i, x_i; A, Ψ)
    joint_log_probs = np.log(joint_rv.pdf(joint_samples))  # pdf is applied on last dimension

    q_log_probs = np.log(np.stack([q_i.pdf(z_i_samples) for q_i, z_i_samples in zip(Q, sampled_z)]))

    return np.sum(joint_log_probs - q_log_probs) / n_samples


def run_fa_em(x, A, Ψ, max_iter=10000, conv_eps=1e-4, outs_dir=None):
    if outs_dir is not None:
        os.makedirs(outs_dir / 'A', exist_ok=True)
        os.makedirs(outs_dir / 'Psi', exist_ok=True)

    elbos = []
    n = x.shape[0]
    mu = (1/n) * x.sum(axis=0)
    x = x - mu

    elbo_iter = 0
    A_prev, Ψ_prev = A, Ψ

    iteration = 0
    while iteration < max_iter:
        print(f'iteration: {iteration}')
        mu_q, Σ_q = fa_E_step(x, A, Ψ)

        # ELBO(Q, theta) <= LL(x,theta) with equality when q_i = p(z_i | x_i; theta)
        # ELBO(Q, theta) = Sum_i=1^n E_{z_i ~ q_i} [log p(x_i, z_i; theta) - log q_i(z_i)]
        # [z,x] ~ N([0, mu], [[I, A^T], [A, AA^T + Ψ]])
        # We will sample from E_{z_i ~ q_i} to estimate
        elbo = fa_ELBO_estimate(x, A, Ψ, mu_q, Σ_q)
        print(f'ELBO after E-step: {elbo}')
        elbos.append([elbo])

        elbo_diff = elbo - elbo_iter
        if abs(elbo_diff) < conv_eps:
            print(f'convergence criteria elbo diff < conv_eps={conv_eps} met')
            break
        print(f'elbo - elbo_prev: {elbo_diff}')
        print(f'|A-A_prev|: {np.linalg.norm(A-A_prev)}; |Ψ-Ψ_prev|: {np.linalg.norm(Ψ-Ψ_prev)}')

        if outs_dir is not None:
            np.save(outs_dir / 'A' / f'iter{iteration}_E', A)
            np.save(outs_dir / 'Psi' / f'iter{iteration}_E', Ψ)
            np.save(outs_dir / 'elbo', elbos)

        elbo_iter = elbo

        A_prev, Ψ_prev = A, Ψ
        A, Ψ = fa_M_step(mu_q, Σ_q, x)

        elbo = fa_ELBO_estimate(x, A, Ψ, mu_q, Σ_q)
        print(f'ELBO after M-step: {elbo}')
        elbos[-1].append(elbo)
        if outs_dir is not None:
            np.save(outs_dir / 'A' / f'iter{iteration}_M', A)
            np.save(outs_dir / 'Psi' / f'iter{iteration}_M', Ψ)
            np.save(outs_dir / 'elbo', elbos)
        iteration += 1



def main(x, outs_dir=None):
    k = 3
    d = x.shape[1]
    A = np.random.rand(d, k)
    Ψ = np.zeros((d,d))
    np.fill_diagonal(Ψ, np.random.random(d))

    from analysis import As_M, Ψs_M
    A, Ψ = As_M[-1], Ψs_M[-1]


    run_fa_em(x, A, Ψ, max_iter=100, outs_dir=outs_dir)

x = ii2.drop(labels='subject', axis=1).to_numpy()
if __name__ == '__main__':
    np.random.seed(42)
    main(x, outs_dir=pathlib.Path('outputs_continued'))  # drop the subj column