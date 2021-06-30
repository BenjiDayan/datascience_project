import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)

def gaussian_pdf(x, mu, sigma):
    """
    Args:
        x: (..., d)
        mu: (..., d)
        sigma: (..., d, d)

    Returns: (...,)
    """
    d = x.shape[-1]
    assert d == mu.shape[-1] and d == sigma.shape[-2] and d == sigma.shape[-1]
    sigma_inv = np.linalg.inv(sigma)
    expanded = np.expand_dims(x - mu, axis=(-2))
    transpose_indices = np.concatenate([np.arange(len(expanded.shape) - 2), [-1, -2]])
    return (2 * np.pi)**(-d/2) * np.linalg.det(sigma_inv) * \
        np.exp(
            (-1/2) * np.squeeze(expanded @ sigma_inv @ expanded.transpose(transpose_indices))
        )


def run_em(x, w, phi, mu, sigma, w_overrides=None, supervised_alpha=1):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).
        w_overrides: Optional[List[Tuple]] optional list of pairs of (i, j) where i are the indices of datapoints for
            which we have supervised latent labels, and j is the cluster for said labels.
        supervised_alpha: float - weight with which to M-step for maximising wrt supervised labels
    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    k = phi.shape[0]  # number of clusters = k
    n = x.shape[0]

    # we make w_overrides have shape (n,) with -1 for no override otherwise 0,1,..,k-1 for which class
    if w_overrides is not None:
        i,j = 0, 0
        w_overrides_new = [-1]*n
        while i < n and j < len(w_overrides):
            if i == w_overrides[j][0]:
                w_overrides_new[i] = w_overrides[j][1]
                j += 1
            i += 1
        w_overrides = np.array(w_overrides_new)
    else:
        w_overrides = -np.ones(n).astype(np.int32)  # all -1 so override no indices

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # (1) E-step: Update your estimates in w
        # w^i_j = p(z_i = j | x_i ; mu, Sigma, phi) = p(x_i | z_i) p(z_i) / p(x_i)
        # = N(x_i; mu_j, Sigma_j) phi_j / (sum_k=1^K N(x_i ; mu_k, Sigma_k) phi_k)

        # x_cluster_copies = np.tile(x, (k, 1, 1)).swapaxes(0, 1)  # (n x k x d) k copies of x (x is nxd)
        # posterior = gaussian_pdf(x_cluster_copies, mu, sigma)  # N(x_i; mu_j, Sigma_j) is (n x k)
        # prob_xi = (phi * posterior).sum(axis=1)  # p(x_i) is (n,)
        # w = (1 / prob_xi).reshape(-1, 1) * phi * posterior  # (n, 1) * (k) * (n x k) = (n x k)
        prob_x, prob_x_given_z, prob_z_given_x = compute_posterior(x, phi, mu, sigma)
        w = prob_z_given_x

        # override w, where w_overrides isn't -1, to the [0, 0, ..., 0, alpha, 0, ..., 0] one hot vector for the cluster
        # (i.e. overriding rows of w, replacing some with the right one hot vector)
        w = np.where(
            np.tile((w_overrides >= 0).reshape(-1, 1), (1, k)),
            supervised_alpha * np.eye(k)[w_overrides],
            w
        )

        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi = w.sum(axis=0)/w.sum()  # i.e. divide by n + alpha tilde{n} in semi supervised case, n in supervised case
        mu = (  # (k, d)
                np.expand_dims(w, -1) @ np.expand_dims(x, -2)  # (n, k, 1) @ (n, 1, d) = (n, k, d)
             ).sum(axis=0) / \
             w.sum(axis=0).reshape(-1, 1)

        expanded = np.expand_dims(x, 1) - np.tile(mu, (n, 1, 1))  # (n, 1, d) - (n, k, d) = (n, k, d)
        cov = (np.expand_dims(expanded, -1) @ np.expand_dims(expanded, 2))  # (n, k, d, 1) @ (n, k, 1, d) = (n, k, d, d)
        sigma = (
            (
                w.reshape(w.shape + (1,1)) * cov  # (n, k, 1, 1) * (n, k, d, d) = (n, k, d, d)
            ).sum(axis=0) / \
            w.sum(axis=0).reshape(-1, 1, 1)  # (k, 1, 1)
        )  # (k, d, d) / (k, 1, 1) = (k, d, d)

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prob_x, prob_x_given_z, prob_z_given_x = compute_posterior(x, phi, mu, sigma)
        # like = (posterior * phi).sum(axis=1)  # (n, k) * (k,) summed over k. (n, k) posterior_i,j = p(x_i | z_i = j)
        prev_ll = ll
        ll = np.log(prob_x).sum()  # take log and sum over the n datapoints
        print(f'iteration {it}: ll: {ll}')
        it += 1

    return w, phi, mu, sigma

def compute_posterior(x, phi, mu, sigma):
    k = phi.shape[0]
    x_cluster_copies = np.tile(x, (k, 1, 1)).swapaxes(0, 1)  # (n x k x d) k copies of x (x is nxd)
    # p(x_i | z_i = j)
    prob_x_given_z = gaussian_pdf(x_cluster_copies, mu, sigma)  # N(x_i; mu_j, Sigma_j) is (n x k)
    # p(x_i) = sum_j p(z_i = j) p(x_i | z_i = j)
    prob_x = (phi * prob_x_given_z).sum(axis=1)  # p(x_i) is (n,)
    prob_z_given_x = (1 / prob_x).reshape(-1, 1) * phi * prob_x_given_z  # (n, 1) * (k) * (n x k) = (n x k)
    return prob_x, prob_x_given_z, prob_z_given_x


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma, alpha=20):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    n = x.shape[0]
    k = phi.shape[0]
    x2 = np.concatenate([x, x_tilde])
    z_tilde = z_tilde.reshape(-1).astype(np.int32)
    w2 = np.concatenate([w, np.eye(k)[z_tilde]])
    w_overrides = [[n + i, z] for i, z in enumerate(z_tilde)]
    w, phi, mu, sigma = run_em(x2, w2, phi, mu, sigma, w_overrides=w_overrides, supervised_alpha=alpha)
    return w, phi, mu, sigma


# Helper functions

def plot_gmm_preds(x, z, with_supervision, plot_id, mu=None, sigma=None, phi=None):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    # mu is (k, d) and sigma (k, d, d) so for each of k clusters:
    if not (mu is None or sigma is None or phi is None):
        for mu_l, sigma_l, phi_l, color in zip(mu, sigma, phi, PLOT_COLORS):
            width, height, angle = covariance_to_ellipse(sigma_l)
            ellipse = matplotlib.patches.Ellipse(mu_l, width, height, angle=angle, color=color, fill=False)
            ax = plt.gca()
            ax.add_patch(ellipse)
            ellipse.set_label(f'ellipse with 2 std sizing; phi: {phi_l}')

        plt.legend()

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)



def covariance_to_ellipse(sigma):
    """
    An ellipse with matplotlib has direction along its height axis, pointing up the y axis. Positive angle
    will tilt this direction into the left half plane.
    Args:
        sigma: (2, 2) symmetric matrix - so it has orthogonal eigenvectors

    Returns: (width, height), (angle in degrees)
    """
    evalues, evecs = np.linalg.eig(sigma)
    e1, e2 = evalues
    evec1, evec2 = evecs.T  # evecs has eigenvectors as columns. So evec1 = evecs[:, 0].

    # whether the height axis is in the left half plane (left side of y axis)
    left_half_plane = np.dot(evec1, [1, 0]) < 0

    angle = (np.arccos(np.dot(evec1, [0, 1])) * 180 / np.pi) * (1 if left_half_plane else -1)
    # This is total height/width (diameter not radius), and we want 2 standard deviations, so 2*2 = 4
    height, width = 4 * np.sqrt(e1), 4 * np.sqrt(e2)
    return width, height, angle


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)

    # x, mu = np.ones(shape=(9, 5, 7, 4)), np.random.rand(9, 5, 7, 4)
    # sigma = np.random.rand(9, 5, 7, 4, 4)
    # out = gaussian_pdf(x, mu, sigma)

    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.

