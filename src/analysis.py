import numpy as np
import os
from matplotlib import pyplot as plt
from factor_analyzer import FactorAnalyzer, Rotator
import pathlib
import re
import pandas as pd
from ddata.parsing import ii2
import sys
sys.path.append('src')
from ddata.em import epsilon_test, fa_ELBO_delta_estimate, fa_E_step, x, sample_z_from_q
from functools import partial


def get_fns(path, regex='iter(\d*)_E.npy'):
    fns = os.listdir(path)
    fns = [x for x in fns if re.match(regex, x)]
    fns.sort(key=lambda x: int(re.findall(regex, x)[0]))
    return fns

def extract_from_dir(outdir):
    As_E = np.array([np.load(outdir / 'A' / fn) for fn in get_fns(outdir / 'A')])
    Ψs_E = np.array([np.load(outdir / 'Psi' / fn) for fn in get_fns(outdir / 'Psi')])
    As_M = np.array([np.load(outdir / 'A' / fn) for fn in get_fns(outdir / 'A', regex='iter('
                                                                                      '\d*)_M.npy')])
    Ψs_M = np.array([np.load(outdir / 'Psi' / fn) for fn in get_fns(outdir / 'Psi', regex='iter('
                                                                                          '\d*)_M.npy')])
    elbos = np.load(outdir / 'elbo.npy', allow_pickle=True)
    A_shape, Ψ_shape = As_E.shape[1:], Ψs_E.shape[1:]
    As, Ψs = np.array(list(zip(As_E, As_M))).reshape(-1, *A_shape), np.array(list(zip(Ψs_E, Ψs_M))).reshape(-1, *Ψ_shape)

    return elbos, As, Ψs

outdirs = [
    pathlib.Path('outputs'),
    pathlib.Path('outputs_continued'),
    pathlib.Path('outputs_continued2'),
]
zipped = list(map(extract_from_dir, outdirs))
zipped = zip(*zipped)
elbos, As, Ψs = [np.concatenate(arr_list) for arr_list in zipped]

A, Ψ = As[-1], Ψs[-1]
weights = pd.read_csv('osfstorage-archive/Experiment 2/weights.csv')
weights = weights.iloc[:, 1:].values

# elbos, As, Ψs = extract_from_dir()
# A, Ψ = As[-1], Ψs[-1]

# np.save(outdir / 'A' / 'all', As)
# np.save(outdir / 'Psi' / 'all', Ψs)

def analyse(elbos):
    fig = plt.figure()
    plt.plot([x[0] for x in elbos])
    plt.plot([x[1] for x in elbos if len(x) == 2])
    i = np.argmax(elbos)
    return i

def compare(A, Ψ):
    x = ii2.drop(labels='subject', axis=1).to_numpy()
    # set([x[:x.index('_')] for x in ii2.columns[1:]])
    questions = {'STAI', 'AES', 'SDS', 'OCI', 'SCZ', 'BIS', 'AUDIT', 'LSAS', 'EAT'}

    print('fitting generic factor analyser')
    fa = FactorAnalyzer(rotation='oblimin')
    fa.fit(x)

    print('rotating our factor loadings')
    rotator = Rotator(method='oblimin')
    A_oblimin = rotator.fit_transform(A)

    return fa, A_oblimin

    k=3
    stuff = {f'fa_obli{i}': A_oblimin[:, i] for i in range(k)}
    stuff.update({f'fa_FA{i}': fa.loadings_[:, i] for i in range(k)})
    stuff['question'] = ii2.columns[1:]
    factors = pd.DataFrame(stuff)
    factors = factors[['question'] + [f'fa_obli{i}' for i in [0,1,2]] + [f'fa_FA{i}' for i in [2,1,0]]]
    # We find this ordering makes the factors seem comparable (though maybe some scaling?)

def do_test_on_A(A, Ψ, n_samples):
    """epsilon tests wrt A's params and the diagonal bits of Ψ's params."""
    mu_q, Σ_q = fa_E_step(x, A, Ψ)
    sampled_z = sample_z_from_q(mu_q, Σ_q, n_samples=n_samples)
    elbo = fa_ELBO_delta_estimate(x, A, Ψ, sampled_z)
    f1 = lambda A: fa_ELBO_delta_estimate(x, A, Ψ, sampled_z) - elbo
    out1 = epsilon_test(A, f1)
    f2 = lambda Ψ: fa_ELBO_delta_estimate(x, A, Ψ, sampled_z) - elbo
    indices = [(i,i) for i in range(Ψ.shape[0])]
    out2 = epsilon_test(Ψ, f2, indices=indices)
    return out1, out2

def shift_outputs(source_folder, sink_folder):
    regex = 'iter(\d*)_M.npy'
    A_fns = get_fns(sink_folder / 'A', regex=regex)

    i = re.findall(regex, A_fns[-1])[0]


# A1, Ψ1 = As_M[3], Ψs_M[3]
# A2, Ψ2 = As_M[11], Ψs_M[11]
#
# A, Ψ = A1, Ψ1
# n_samples=15
# mu_q, Σ_q = fa_E_step(x, A, Ψ)
# sampled_z = sample_z_from_q(mu_q, Σ_q, n_samples=n_samples)
# elbo = fa_ELBO_delta_estimate(x, A, Ψ, sampled_z)
# if __name__ == '__main__':
#     outputs = []
#     outputs.append(do_test_on_A(As_E[10], Ψs_E[10], 15))
#     outputs.append(do_test_on_A(As_M[10], Ψs_M[10], 15))