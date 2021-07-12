import numpy as np
import os
#from matplotlib import pyplot as plt
#from factor_analyzer import FactorAnalyzer, Rotator
import pathlib
import re
import pandas as pd
from ddata.parsing import ii2
import sys
sys.path.append('src')
from ddata.em import epsilon_test, fa_ELBO_delta_estimate, fa_E_step, x, sample_z_from_q
from functools import partial

outdir = pathlib.Path('outputs')
elbos = np.load(outdir / 'elbo.npy')

def get_fns(path):
    fns = os.listdir(path)
    fns = [x for x in fns if re.match('iter(\d*).npy', x)]
    fns.sort(key=lambda x: int(re.findall('iter(\d*).npy', x)[0]))
    return fns

As = np.array([np.load(outdir / 'A' / fn) for fn in get_fns(outdir / 'A')])
Ψs = np.array([np.load(outdir / 'Psi' / fn) for fn in get_fns(outdir / 'Psi')])

# np.save(outdir / 'A' / 'all', As)
# np.save(outdir / 'Psi' / 'all', Ψs)

def analyse():
    # plt.plot(elbos)
    i = np.argmax(elbos)
    A = As[i]
    Ψ = Ψs[i]

def compare():
    x = ii2.drop(labels='subject', axis=1).to_numpy()
    # set([x[:x.index('_')] for x in ii2.columns[1:]])
    questions = {'STAI', 'AES', 'SDS', 'OCI', 'SCZ', 'BIS', 'AUDIT', 'LSAS', 'EAT'}

    print('fitting generic factor analyser')
    fa = FactorAnalyzer(rotation='oblimin')
    fa.fit(x)

    print('rotating our factor loadings')
    rotator = Rotator(method='oblimin')
    A_oblimin = rotator.fit_transform(A)

    k=3
    stuff = {f'fa_obli{i}': A_oblimin[:, i] for i in range(k)}
    stuff.update({f'fa_FA{i}': fa.loadings_[:, i] for i in range(k)})
    stuff['question'] = ii2.columns[1:]
    factors = pd.DataFrame(stuff)
    factors = factors[['question'] + [f'fa_obli{i}' for i in [0,1,2]] + [f'fa_FA{i}' for i in [2,1,0]]]
    # We find this ordering makes the factors seem comparable (though maybe some scaling?)


def do_test_on_A(A, Ψ):
    """epsilon tests wrt A's params."""
    mu_q, Σ_q = fa_E_step(x, A, Ψ)
    sampled_z = sample_z_from_q(mu_q, Σ_q, n_samples=100)
    Ψ_inv = np.linalg.inv(Ψ)
    f = lambda A: fa_ELBO_delta_estimate(x, A, Ψ_inv, sampled_z)
    out = epsilon_test(A, f)
    return out

if __name__ == '__main__':
    A, Ψ = As[8], Ψs[8]
    do_test_on_A(A, Ψ)