import numpy as np
import os
#from matplotlib import pyplot as plt
#from factor_analyzer import FactorAnalyzer, Rotator
import pathlib
import re
import pandas as pd
import sys
sys.path.append('src')
from ddata.em import epsilon_test, fa_ELBO_delta_estimate, fa_ELBO_delta_estimate_better, \
    fa_E_step, x, sample_z_from_q
from functools import partial

outdir = pathlib.Path('outputs')
elbos = np.load(outdir / 'elbo.npy')

def get_fns(path, regex='iter(\d*)_E.npy'):
    fns = os.listdir(path)
    fns = [x for x in fns if re.match(regex, x)]
    fns.sort(key=lambda x: int(re.findall(regex, x)[0]))
    return fns

As_E = np.array([np.load(outdir / 'A' / fn) for fn in get_fns(outdir / 'A')])
Ψs_E = np.array([np.load(outdir / 'Psi' / fn) for fn in get_fns(outdir / 'Psi')])
As_M = np.array([np.load(outdir / 'A' / fn) for fn in get_fns(outdir / 'A', regex='iter('
                                                                                  '\d*)_M.npy')])
Ψs_M = np.array([np.load(outdir / 'Psi' / fn) for fn in get_fns(outdir / 'Psi', regex='iter('
                                                                                      '\d*)_M.npy')])

# np.save(outdir / 'A' / 'all', As)
# np.save(outdir / 'Psi' / 'all', Ψs)
#
# # plt.plot(elbos)
# i = np.argmax(elbos)
# A = As[i]
# Ψ = Ψs[i]
#
# from ddata.parsing import ii2
# x = ii2.drop(labels='subject', axis=1).to_numpy()
# # set([x[:x.index('_')] for x in ii2.columns[1:]])
# questions = {'STAI', 'AES', 'SDS', 'OCI', 'SCZ', 'BIS', 'AUDIT', 'LSAS', 'EAT'}
#
# print('fitting generic factor analyser')
# fa = FactorAnalyzer(rotation='oblimin')
# fa.fit(x)
#
# print('rotating our factor loadings')
# rotator = Rotator(method='oblimin')
# A_oblimin = rotator.fit_transform(A)
#
# k=3
# stuff = {f'fa_obli{i}': A_oblimin[:, i] for i in range(k)}
# stuff.update({f'fa_FA{i}': fa.loadings_[:, i] for i in range(k)})
# stuff['question'] = ii2.columns[1:]
# factors = pd.DataFrame(stuff)
# factors = factors[['question'] + [f'fa_obli{i}' for i in [0,1,2]] + [f'fa_FA{i}' for i in [2,1,0]]]
# # We find this ordering makes the factors seem comparable (though maybe some scaling?)



A, Ψ = As_M[10], Ψs_M[10]


if __name__ == '__main__':
    def test(A, Ψ, n_samples):
        mu_q, Σ_q = fa_E_step(x, A, Ψ)
        sampled_z = sample_z_from_q(mu_q, Σ_q, n_samples=n_samples)
        elbo = fa_ELBO_delta_estimate_better(x, A, Ψ, sampled_z)
        f1 = lambda A: fa_ELBO_delta_estimate_better(x, A, Ψ, sampled_z) - elbo
        out1 = epsilon_test(A, f1)
        f2 = lambda Ψ: fa_ELBO_delta_estimate_better(x, A, Ψ, sampled_z) - elbo
        indices = [(i,i) for i in range(Ψ.shape[0])]
        out2 = epsilon_test(Ψ, f2, indices=indices)
        return out1, out2
    outputs = []
    outputs.append(test(As_E[10], Ψs_E[10], 15))
    outputs.append(test(As_M[10], Ψs_M[10], 15))