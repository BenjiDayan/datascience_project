import numpy as np
import os
#from matplotlib import pyplot as plt
#from factor_analyzer import FactorAnalyzer, Rotator
import pathlib
import re
import pandas as pd

outdir = pathlib.Path('outputs')
elbos = np.load(outdir / 'elbo.npy')

def get_fns(path):
    fns = os.listdir(path)
    fns = [x for x in fns if re.match('iter(\d*).npy', x)]
    fns.sort(key=lambda x: int(re.findall('iter(\d*).npy', x)[0]))
    return fns

As = np.array([np.load(outdir / 'A' / fn) for fn in get_fns(outdir / 'A')])
Ψs = np.array([np.load(outdir / 'Psi' / fn) for fn in get_fns(outdir / 'Psi')])
np.save(outdir / 'A' / 'all', As)
np.save(outdir / 'Psi' / 'all', Ψs)

# plt.plot(elbos)
i = np.argmax(elbos)
A = As[i]
Ψ = Ψs[i]

from ddata.parsing import ii2
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


import sys
sys.path.append('src')
from ddata.em import epsilon_test, fa_ELBO_estimate, fa_E_step, x
from functools import partial
A, Ψ = As[8], Ψs[8]
mu_q, Σ_q = fa_E_step(x, A, Ψ)
f = lambda A: fa_ELBO_estimate(x, A, Ψ, mu_q, Σ_q)
out = epsilon_test(A, f)