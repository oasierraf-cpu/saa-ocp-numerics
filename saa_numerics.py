#!/usr/bin/env python3
"""
Numerical experiments for the paper

    M. S. Aronna, J. M. Machado and O. Sierra,
    "Asymptotic analysis of sample average approximations for optimal control problems with uncertain dynamics".

Sample average approximation (SAA) of optimal control problems with control-affine dynamics and
uncertain parameters, with a Tikhonov term (alpha/2)||u||^2 and box constraints on the control.
Two examples are implemented:

  1) oscillator : ensemble of linear oscillators with uncertain frequency (convex problem);
                  alpha = 0.3 (inactive constraints) and alpha = 0.05 (active constraints).
  2) sit        : sterile insect technique model with the parameters of Bidi, Coron, Hayat and
                  Lichtle (Table 1), uncertain competitiveness q ~ U[0.91, 1] and sterile-male death
                  rate dS ~ truncated normal; alpha = 0.1 (inactive) and alpha = 0.03 (active upper bound).

For each example the script computes the solution of the "true" problem by tensor Gauss-Legendre
quadrature, the asymptotic quantities of the central limit theorems (variance of the optimal value,
covariance operator of the optimal control), and Monte Carlo statistics of the SAA estimators for
increasing sample sizes M, and draws the figures of the paper.

Usage
-----
    python saa_numerics.py                         # redraw both figures from results/*.json (seconds)
    python saa_numerics.py --run oscillator        # rerun the oscillator with the settings of the paper
    python saa_numerics.py --run sit               # rerun the SIT example with the settings of the paper
    python saa_numerics.py --run all --mode quick  # fast test run with fewer replications
    python saa_numerics.py --help

Outputs: results/oscillator.json, results/sit_paper.json, figures/fig_oscillator.{pdf,png},
figures/fig_sit_paper.{pdf,png}.  All random numbers come from fixed seeds.
"""
import os
# One BLAS thread per process: avoids oversubscription (and rare hangs after fork on macOS) with the small
# matrices used here.  This must run BEFORE numpy is imported.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from math import sqrt, pi

import numpy as np
import scipy
from scipy.optimize import minimize, lsq_linear
from scipy.stats import truncnorm

HERE = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# 0. Common utilities
# =====================================================================
def boxqp(A, q, lb, ub):
    """Solve min 1/2 u'Au + q'u subject to lb <= u <= ub (A symmetric positive definite)
       by bounded-variable least squares.  Returns the solution and its projected-gradient residual."""
    jit = 0.0
    while True:
        try:
            L = np.linalg.cholesky(A + jit*np.eye(len(A))); break
        except np.linalg.LinAlgError:
            jit = max(1e-12, 10*jit)
    bb = -np.linalg.solve(L, q)
    r = lsq_linear(L.T, bb, bounds=(lb, ub), method='bvls', tol=1e-15, max_iter=2000)
    u = r.x; g = A@u + q; c = np.mean(np.diag(A))
    return u, np.max(np.abs(u-np.clip(u-g/c, lb, ub)))


def metadata(**kw):
    """Information stored in the output files to document how they were produced."""
    meta = dict(created=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform())
    meta.update(kw)
    return meta


def pyplot(show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'legend.fontsize': 6.5, 'lines.linewidth': 1.1})
    return plt


# =====================================================================
# 1. Oscillator with uncertain frequency
#    x1' = x2, x2' = -w^2 x1 + u, x(0) = (1,0), g(x) = |x|^2/2, |u| <= 1, w ~ U[0.5,1.5], T = 3.
#    The final state is affine in u, so every SAA problem is a convex box-constrained QP.
# =====================================================================
def run_oscillator(Nrep=1000, outdir='results'):
    T = 3.0; N = 100; dt = T/N; tgrid = np.linspace(0, T, N+1); umax = 1.0
    wmin, wmax = 0.5, 1.5

    def GA(w):
        """x(T) = a(w) + G(w) u for piecewise constant u (exact integration)."""
        w = np.atleast_1d(w)[:, None]
        tk = tgrid[:-1][None, :]; tk1 = tgrid[1:][None, :]
        G1 = (np.cos(w*(T-tk1))-np.cos(w*(T-tk)))/w**2
        G2 = (np.sin(w*(T-tk))-np.sin(w*(T-tk1)))/w
        G = np.stack([G1, G2], axis=1)                                     # (M,2,N)
        a = np.stack([np.cos(w[:, 0]*T), -w[:, 0]*np.sin(w[:, 0]*T)], axis=1)  # (M,2) for x0=(1,0)
        return G, a

    def moments(w, wts=None):
        G, a = GA(w); M = len(w)
        if wts is None: wts = np.full(M, 1.0/M)
        Q = np.einsum('m,mik,mil->kl', wts, G, G); q = np.einsum('m,mik,mi->k', wts, G, a)
        return Q, q, G, a

    def solve(Q, q, alpha):
        A = Q+alpha*dt*np.eye(N)
        return boxqp(A, q, -umax, umax)

    # "true" problem: 200-node Gauss-Legendre quadrature in w
    gl_x, gl_w = np.polynomial.legendre.leggauss(200)
    wq = (wmin+wmax)/2+(wmax-wmin)/2*gl_x; ww = gl_w/2
    rng = np.random.default_rng(12345)
    Qr, qr, Gq, aq = moments(wq, ww)
    tests = {'h1': np.ones(N), 'h2': np.cos(np.pi*(tgrid[:-1]+dt/2)/T)}   # test functions h_1, h_2
    Ms = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
    out = {}
    t_start = time.time()
    for alpha in [0.3, 0.05]:
        Hmat = Qr+alpha*dt*np.eye(N)
        ubar, res = solve(Qr, qr, alpha)
        V = 0.5*np.sum(ww*np.sum((aq+np.einsum('mik,k->mi', Gq, ubar))**2, 1))+0.5*alpha*dt*ubar@ubar
        gval = 0.5*np.sum((aq+np.einsum('mik,k->mi', Gq, ubar))**2, 1)
        sig2V = np.sum(ww*gval**2)-np.sum(ww*gval)**2                  # sigma^2(hat u)
        r = aq+np.einsum('mik,k->mi', Gq, ubar)                         # residual x(T)
        xi = np.einsum('mik,mi->mk', Gq, r)                             # coefficients of grad g_T
        Exi = np.sum(ww[:, None]*xi, 0)
        Kcov = np.einsum('m,mk,ml->kl', ww, xi-Exi, xi-Exi)             # covariance K(hat u)
        act = np.abs(np.abs(ubar)-umax) < 1e-10; I = ~act
        Bop = np.zeros((N, N)); Bop[np.ix_(I, I)] = np.linalg.inv(Hmat[np.ix_(I, I)])
        Sigma = Bop@Kcov@Bop                                            # asymptotic covariance
        pred = {k: float(dt**2*hfun@Sigma@hfun) for k, hfun in tests.items()}
        trSig = float(dt*np.trace(Sigma)); pstd = np.sqrt(np.diag(Sigma))
        rec = {M: {'dV': [], 'err2': [], 'z': {k: [] for k in tests}, 'cov': {k: [] for k in tests}} for M in Ms}
        dev_store = {}
        t0 = time.time()
        for rep in range(Nrep):
            wall = rng.uniform(wmin, wmax, Ms[-1])                      # nested samples
            for M in Ms:
                w = wall[:M]
                G, a = GA(w); Gs = G.reshape(2*M, N)
                QM = Gs.T@Gs/M; qM = np.einsum('mik,mi->k', G, a)/M
                uM, resM = solve(QM, qM, alpha)
                rM = a+np.einsum('mik,k->mi', G, uM)
                VM = 0.5*np.mean(np.sum(rM**2, 1))+0.5*alpha*dt*uM@uM
                eM = uM-ubar
                rec[M]['dV'].append(VM-V); rec[M]['err2'].append(dt*eM@eM)
                # plug-in covariance (Corollary on inference), Hessian inverted on the empirical inactive set
                xiM = np.einsum('mik,mi->mk', G, rM); xiM -= xiM.mean(0)
                KM = xiM.T@xiM/M
                actM = np.abs(np.abs(uM)-umax) < 1e-10; IM = ~actM
                HM = QM+alpha*dt*np.eye(N)
                for k, hfun in tests.items():
                    z = np.sqrt(M)*dt*hfun@eM
                    rec[M]['z'][k].append(z)
                    x = np.zeros(N); x[IM] = np.linalg.solve(HM[np.ix_(IM, IM)], hfun[IM])
                    vM = dt**2*x@KM@x
                    rec[M]['cov'][k].append(bool(abs(z) <= 1.959964*np.sqrt(vM)))
                if M in (40, 640, 5120):
                    dev_store.setdefault(M, []).append(np.sqrt(M)*eM)
            if rep == 0: print('  oscillator alpha=%g: first replication %.1fs' % (alpha, time.time()-t0), flush=True)
        res_a = {'V': float(V), 'sig2V': float(sig2V), 'pred': pred, 'trSig': trSig, 'nact': int(act.sum()),
                 'ubar': ubar.tolist(), 'pstd': pstd.tolist(),
                 'M': Ms,
                 'EabsdV': [float(np.mean(np.abs(rec[M]['dV']))) for M in Ms],
                 'Eerr2': [float(np.mean(rec[M]['err2'])) for M in Ms],
                 'coverage': {k: [float(np.mean(rec[M]['cov'][k])) for M in Ms] for k in tests},
                 'zvar': {k: [float(np.var(rec[M]['z'][k])) for M in Ms] for k in tests},
                 'sqrtM_dV': {M: (np.sqrt(M)*np.array(rec[M]['dV'])).tolist() for M in (40, 640, 5120)},
                 'z': {M: {k: rec[M]['z'][k] for k in tests} for M in (40, 640, 5120)},
                 'devstd': {M: np.std(np.array(dev_store[M]), 0).tolist() for M in dev_store},
                 'devmeanabs': {M: np.mean(np.abs(np.array(dev_store[M])), 0).tolist() for M in dev_store}}
        out[str(alpha)] = res_a
        print('  oscillator alpha=%g done in %.0fs' % (alpha, time.time()-t0), flush=True)
    # trace of the asymptotic covariance as a function of alpha
    alist = np.logspace(-2, 1, 25); trs = []; nacts = []
    for alpha in alist:
        Hmat = Qr+alpha*dt*np.eye(N); ub, _ = solve(Qr, qr, alpha)
        r = aq+np.einsum('mik,k->mi', Gq, ub); xi = np.einsum('mik,mi->mk', Gq, r); Exi = np.sum(ww[:, None]*xi, 0)
        Kcov = np.einsum('m,mk,ml->kl', ww, xi-Exi, xi-Exi)
        act = np.abs(np.abs(ub)-umax) < 1e-10; I = ~act
        Bop = np.zeros((N, N)); Bop[np.ix_(I, I)] = np.linalg.inv(Hmat[np.ix_(I, I)])
        trs.append(float(dt*np.trace(Bop@Kcov@Bop))); nacts.append(int(act.sum()))
    out['sweep'] = {'alpha': alist.tolist(), 'tr': trs, 'nact': nacts}
    out['meta'] = metadata(example='oscillator', Nrep=Nrep, seed=12345, runtime_s=round(time.time()-t_start))
    path = os.path.join(outdir, 'oscillator.json')
    with open(path, 'w') as fh: json.dump(out, fh)
    print('  saved', path, flush=True)
    return out


def plot_oscillator(json_path, figdir='figures', show=False):
    plt = pyplot(show)
    with open(json_path) as fh: d = json.load(fh)
    T = 3.0; N = 100; dt = T/N; t = np.arange(N)*dt+dt/2
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6))
    a = ax[0, 0]
    for al, c in [('0.3', 'C0'), ('0.05', 'C3')]:
        u = np.array(d[al]['ubar']); a.step(np.r_[0, t+dt/2], np.r_[u[0], u], where='pre', color=c, label=r'$\alpha=%s$' % al)
    a.axhline(1, ls=':', c='k', lw=.8); a.axhline(-1, ls=':', c='k', lw=.8)
    a.set_xlabel('$t$'); a.set_title(r'(a) optimal controls $\hat u$'); a.legend()
    a = ax[0, 1]; hV = []; hU = []
    for al, c, mk in [('0.3', 'C0', 'o'), ('0.05', 'C3', 's')]:
        r = d[al]; M = np.array(r['M'])
        l1, = a.loglog(M, r['EabsdV'], mk+'-', c=c, ms=3, label=r'$\mathbb{E}|V_M-V|$, $\alpha=%s$' % al); hV.append(l1)
        l2, = a.loglog(M, r['Eerr2'], mk+'--', c=c, ms=3, mfc='none', label=r'$\mathbb{E}\|\hat u_M-\hat u\|^2$, $\alpha=%s$' % al); hU.append(l2)
        a.loglog(M, np.sqrt(2*r['sig2V']/pi)/np.sqrt(M), ':', c=c, lw=.8)
        a.loglog(M, r['trSig']/M, ':', c=c, lw=.8)
    a.set_xlabel('$M$'); a.set_title('(b) errors (dotted: CLT predictions)')
    a.add_artist(a.legend(handles=hV, loc='upper right')); a.legend(handles=hU, loc='lower left')
    a = ax[0, 2]
    r = d['0.3']; x = np.linspace(-4, 4, 200)
    for key, lab, c in [('V', '$V$', 'k'), ('h1', r'$\langle\cdot,h_1\rangle$', 'C0'), ('h2', r'$\langle\cdot,h_2\rangle$', 'C2')]:
        if key == 'V': z = np.array(r['sqrtM_dV']['640'])/sqrt(r['sig2V'])
        else: z = np.array(r['z']['640'][key])/sqrt(r['pred'][key])
        a.hist(z, bins=30, range=(-4, 4), density=True, histtype='step', color=c, label=lab)
    a.plot(x, np.exp(-x**2/2)/sqrt(2*pi), 'r--', lw=.9, label=r'$\mathcal{N}(0,1)$')
    a.set_title(r'(c) histograms, $\alpha=0.3$, $M=640$'); a.legend()
    a = ax[1, 0]
    for al, c, mk in [('0.3', 'C0', 'o'), ('0.05', 'C3', 's')]:
        r = d[al]; M = np.array(r['M'])
        for k, ls in [('h1', '-'), ('h2', '--')]:
            a.semilogx(M, np.array(r['zvar'][k])/r['pred'][k], mk+ls, c=c, ms=3, mfc='none' if ls == '--' else c, label=r'$\alpha=%s$, $h_%s$' % (al, k[-1]))
    a.axhline(1, c='k', lw=.7); a.set_ylim(0.6, 1.9); a.set_xlabel('$M$'); a.set_title('(d) empirical / predicted variance'); a.legend()
    a = ax[1, 1]
    r = d['0.05']; ps = np.array(r['pstd']); u = np.array(r['ubar']); act = np.abs(np.abs(u)-1) < 1e-10
    for M, c in [('40', '0.7'), ('640', 'C1'), ('5120', 'C3')]:
        a.plot(t, r['devstd'][M], c=c, label=r'$M=%s$' % M)
    a.plot(t, ps, 'k--', lw=.9, label='predicted')
    for i in np.where(act)[0]: a.axvspan(i*dt, (i+1)*dt, color='0.9', lw=0)
    a.set_xlabel('$t$'); a.set_title(r'(e) std of $\sqrt{M}(\hat u_M-\hat u)(t)$, $\alpha=0.05$'); a.legend()
    a = ax[1, 2]
    for al, c, mk in [('0.3', 'C0', 'o'), ('0.05', 'C3', 's')]:
        r = d[al]; M = np.array(r['M'])
        for k, ls in [('h1', '-'), ('h2', '--')]:
            a.semilogx(M, r['coverage'][k], mk+ls, c=c, ms=3, mfc='none' if ls == '--' else c, label=r'$\alpha=%s$, $h_%s$' % (al, k[-1]))
    a.axhline(.95, c='k', lw=.7); a.set_ylim(0.75, 1.0); a.set_xlabel('$M$'); a.set_title('(f) coverage of 95% intervals'); a.legend(loc='lower right')
    plt.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 110})):
        plt.savefig(os.path.join(figdir, 'fig_oscillator.'+ext), **kw)
    print('  saved', os.path.join(figdir, 'fig_oscillator.pdf'), flush=True)
    if show: plt.show()
    plt.close(fig)


# =====================================================================
# 2. Sterile insect technique (SIT)
#
#    Dictionary between Bidi-Coron-Hayat-Lichtle (system (2.1)-(2.4), Table 1) and the model used here
#    (states A, F, M, S normalized by the carrying capacity K):
#      E  aquatic density                              A  aquatic stage
#      beta_E  = 8     oviposition rate                b    = 8
#      nu_E    = 0.25  hatching rate                   lam  = 0.25  (emergence rate)
#      delta_E = 0.03  aquatic death rate              dA   = 0.03
#      delta_F = 0.04  female death rate               dF   = 0.04
#      delta_M = 0.10  wild male death rate            dM   = 0.10
#      delta_s = 0.12  sterile male death rate         dS   = 0.12  (UNCERTAIN: truncated normal)
#      nu      = 0.49  prob. of emerging as female     rsex = 0.49
#      K       = 5e4   carrying capacity for eggs      states normalized by K
#      u_max   = 3e5   release bound (sterile/day)     CHI  = u_max/K = 6  (S' = CHI*u - dS*S, u in [0,1])
#      M/(M+M_s)                                       M/(M+q*S), q = 1 in the paper (UNCERTAIN: q ~ U[0.91,1])
#    Cost: E[F(T) + M(T)] + alpha/2 ||u||^2, T = 90 days, 0 <= u <= 1.
# =====================================================================
PAPER = dict(betaE=8.0, nuE=0.25, deltaE=0.03, deltaF=0.04, deltaM=0.10, deltas=0.12,
             nu=0.49, K=5e4, umax=3e5)

T = 90.0; N = 90; NS = 4                 # horizon (days), control intervals, substeps per interval
dt = T/N; h = dt/NS; tgrid = np.linspace(0, T, N+1); umax = 1.0
b = PAPER['betaE']; rsex = PAPER['nu']; CHI = PAPER['umax']/PAPER['K']
cc = 0.0                                  # linear release cost (omitted, see the paper)

NAMES = ['lam', 'dA', 'dF', 'dM', 'dS', 'q']
NOM = dict(lam=PAPER['nuE'], dA=PAPER['deltaE'], dF=PAPER['deltaF'], dM=PAPER['deltaM'],
           dS=PAPER['deltas'], q=0.955)
# Uncertainty model.  None = fixed at NOM.  Otherwise ('uniform',lo,hi) or ('truncnorm',mean,sd,lo,hi).
UNC = dict(lam=None, dA=None, dF=None, dM=None,
           dS=('truncnorm', PAPER['deltas'], 0.01, 0.10, 0.14),   # mean 0.12, sd 0.01, truncated to [0.10,0.14]
           q=('uniform', 0.91, 1.0))                              # q ~ U[0.91,1]

# Initial state: wild-population equilibrium without sterile males (S = 0).
Aeq = 1 - (NOM['dA']+NOM['lam'])*NOM['dF']/(b*rsex*NOM['lam'])
assert Aeq > 0, 'no positive equilibrium'
x0 = np.array([Aeq, rsex*NOM['lam']*Aeq/NOM['dF'], (1-rsex)*NOM['lam']*Aeq/NOM['dM'], 0.0])

UNC_NAMES = [n for n in NAMES if UNC[n] is not None]


def nodes_weights(spec, n):
    x, w = np.polynomial.legendre.leggauss(n)
    if spec[0] == 'uniform':
        lo, hi = spec[1], spec[2]
        return (lo+hi)/2 + (hi-lo)/2*x, w/2
    _, mu, sd, lo, hi = spec                       # truncated normal: Gauss-Legendre in the quantile variable
    a, bb = (lo-mu)/sd, (hi-mu)/sd
    return truncnorm.ppf((x+1)/2, a, bb, loc=mu, scale=sd), w/2


def quad(n):
    """Tensor quadrature over the uncertain parameters (n nodes each)."""
    nd = [nodes_weights(UNC[k], n) for k in UNC_NAMES]
    grids = np.meshgrid(*[v[0] for v in nd], indexing='ij')
    W = np.ones(())
    for v in nd: W = np.multiply.outer(W, v[1])
    Mq = W.size; gmap = {k: g.ravel() for k, g in zip(UNC_NAMES, grids)}
    P = tuple((gmap[k] if k in gmap else np.full(Mq, NOM[k]))[None, :] for k in NAMES)
    return P, W.ravel()[None, :]


def sample(rng, R, M):
    out = []
    for k in NAMES:
        s = UNC[k]
        if s is None: out.append(np.full((R, M), NOM[k]))
        elif s[0] == 'uniform': out.append(rng.uniform(s[1], s[2], (R, M)))
        else:
            _, mu, sd, lo, hi = s
            out.append(truncnorm.rvs((lo-mu)/sd, (hi-mu)/sd, loc=mu, scale=sd, size=(R, M), random_state=rng))
    return tuple(out)


def forward(U, P, keep=True):
    """Semi-implicit Euler with NS substeps per control interval: the stiff term of A is treated implicitly,
       A+ = (A + h*Rr)/(1 + h*(Rr + dA + lam)),  Rr = b*M*F/(M+q*S); then F, M use A+ and S is explicit."""
    lam, dA, dF, dM, dS, q = P
    R, M = lam.shape
    cF = rsex*lam; cM = (1-rsex)*lam; cA = dA+lam
    a = np.full((R, M), x0[0]); f = np.full((R, M), x0[1])
    m = np.full((R, M), x0[2]); s = np.full((R, M), x0[3])
    Xs = np.empty((N*NS+1, 4, R, M)) if keep else None
    if keep: Xs[0] = (a, f, m, s)
    for k in range(N):
        uk = U[:, k][:, None]
        for j in range(NS):
            Rr = b*m*f/(m + q*s)
            an = (a + h*Rr)/(1 + h*(Rr + cA))
            fn = f + h*(cF*an - dF*f); mn = m + h*(cM*an - dM*m); sn = s + h*(CHI*uk - dS*s)
            a, f, m, s = an, fn, mn, sn
            if keep: Xs[k*NS+j+1] = (a, f, m, s)
    return (a, f, m, s), Xs


def obj_grad(U, P, W, alpha, per_sample=False):
    """J = sum W [F(T)+M(T)] + cc*dt*sum(u) + alpha/2*dt*sum(u^2) and its gradient (coefficients),
       computed exactly for the discrete scheme by the discrete adjoint equation."""
    lam, dA, dF, dM, dS, q = P
    R, M = lam.shape
    (a, f, m, s), Xs = forward(U, P, keep=True)
    val = np.sum(W*(f+m), 1) + cc*dt*U.sum(1) + 0.5*alpha*dt*np.sum(U**2, 1)
    cF = rsex*lam; cM = (1-rsex)*lam; cA = dA+lam
    pa = np.zeros((R, M)); pf = np.ones((R, M)); pm = np.ones((R, M)); ps = np.zeros((R, M))
    G = np.empty((R, M, N)) if per_sample else None
    grad = np.empty((R, N))
    for k in range(N-1, -1, -1):
        acc = np.zeros((R, M))
        for j in range(NS-1, -1, -1):
            acc += h*(CHI*ps + cc)
            a_, f_, m_, s_ = Xs[k*NS+j]
            D = m_ + q*s_
            Rr = b*m_*f_/D
            Dn = 1 + h*(Rr + cA)
            gR = h*(1 - a_ + h*cA)/Dn**2            # d a+ / d Rr
            r_f = b*m_/D; r_m = b*f_*q*s_/D**2; r_s = -b*m_*f_*q/D**2
            pae = pa + h*(cF*pf + cM*pm)             # costate of A+ (F+ and M+ depend on A+)
            gp = gR*pae
            npa = pae/Dn
            npf = gp*r_f + (1 - h*dF)*pf
            npm = gp*r_m + (1 - h*dM)*pm
            nps = gp*r_s + (1 - h*dS)*ps
            pa, pf, pm, ps = npa, npf, npm, nps
        if per_sample: G[:, :, k] = acc
        grad[:, k] = np.sum(W*acc, 1)
    grad += alpha*dt*U
    if per_sample: return val, grad, G, (f+m)
    return val, grad


def solve_true(P, W, alpha, u0=None):
    if u0 is None: u0 = np.full(N, 0.05)
    fun = lambda u: tuple(np.asarray(v).ravel() if i else float(v[0]) for i, v in enumerate(obj_grad(u[None, :], P, W, alpha)))
    r = minimize(fun, u0, jac=True, method='L-BFGS-B', bounds=[(0, umax)]*N,
                 options={'ftol': 1e-16, 'gtol': 1e-12, 'maxiter': 5000, 'maxcor': 30})
    return r.x, r


def hessian_fd(u, P, W, alpha, eps=1e-6):
    """Hessian by central finite differences of the exact discrete gradient."""
    H = np.empty((N, N))
    for k in range(N):
        e = np.zeros(N); e[k] = eps
        H[:, k] = (obj_grad((u+e)[None, :], P, W, alpha)[1][0] - obj_grad((u-e)[None, :], P, W, alpha)[1][0])/(2*eps)
    return (H+H.T)/2


def explore(alphas, nq=12):
    """Quick look at the optimal control for several values of alpha (not used for the figures)."""
    Pq, Wq = quad(nq)
    for al in alphas:
        u, r = solve_true(Pq, Wq, al)
        H = hessian_fd(u, Pq, Wq, al); I = (u > 1e-9) & (u < umax-1e-9)
        print('alpha=%g  min u=%.3g  max u=%.3g  #up=%d #lo=%d  V=%.5f  mineigI=%.3g' %
              (al, u.min(), u.max(), (u > umax-1e-9).sum(), (u < 1e-9).sum(), r.fun,
               np.linalg.eigvalsh(H[np.ix_(I, I)]/dt).min() if I.any() else np.nan))


def run_alpha(alpha, Nrep, Ncov, MCOV, NQ, Ms, SMAX, logdir='.'):
    """Full experiment for one value of alpha (true problem, asymptotic quantities, Monte Carlo)."""
    Pq, Wq = quad(NQ); w = Wq[0]
    tests = {'h1': np.ones(N), 'h2': np.cos(np.pi*(tgrid[:-1]+dt/2)/T)}
    EPS = 1e-6

    def hessian_free(u, Pr, g0, idx):
        """Hessian of the SAA objective restricted to the free indices idx (one-sided finite differences
           of the exact discrete gradient); only the columns that are needed are computed."""
        M = Pr[0].shape[1]; chunk = max(1, SMAX//M); nf = len(idx); H = np.empty((nf, nf))
        for st in range(0, nf, chunk):
            kk = idx[st:st+chunk]; n = len(kk)
            Up = np.tile(u, (n, 1)); Up[np.arange(n), kk] += EPS
            P2 = tuple(np.repeat(p, n, axis=0) for p in Pr)
            g = obj_grad(Up, P2, np.full((n, M), 1.0/M), alpha)[1]
            H[:, st:st+n] = ((g[:, idx]-g0[idx])/EPS).T
        return (H+H.T)/2

    def coverage_rep(uM, Pr):
        """Plug-in variances <Sigma_M h, h> (Hessian inverted on the empirical inactive set)."""
        M = Pr[0].shape[1]
        _, g0, G, _ = obj_grad(uM[None, :], Pr, np.full((1, M), 1.0/M), alpha, per_sample=True)
        g0 = g0[0]; xiM = G[0]; xiM = xiM-xiM.mean(0); KM = xiM.T@xiM/M
        IM = ~((uM < 1e-10) | (uM > umax-1e-10)); idx = np.where(IM)[0]; res = {}
        HF = hessian_free(uM, Pr, g0, idx)
        for k, hf in tests.items():
            x = np.zeros(N); x[idx] = np.linalg.solve(HF, hf[idx])
            res[k] = float(dt**2*x@KM@x)
        return res

    logfile = os.path.join(logdir, 'progress_alpha_%g.txt' % alpha)

    def log(*args):                       # print and also append to a file (worker processes may not show prints)
        msg = ' '.join(str(x) for x in args); print('  [alpha=%g] ' % alpha + msg, flush=True)
        with open(logfile, 'a') as fh: fh.write(msg + '\n')
    open(logfile, 'w').close()
    t0 = time.time(); log('start')
    ubar, r = solve_true(Pq, Wq, alpha); V = float(r.fun)
    # uniqueness check by multistart
    rng = np.random.default_rng(7); dist = []; vals = []
    for s in range(10):
        u0 = rng.uniform(0, umax, N)*0.2 if s % 2 else np.full(N, rng.uniform(0, 0.2))
        us, rs = solve_true(Pq, Wq, alpha, u0)
        dist.append(float(np.sqrt(dt)*np.linalg.norm(us-ubar))); vals.append(float(rs.fun))
    Hmat = hessian_fd(ubar, Pq, Wq, alpha)
    _, d, Gs, gval = obj_grad(ubar[None, :], Pq, Wq, alpha, per_sample=True)
    d = d[0]; xi = Gs[0].T; gval = gval[0]
    up = ubar > umax-1e-9; lo = ubar < 1e-9; act = up | lo; I = ~act
    eigI = np.linalg.eigvalsh(Hmat[np.ix_(I, I)]/dt); eigfull = np.linalg.eigvalsh(Hmat/dt)
    log('#up', up.sum(), '#lo', lo.sum(), 'min u', ubar.min(), 'max u', ubar.max(),
        'mineigI', eigI.min(), 'multistart', max(dist))
    Eg = xi@w; Kcov = ((xi-Eg[:, None])*w[None, :])@(xi-Eg[:, None]).T
    Bop = np.zeros((N, N)); Bop[np.ix_(I, I)] = np.linalg.inv(Hmat[np.ix_(I, I)]); Sigma = Bop@Kcov@Bop
    sig2V = float(np.sum(w*gval**2)-np.sum(w*gval)**2)
    pred = {k: float(dt**2*hf@Sigma@hf) for k, hf in tests.items()}
    _, Xs = forward(ubar[None, :], Pq, keep=True); Xs = Xs[::NS, :, 0, :]; mt1 = np.einsum('s,tis->ti', w, Xs)
    _, Xs = forward(np.zeros((1, N)), Pq, keep=True); Xs = Xs[::NS, :, 0, :]; mt0 = np.einsum('s,tis->ti', w, Xs)
    cdiag = float(np.mean(np.diag(Hmat)))

    def saa_batch(P, M, maxit=60, tol=1e-10):
        """Solve R SAA problems at once: projected Newton steps with the Hessian of the true problem,
           warm-started at hat u, with an L-BFGS-B fallback for the (rare) problems that do not converge."""
        R = P[0].shape[0]; W = np.full((R, M), 1.0/M)
        U = np.tile(ubar, (R, 1)); Vv = np.zeros(R); done = np.zeros(R, bool)
        for it in range(maxit):
            idx = np.where(~done)[0]
            if idx.size == 0: break
            Pi = tuple(p[idx] for p in P)
            val, g = obj_grad(U[idx], Pi, W[idx], alpha); Vv[idx] = val
            resid = np.max(np.abs(U[idx]-np.clip(U[idx]-g/cdiag, 0, umax)), 1)
            ok = resid < tol; done[idx[ok]] = True
            for jj in np.where(~ok)[0]:
                r_ = idx[jj]; dlt, _ = boxqp(Hmat, g[jj], -U[r_], umax-U[r_]); U[r_] += dlt
        for r_ in np.where(~done)[0]:
            u1, rr = solve_true(tuple(p[r_:r_+1] for p in P), W[r_:r_+1], alpha, U[r_]); U[r_] = u1; Vv[r_] = rr.fun
        return U, Vv, int((~done).sum())

    rec = {M: {'dV': [], 'err2': [], 'z': {k: [] for k in tests}} for M in Ms}; dev = {}
    cov = {M: {k: [] for k in tests} for M in MCOV}; nfail = 0
    rng = np.random.default_rng(2026)
    for M in Ms:
        Rb = max(1, SMAX//M); done_reps = 0
        while done_reps < Nrep:
            R = min(Rb, Nrep-done_reps); P = sample(rng, R, M)
            U, Vv, nf = saa_batch(P, M); nfail += nf
            E = U-ubar
            rec[M]['dV'] += list(Vv-V); rec[M]['err2'] += list(dt*np.sum(E**2, 1))
            for k, hf in tests.items(): rec[M]['z'][k] += list(np.sqrt(M)*dt*(E@hf))
            if M in (40, 640, 2560): dev.setdefault(M, []).append(np.sqrt(M)*E)
            if M in MCOV:
                for rr in range(R):
                    if len(cov[M]['h1']) >= Ncov: break
                    v = coverage_rep(U[rr], tuple(p[rr:rr+1] for p in P))
                    zz = {k: np.sqrt(M)*dt*(E[rr]@tests[k]) for k in tests}
                    for k in tests: cov[M][k].append(bool(abs(zz[k]) <= 1.959964*np.sqrt(v[k])))
            done_reps += R
        log('M', M, 'done', round(time.time()-t0), 's,  sqrt(M) E|V_M-V| =', np.sqrt(M)*np.mean(np.abs(rec[M]['dV'])))
    dev = {M: np.vstack(v) for M, v in dev.items()}
    res = {
        'alpha': alpha, 'V': V, 'sig2V': sig2V, 'pred': pred, 'trSig': float(dt*np.trace(Sigma)),
        'ubar': ubar.tolist(), 'pstd': np.sqrt(np.diag(Sigma)).tolist(),
        'multistart_maxdist': max(dist), 'multistart_valspread': max(vals)-min(vals),
        'mineigI': float(eigI.min()), 'mineigfull': float(eigfull.min()),
        'nup': int(up.sum()), 'nlo': int(lo.sum()), 'umin': float(ubar.min()), 'umax': float(ubar.max()),
        'mind_act': float(np.abs(d[act]).min()/dt) if act.any() else None, 'fallbacks': nfail,
        'EF_u': mt1[:, 1].tolist(), 'EF_0': mt0[:, 1].tolist(),
        'EabsdV': [float(np.mean(np.abs(rec[M]['dV']))) for M in Ms],
        'Eerr2': [float(np.mean(rec[M]['err2'])) for M in Ms],
        'zvar': {k: [float(np.var(rec[M]['z'][k])) for M in Ms] for k in tests},
        'sqrtM_dV': {M: (np.sqrt(M)*np.array(rec[M]['dV'])).tolist() for M in (40, 640, 2560) if M in rec},
        'z': {M: {k: rec[M]['z'][k] for k in tests} for M in (40, 640, 2560) if M in rec},
        'devstd': {M: np.std(dev[M], 0).tolist() for M in dev},
        'coverage': {k: [float(np.mean(cov[M][k])) if cov[M][k] else None for M in MCOV] for k in tests}}
    log('done in', round(time.time()-t0), 's')
    return str(alpha), res


def run_sit(Nrep=500, Ncov=200, MCOV=(20, 40, 80, 160, 320, 640, 1280), ALPHAS=(0.1, 0.03), NQ=20, Ms=None,
            SMAX=20000, parallel=False, outdir='results'):
    Ms = Ms or [10, 20, 40, 80, 160, 320, 640, 1280, 2560]
    t_start = time.time()
    args = [(al, Nrep, Ncov, MCOV, NQ, Ms, SMAX, outdir) for al in ALPHAS]
    if parallel and len(ALPHAS) > 1:                      # one process per value of alpha
        import multiprocessing as mp
        with mp.get_context('fork').Pool(len(ALPHAS)) as pool: results = pool.starmap(run_alpha, args)
    else:
        results = [run_alpha(*a) for a in args]
    out = {'Nrep': Nrep, 'Ncov': Ncov, 'M': Ms, 'MCOV': list(MCOV), 'NQ': NQ, 'cc': cc,
           'uncertain': UNC_NAMES, 'NS': NS, 'alphas': dict(results)}
    out['tgrid'] = tgrid.tolist()
    out['meta'] = metadata(example='sit', seeds={'multistart': 7, 'sampling': 2026},
                           runtime_s=round(time.time()-t_start))
    path = os.path.join(outdir, 'sit_paper.json')
    with open(path, 'w') as fh: json.dump(out, fh)
    print('  saved', path, flush=True)
    return out


def plot_sit(json_path, figdir='figures', show=False):
    plt = pyplot(show)
    with open(json_path) as fh: D = json.load(fh)
    ALPHAS = [float(k) for k in D['alphas']]          # (inactive, active), in the order of the run
    t = tgrid[:-1]+dt/2
    al_in, al_ac = list(D['alphas'])
    d_in, d_ac = D['alphas'][al_in], D['alphas'][al_ac]
    col = {al_in: 'C0', al_ac: 'C3'}
    lab = {k: r'$\alpha=%g$' % float(k) for k in D['alphas']}
    M = np.array(D['M']); Mc = np.array(D['MCOV'])
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6))
    a = ax[0, 0]
    for k in (al_in, al_ac):
        u = np.array(D['alphas'][k]['ubar']); a.step(tgrid, np.r_[u, u[-1]], where='post', color=col[k], label=lab[k])
    a.axhline(umax, ls=':', c='k', lw=.8); a.set_xlabel('$t$ (days)'); a.set_title(r'(a) optimal controls $\hat u$'); a.legend()
    a = ax[0, 1]; hV = []; hU = []
    for k, mk in ((al_in, 'o'), (al_ac, 's')):
        r = D['alphas'][k]
        l1, = a.loglog(M, r['EabsdV'], mk+'-', c=col[k], ms=3, label=r'$\mathbb{E}|V_M-V|$, '+lab[k]); hV.append(l1)
        l2, = a.loglog(M, r['Eerr2'], mk+'--', c=col[k], ms=3, mfc='none', label=r'$\mathbb{E}\|\hat u_M-\hat u\|^2$, '+lab[k]); hU.append(l2)
        a.loglog(M, np.sqrt(2*r['sig2V']/pi)/np.sqrt(M), ':', c=col[k], lw=.8); a.loglog(M, r['trSig']/M, ':', c=col[k], lw=.8)
    a.set_xlabel('$M$'); a.set_title('(b) errors (dotted: CLT predictions)')
    a.add_artist(a.legend(handles=hV, loc='upper right')); a.legend(handles=hU, loc='lower left')
    a = ax[0, 2]; r = d_in; x = np.linspace(-4, 4, 200)
    for key, l, c in [('V', '$V$', 'k'), ('h1', r'$\langle\cdot,h_1\rangle$', 'C0'), ('h2', r'$\langle\cdot,h_2\rangle$', 'C2')]:
        z = np.array(r['sqrtM_dV']['640'])/sqrt(r['sig2V']) if key == 'V' else np.array(r['z']['640'][key])/sqrt(r['pred'][key])
        a.hist(z, bins=30, range=(-4, 4), density=True, histtype='step', color=c, label=l)
    a.plot(x, np.exp(-x**2/2)/sqrt(2*pi), 'r--', lw=.9, label=r'$\mathcal{N}(0,1)$')
    a.set_title(r'(c) histograms, $\alpha=%g$, $M=640$' % ALPHAS[0]); a.legend()
    a = ax[1, 0]
    for k, mk in ((al_in, 'o'), (al_ac, 's')):
        r = D['alphas'][k]
        for hh, ls in (('h1', '-'), ('h2', '--')):
            a.semilogx(M, np.array(r['zvar'][hh])/r['pred'][hh], mk+ls, c=col[k], ms=3, mfc='none' if ls == '--' else col[k], label=lab[k]+r', $h_%s$' % hh[-1])
    a.axhline(1, c='k', lw=.7); a.set_ylim(0.5, 1.7); a.set_xlabel('$M$'); a.set_title('(d) empirical / predicted variance'); a.legend()
    a = ax[1, 1]; r = d_ac; ub = np.array(r['ubar']); act = (ub > umax-1e-9) | (ub < 1e-9)
    for Mv, c in [('40', '0.7'), ('640', 'C1'), ('2560', 'C3')]: a.plot(t, r['devstd'][Mv], c=c, label=r'$M=%s$' % Mv)
    a.plot(t, r['pstd'], 'k--', lw=.9, label='predicted')
    for i in np.where(act)[0]: a.axvspan(tgrid[i], tgrid[i+1], color='0.9', lw=0)
    a.set_xlabel('$t$ (days)'); a.set_title(r'(e) std of $\sqrt{M}(\hat u_M-\hat u)(t)$, $\alpha=%g$' % ALPHAS[1]); a.legend()
    a = ax[1, 2]
    for k, mk in ((al_in, 'o'), (al_ac, 's')):
        r = D['alphas'][k]
        for hh, ls in (('h1', '-'), ('h2', '--')):
            a.semilogx(Mc, r['coverage'][hh], mk+ls, c=col[k], ms=3, mfc='none' if ls == '--' else col[k], label=lab[k]+r', $h_%s$' % hh[-1])
    a.axhline(.95, c='k', lw=.7); a.set_ylim(0.8, 1.0); a.set_xlabel('$M$')
    a.set_xticks(Mc); a.set_xticklabels([str(int(v)) for v in Mc], rotation=45); a.minorticks_off()
    a.set_title('(f) coverage of 95% intervals'); a.legend(loc='lower right')
    plt.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 110})):
        plt.savefig(os.path.join(figdir, 'fig_sit_paper.'+ext), **kw)
    print('  saved', os.path.join(figdir, 'fig_sit_paper.pdf'), flush=True)
    if show: plt.show()
    plt.close(fig)


# =====================================================================
# 3. Command line
# =====================================================================
SETTINGS = {   # 'paper': replications used in the paper; 'quick': reduced, for testing the installation
    'paper': dict(osc_Nrep=1000, sit_Nrep=500, Ncov=200, MCOV=(20, 40, 80, 160, 320, 640, 1280)),
    'quick': dict(osc_Nrep=100, sit_Nrep=50, Ncov=30, MCOV=(40, 160, 640)),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('Usage')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', choices=['none', 'oscillator', 'sit', 'all'], default='none',
                   help="experiments to rerun (default: none, only redraw the figures from the saved results)")
    p.add_argument('--mode', choices=['paper', 'quick'], default='paper',
                   help="'paper': replications used in the paper; 'quick': fewer replications, for testing")
    p.add_argument('--outdir', default=os.path.join(HERE, 'results'), help='folder for the .json results')
    p.add_argument('--figdir', default=os.path.join(HERE, 'figures'), help='folder for the figures')
    p.add_argument('--parallel', action='store_true',
                   help='SIT: one process per value of alpha (uses fork; not available on Windows)')
    p.add_argument('--show', action='store_true', help='also display the figures on screen')
    args = p.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True); os.makedirs(args.figdir, exist_ok=True)
    S = SETTINGS[args.mode]
    print('python %s | numpy %s | scipy %s' % (platform.python_version(), np.__version__, scipy.__version__))
    t0 = time.time()
    if args.run in ('oscillator', 'all'):
        print('Oscillator (%s mode, %d replications)...' % (args.mode, S['osc_Nrep']), flush=True)
        run_oscillator(S['osc_Nrep'], outdir=args.outdir)
    if args.run in ('sit', 'all'):
        print('SIT (%s mode, %d replications)...' % (args.mode, S['sit_Nrep']), flush=True)
        run_sit(S['sit_Nrep'], S['Ncov'], S['MCOV'], parallel=args.parallel, outdir=args.outdir)
    print('Figures...', flush=True)
    for name, plot in (('oscillator.json', plot_oscillator), ('sit_paper.json', plot_sit)):
        path = os.path.join(args.outdir, name)
        if os.path.exists(path):
            plot(path, args.figdir, args.show)
        else:
            print('  %s not found; run with --run first' % path)
    print('Done in %.0f s.' % (time.time()-t0))


if __name__ == '__main__':
    main()
