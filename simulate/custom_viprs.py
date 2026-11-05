"""VIPRS implementation (from A1).
Spike-and-slab, mean-field VI.
E-step runs coordinate ascent within each LD block; M-step updates tau_beta and pi.
"""

import numpy as np

def _e_step(beta_marg, groups, Rb, n, tau_eps, tau_beta, pi, mu, tau, gamma, q):
    for g, Rg in zip(groups, Rb):
        qg = q[g].copy()
        Rq = Rg @ qg
        for j in range(len(g)):
            Rq -= Rg[:, j] * qg[j]  # leave variant j out
            tau_j = n * Rg[j, j] * tau_eps + tau_beta
            mu_j = n * tau_eps / tau_j * (beta_marg[g[j]] - Rq[j])

            u = (np.log(pi / (1 - pi)) + 0.5 * np.log(tau_beta / tau_j) + 0.5 * tau_j * mu_j ** 2)

            gamma_j = 1.0 / (1.0 + np.exp(-np.clip(u, -30, 30)))
            tau[g[j]], mu[g[j]], gamma[g[j]] = tau_j, mu_j, gamma_j
            qg[j] = gamma_j * mu_j
            Rq += Rg[:, j] * qg[j]
        q[g] = qg
    return mu, tau, gamma, q

def _m_step(mu, tau, gamma):
    tau_beta = gamma.sum() / ((gamma * (mu ** 2 + 1.0 / tau)).sum() + 1e-9)
    pi = gamma.mean()
    return float(np.clip(tau_beta, 1e-3, 1e8)), float(np.clip(pi, 1e-4, 0.5))

def custom_viprs(beta_marg, R, n, block_of_snp=None, h2=0.6, pi=0.01, n_iter=120, tol=1e-7, return_state=False):
    m = len(beta_marg)
    tau_eps = 1.0 / (1.0 - h2)
    tau_beta = (m * pi) / h2

    if block_of_snp is None:
        block_of_snp = np.zeros(m, dtype=int)
    groups = [np.where(block_of_snp == b)[0] for b in np.unique(block_of_snp)]
    Rb = [R[np.ix_(g, g)] for g in groups]

    mu, gamma = np.zeros(m), np.full(m, pi)
    tau = np.full(m, tau_beta)
    q = gamma * mu

    for _ in range(n_iter):
        q_old = q.copy()
        mu, tau, gamma, q = _e_step(beta_marg, groups, Rb, n, tau_eps, tau_beta, pi, mu, tau, gamma, q)
        tau_beta, pi = _m_step(mu, tau, gamma)
        if np.max(np.abs(q - q_old)) < tol:
            break

    if return_state:
        return q, dict(mu=mu, tau=tau, gamma=gamma, tau_beta=tau_beta, pi=pi, tau_eps=tau_eps, groups=groups, Rb=Rb)

    return q

def elbo(beta_marg, n, state, eps=1e-12):
    mu, tau, g = state["mu"], state["tau"], np.clip(state["gamma"], eps, 1 - eps)
    tb, pi, te = state["tau_beta"], state["pi"], state["tau_eps"]
    q = g * mu
    quad, diag_part = 0.0, 0.0

    for idx, Rg in zip(state["groups"], state["Rb"]):
        qg = q[idx]
        quad += 0.5 * (qg @ Rg @ qg - np.sum(qg ** 2 * np.diag(Rg)))
        diag_part += np.sum(g[idx] * (mu[idx] ** 2 + 1 / tau[idx]) * np.diag(Rg))

    lik = (n / 2 * np.log(te) - n / 2 * np.log(2 * np.pi) - te / 2 * n
           + te * n * np.sum(q * beta_marg) - te / 2 * n * diag_part - te * n * quad)

    prior_beta = np.sum(g * (0.5 * np.log(tb / (2 * np.pi)) - 0.5 * tb * (mu ** 2 + 1 / tau)))
    q_beta = np.sum(g * (0.5 * np.log(tau / (2 * np.pi)) - 0.5))
    prior_s = np.sum(g * np.log(pi) + (1 - g) * np.log(1 - pi))
    q_s = np.sum(g * np.log(g) + (1 - g) * np.log(1 - g))
    
    return lik + prior_beta - q_beta + prior_s - q_s