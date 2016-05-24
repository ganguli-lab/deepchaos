import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import hypsecant
from scipy.stats import norm



epsabs = 1e-2
epsrel = 1e-2
dz = 0.05
zmin = -10.0
zmax = 10.0

def fast_integral(integrand, zmin, zmax, dz, ndim=1):
    zs = np.r_[zmin:zmax:dz]
    if ndim > 1:
        zgrid = np.meshgrid(*((zs,) * ndim))
    else:
        zgrid = (zs,)
    out = integrand(*zgrid)
    return out.sum(tuple(np.arange(ndim))) * dz**ndim


def qmap(qin, weight_sigma=1.0 , bias_sigma=0.0, nonlinearity=np.tanh,
         epsabs=epsabs, epsrel=epsrel, zmin=-10, zmax=10, dz=dz, fast=True):
    qin = np.atleast_1d(qin)
    # Perform Gaussian integral
    def integrand(z):
        return norm.pdf(z[:, None]) * nonlinearity(np.sqrt(qin[None, :]) * z[:, None])**2
    integral = fast_integral(integrand, zmin, zmax, dz=dz)
    return weight_sigma**2 * integral + bias_sigma**2

def compute_chi1(qstar, weight_sigma=1.0, bias_sigma=0.01, dphi=np.tanh):
    def integrand(z):
        return norm.pdf(z) * dphi(np.sqrt(qstar) * z)**2
    integral = quad(integrand, zmin, zmax, epsabs=epsabs, epsrel=epsrel)[0]
    return weight_sigma**2 * integral

def compute_chi2(qstar, weight_sigma=1.0, bias_sigma=0.01, d2phi=np.tanh, hidden_units=100):
    def integrand(z):
        return norm.pdf(z) * d2phi(np.sqrt(qstar) * z)**2
    integral = quad(integrand, zmin, zmax, epsabs=epsabs, epsrel=epsrel)[0]
    return  weight_sigma**2 * integral  / hidden_units# np.sqrt(din)

def kappa_map(kappa, chi1, chi2):
    return 3 * chi2 / chi1**2 + 1/chi1 * kappa

def covmap(q1, q2, q12, weight_sigma, bias_sigma, nonlinearity=np.tanh, zmin=-10, zmax=10, dz=dz, fast=True):
    q1 = np.atleast_1d(q1)
    q2 = np.atleast_1d(q2)
    q12 = np.atleast_1d(q12)

    u1 = np.sqrt(q1)
    u2 = q12 / np.sqrt(q1)
    # XXX: tolerance fudge factor
    u3 = np.sqrt(q2 - q12**2 / q1 + 1e-8)
    def integrand(z1, z2):
        return norm.pdf(z1[..., None]) * norm.pdf(z2[..., None]) * (
            nonlinearity(u1[None, None, :] * z1[..., None]) *
            nonlinearity(u2[None, None, :] * z1[..., None] + u3[None, None, :] * z2[..., None]))
    integral = fast_integral(integrand, zmin, zmax, dz, ndim=2)
    return weight_sigma**2 * integral + bias_sigma**2

def q_fixed_point(weight_sigma, bias_sigma, nonlinearity, max_iter=500, tol=1e-9, qinit=3.0, fast=True, tol_frac=0.01):
    """Compute fixed point of q map"""
    q = qinit
    qs = []
    for i in xrange(max_iter):
        qnew = qmap(q, weight_sigma, bias_sigma, nonlinearity, fast=fast)
        err = np.abs(qnew - q)
        qs.append(q)
        if err < tol:
            break
        q = qnew
    # Find first time it gets within tol_frac fracitonal error of q*
    frac_err = (np.array(qs) - q)**2 / (1e-9 + q**2)
    t = np.flatnonzero(frac_err < tol_frac)[0]
    return t, q
