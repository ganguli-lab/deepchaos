import numpy as np
from tqdm import tqdm

def local_curvature(g, h):
    normg2 = (g**2).sum(1)
    normh2 = (h**2).sum(1)
    return (normh2 / normg2**2) - (g * h).sum(1)**2 / normg2**3

def global_curvature_term(g, h):
    return np.sqrt(local_curvature(g, h)) * np.linalg.norm(g, axis=1)

def get_second_order_coeff(x, y):
    z = np.polyfit(x, y, 2)
    return -z[0]

def compute_curvature_statistics(net, ts_fine, ts_coarse, weight_sigmas, bias_sigmas, qstar=None, include_hessian=False, randomize=True):
    nw = len(weight_sigmas)
    nb = len(bias_sigmas)
    n_interp = len(ts_fine)
    n_hidden_layers = net.n_hidden_layers
    n_hidden_units = net.n_hidden_units
    n_coarse = len(ts_coarse)
    acts = np.zeros((len(weight_sigmas), len(bias_sigmas), n_coarse, n_hidden_layers, n_interp, n_hidden_units))
    jacs = np.zeros_like(acts)
    hessians = np.zeros_like(jacs)


    nw, nb = len(weight_sigmas), len(bias_sigmas)
    r_jacs = np.zeros((nw, nb, n_interp, n_hidden_layers))
    r_acts = np.zeros_like(r_jacs)
    r_acts_centered = np.zeros_like(r_jacs)
    r_acts_std = np.zeros_like(r_acts)
    r_acts_centered_std = np.zeros_like(r_acts_centered)
    r_jacs_std = np.zeros_like(r_jacs)

    local_curvatures = np.zeros((nw, nb, n_interp, n_hidden_layers))
    global_curvatures = np.zeros_like(local_curvatures)
    second_deriv_autocorr = np.zeros((nw, nb, n_hidden_layers))

    corr_matrix_acts = np.zeros((nw, nb, len(ts_coarse), n_hidden_layers, n_interp, n_interp))
    corr_matrix_acts_centered = np.zeros_like(corr_matrix_acts)
    corr_matrix_jacs = np.zeros_like(corr_matrix_acts)

    # Compute activations and derivatives for each weight sigma
    for widx, weight_sigma in enumerate(tqdm(weight_sigmas)):
        for bidx, bias_sigma in enumerate(bias_sigmas):
            if qstar is not None:
                # Scale input circle so the norm of Wx+b at the first layer is q*
                scale = np.sqrt((float(net.input_dim) *  qstar[widx, bidx] -  bias_sigma**2) / weight_sigma**2)
            else:
                scale = 1.0
            net.input_layer.set_scale(scale)
            for tidx, t_off in enumerate(ts_coarse):
                if randomize:
                    net.randomize(bias_sigma, weight_sigma)
                cur_ts = ts_fine[:, None] + t_off
                out = net.get_acts_and_derivatives(cur_ts, include_hessian=include_hessian)
                # Note if include_hessian is False, then cur_hessians will be cur_jacs, the Jacobians
                cur_acts, cur_jacs, cur_hessians = out[:n_hidden_layers], out[n_hidden_layers:2*n_hidden_layers], out[-n_hidden_layers:]
                for lidx, act in enumerate(cur_acts[:-1]):
                    acts[widx, bidx, tidx, lidx] = cur_acts[lidx]
                    jacs[widx, bidx, tidx, lidx] = cur_jacs[lidx]
                    hessians[widx, bidx, tidx, lidx] = cur_hessians[lidx]

    acts_centered = acts - acts.mean(-2, keepdims=True)

    # Compute some useful statistics
    out = {}
    for widx in xrange(nw):
        for bidx in xrange(nb):
            for tidx in xrange(n_coarse):
                for lidx in xrange(n_hidden_layers):
                    cur_acts  = acts[widx, bidx, tidx, lidx]
                    cur_jacs  = jacs[widx, bidx, tidx, lidx]
                    cur_hessians = hessians[widx, bidx, tidx, lidx]

                    # Compute curvature metrics
                    if include_hessian:
                        local_curvatures[widx, bidx, :, lidx] += local_curvature(cur_jacs, cur_hessians) / n_coarse
                        global_curvatures[widx, bidx, :, lidx] += global_curvature_term(cur_jacs, cur_hessians) / n_coarse

                    # Compute autocorrelations
                    for r_, rs, corr_matrix, r_std in (
                            (r_acts, acts, corr_matrix_acts, r_acts_std),
                            (r_jacs, jacs, corr_matrix_jacs, r_jacs_std),
                            (r_acts_centered, acts_centered, corr_matrix_acts_centered, r_acts_centered_std)):
                        V = rs[widx, bidx, tidx, lidx]
                        V = V / np.linalg.norm(V, axis=1)[:, None]
                        R = np.dot(V, V.T)
                        corr_matrix[widx, bidx, tidx, lidx] = R
                        Rstack = R
                        for i in xrange(n_interp):
                            d = np.diag(Rstack, i-n_interp/2)
                            r_[widx, bidx, i, lidx] += d.mean()/len(ts_coarse)
                            #XXX: computing std within one network and then averaging (mean of std)
                            # could look at std of mean across networks instead
                            r_std[widx, bidx, i, lidx] += d.std() / len(ts_coarse)
            second_deriv_autocorr[widx, bidx] = [get_second_order_coeff(ts_fine, y) for y in r_jacs[widx, bidx].T]

    global_curvatures_raw = global_curvatures.copy()
    global_curvatures = global_curvatures.mean(-2)
    return dict(locals())

