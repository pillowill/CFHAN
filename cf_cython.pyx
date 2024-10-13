# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import cython
import numpy as pnp
from scipy.spatial.distance import cdist, cosine
cimport numpy as cnp
from cython.parallel cimport prange

cdef double calc_cf(int a, int b, double thresh,int[:] nns_a, int[:] nns_b,
                    double[:,:] simi_mat, double[:,:] T_f,
                    double[:,:] adj, double[:,:] T_cf, double[:,:] adj_cf)nogil:

    cdef int i = 0
    cdef int j = 0

    while i < nns_a.shape[0] - 1 and j < nns_b.shape[0] - 1:
        if simi_mat[a, nns_a[i]] + simi_mat[nns_b[j], b] > 2 * thresh:
            T_cf[a, b] = T_f[a, b]
            adj_cf[a, b] = adj[a, b]
            break
        if T_f[nns_b[j], nns_a[i]] != T_f[a, b]:
            T_cf[a, b] = 1 - T_f[a, b]
            adj_cf[a, b] = adj[nns_b[j], nns_a[i]]
            break
        if simi_mat[a, nns_a[i + 1]] < simi_mat[nns_b[j + 1], b]:
            i += 1
        else:
            j += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def get_CF_cython(double[:,:] adj, double[:,:] lnc_node_embs,
                  double[:,:] dis_node_embs, double[:,:] T_f,
                  str dist='euclidean', double thresh=0.5):

    cdef int a, b
    cdef double[:,:] simi_mat = cdist(lnc_node_embs, dis_node_embs, dist)

    cdef double[:,:] T_cf = pnp.zeros_like(adj)
    cdef double[:,:] adj_cf = pnp.zeros_like(adj)

    cdef int[:,:] lnc_node_nns = pnp.argsort(simi_mat, axis=1).astype(pnp.int32)
    cdef int[:,:] dis_node_nns = pnp.argsort(simi_mat.T, axis=1).astype(pnp.int32)

    thresh = pnp.percentile(simi_mat, thresh)

    for a in prange(adj.shape[0], nogil=True):
        for b in range(adj.shape[1]):
            with gil:
                nns_a = lnc_node_nns[a]
                nns_b = dis_node_nns[b]
                calc_cf(a, b, thresh, nns_a, nns_b, simi_mat, T_f, adj, T_cf, adj_cf)

    return pnp.array(T_cf), pnp.array(adj_cf)