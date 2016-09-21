import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector

cdef extern from "cpp/shiftdt.h":
    void dt1d(double *src, double *dst, int *ptr, int step,
              int len, double a, double b, int dshift, int dlen, double dstep)

cdef int eps = 10 ** (-7)

def distance_transform_cython(double[:, :] score, double[:] w, int startx,
                              int starty, int Nx, int Ny, int step):
    cdef int sizy = score.shape[0]
    cdef int sizx = score.shape[1]
    cdef double ax, bx, ay, by
    ax, bx, ay, by = w
    assert(np.abs(ax) > eps)  # ax, ay should be non-zero.
    assert(np.abs(ay) > eps)
    startx -= 1   # due to python numbering (conversion to c).
    starty -= 1   # due to python numbering (conversion to c).

    cdef np.ndarray[double, ndim=2] M = np.zeros((Ny, Nx), dtype=np.double)
    cdef np.ndarray[int, ndim=2] Ix = np.empty((Ny, Nx), dtype=np.int32)
    cdef np.ndarray[int, ndim=2] Iy = np.empty((Ny, Nx), dtype=np.int32)
    cdef np.ndarray[double, ndim=1] tmpM = np.zeros((Nx * sizy, ), dtype=np.double)
    cdef np.ndarray[int, ndim=1] tmpIx = np.zeros((Nx * sizy, ), dtype=np.int32)
    cdef int x, y

    for y in range(sizy):
        dt1d(&score[y, 0], &tmpM[y*Nx], &tmpIx[y*Nx], 1, sizx, ax, bx, startx, Nx, step)

    for x in range(Nx):
        dt1d(&tmpM[x], &M[0,x], &Iy[0, x], Nx, sizy, ay, by, starty, Ny, step)

    for y in range(Ny):
        for x in range(Nx):
            Ix[y, x] = tmpIx[Iy[y, x]*Nx + x]

    return M, Ix, Iy
