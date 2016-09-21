#define INF 1E20
#include <math.h>
#include <sys/types.h>

/*
 * shiftdt.h
 * Generalized distance transforms based on Felzenswalb and Huttenlocher.
 * This applies a min convolution of an arbitrary quadratic function ax^2 + bx.
 * It results on an shifted, subsampled grid (useful for passing messages between
 * variables in different domains).
 */

static inline int square(int x) { return x*x; }

// dt1d(source,destination_val,destination_ptr,source_step,source_length,
//      a,b,dest_shift,dest_length,dest_step)
void dt1d(double *src, double *dst, int *ptr, int step, int len, double a,
          double b, int dshift, int dlen, double dstep) {
  int   *v = new int[len];
  float *z = new float[len+1];
  int k = 0, q, diff;
  float s;

  v[0] = 0;
  z[0] = -INF;
  z[1] = +INF;

  for (q = 1; q <= len-1; q++) {
    diff = q - v[k];
    s  = ((src[q * step] - src[v[k] * step]) - b * diff + a * (square(q) - square(v[k]))) / (2 * a * diff);
    while (s <= z[k]) {
      k--;
      diff = q - v[k];
      s  = ((src[q * step] - src[v[k] * step]) - b * diff + a * (square(q) - square(v[k]))) / (2 * a * diff);
    }
    k++;
    v[k]   = q;
    z[k]   = s;
    z[k+1] = +INF;
  }

   k = 0;
   q = dshift;

   for (int i=0; i <= dlen-1; i++) {
     while (z[k+1] < q)
       k++;
     diff = q - v[k];
     dst[i * step] = a * square(diff) + b * diff + src[v[k] * step];
     ptr[i * step] = v[k];
     q += dstep;
  }

  delete [] v;
  delete [] z;
}
