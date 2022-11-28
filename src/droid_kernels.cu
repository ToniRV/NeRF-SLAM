#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

// #include "utils.cuh"

#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
typedef std::vector<std::vector<long>> graph_t;
typedef std::vector<torch::Tensor> tensor_list_t;



#define MIN_DEPTH 0.25

#define THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + THREADS - 1) / THREADS)


#define GPU_1D_KERNEL_LOOP(k, n) \
  for (size_t k = threadIdx.x; k<n; k += blockDim.x)


__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid +  8];
  sdata[tid] += sdata[tid +  4];
  sdata[tid] += sdata[tid +  2];
  sdata[tid] += sdata[tid +  1];
}

__device__ void blockReduce(volatile float *sdata) {
  unsigned int tid = threadIdx.x;
  __syncthreads();

  // if (threadIdx.x < 256) {sdata[tid] += sdata[tid + 256]; } __syncthreads();
  if (threadIdx.x < 128) {sdata[tid] += sdata[tid + 128]; } __syncthreads();
  if (threadIdx.x <  64) {sdata[tid] += sdata[tid +  64]; } __syncthreads();

  if (tid < 32) warpReduce(sdata, tid);
  __syncthreads();
}


//    EIGEN_DEVICE_FUNC Point operator*(Point const& p) const {
//      const Quaternion& q = unit_quaternion;
//      Point uv = q.vec().cross(p);
//      uv += uv;
//      return p + q.w()*uv + q.vec().cross(uv);
//    }
// Mind that sometimes I pass Y == X, so don't use X[i] after editing Y[i]
// since they might be at the same memory location...
__device__ void
actSO3(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

__device__  void
actSE3(const float *t, const float *q, const float *X, float *Y) {
  actSO3(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}

// Sometimes we send X == Y
__device__ void
adjSE3(const float *t, const float *q, const float *X, float *Y) {
  float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  actSO3(qinv, &X[0], &Y[0]); // Calculates [X0, X1, X2] * R_ij = (R_ij^t *[X0,X1,X2]^t)^t (upper left of Ji * Adj_ij)
  actSO3(qinv, &X[3], &Y[3]); // Calculates [X3, X4, X5] * R_ij (lower right of Ji * Adj_ij)

  // Calculates [X0, X1, X3] * [t]_x (aka cross product of X0..3 and t)
  float u[3], v[3]; 
  u[0] = t[2]*X[1] - t[1]*X[2];
  u[1] = t[0]*X[2] - t[2]*X[0];
  u[2] = t[1]*X[0] - t[0]*X[1];

  // Last rotation multiplication, and obtain upper right of Ji * Adj_ij
  actSO3(qinv, u, v);
  Y[3] += v[0];
  Y[4] += v[1];
  Y[5] += v[2];
}

// Computes: Gi, Gj -> Gij = Gj * Gi^{-1}
__device__ void 
relSE3(const float *ti, const float *qi, const float *tj, const float *qj, float *tij, float *qij) {
  // Hamilton product of two quaternions, qj is inverted. TODO: check this math...
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0], 
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2], 

  actSO3(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];
}

  
__device__ void
expSO3(const float *phi, float* q) {
  // SO3 exponential map
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta_p4 = theta_sq * theta_sq;

  float theta = sqrtf(theta_sq);
  float imag, real;

  if (theta_sq < 1e-8) {
    imag = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_p4;
    real = 1.0 - (1.0/ 8.0)*theta_sq + (1.0/ 384.0)*theta_p4;
  } else {
    imag = sinf(0.5 * theta) / theta;
    real = cosf(0.5 * theta);
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;

}

__device__ void
crossInplace(const float* a, float *b) {
  float x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

__device__ void
expSE3(const float *xi, float* t, float* q) {
  // SE3 exponential map

  expSO3(xi + 3, q);
  float tau[3] = {xi[0], xi[1], xi[2]};
  float phi[3] = {xi[3], xi[4], xi[5]};

  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta = sqrtf(theta_sq);

  t[0] = tau[0]; 
  t[1] = tau[1]; 
  t[2] = tau[2];

  if (theta > 1e-4) {
    float a = (1 - cosf(theta)) / theta_sq;
    crossInplace(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];

    float b = (theta - sinf(theta)) / (theta * theta_sq);
    crossInplace(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
  }
}


// Doesn't have concept of kf0, kf1! Flows are hence computed outside optimization window as well.
__global__ void projective_transform_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> target,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> body_poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> extrinsics, // cam_T_imu
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> vs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eiz,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Ejz,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Cii,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bz)
{
  const int M = blockIdx.x; // Number of flow measurements.
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  int ix = static_cast<int>(ii[M]);
  int jx = static_cast<int>(jj[M]);

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  __shared__ float cam_T_body_tij[3];
  __shared__ float cam_T_body_qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];

    // Hard-code extrinsics for now! cam_T_imu
    cam_T_body_tij[0] = extrinsics[0]; // 0.0652229;
    cam_T_body_tij[1] = extrinsics[1]; // -0.0207064;
    cam_T_body_tij[2] = extrinsics[2]; // -0.0080546;

    cam_T_body_qij[0] = extrinsics[3]; // 0.00770718; //x
    cam_T_body_qij[1] = extrinsics[4]; // -0.01049932;//y
    cam_T_body_qij[2] = extrinsics[5]; // -0.7017528; //z
    cam_T_body_qij[3] = extrinsics[6]; // 0.71230146; //w
  }

  __syncthreads();

  if (ix == jx) {
    // stereo frames
    if (thread_id == 0) {
      tij[0] =  -0.1; // Baseline...
      tij[1] =     0;
      tij[2] =     0;
      qij[0] =     0;
      qij[1] =     0;
      qij[2] =     0;
      qij[3] =     1;
    }
  } else {
    // mono frames

    // load poses from global memory
    if (thread_id < 3) {
      ti[thread_id] = poses[ix][thread_id];
      tj[thread_id] = poses[jx][thread_id];
    }

    if (thread_id < 4) {
      qi[thread_id] = poses[ix][thread_id+3];
      qj[thread_id] = poses[jx][thread_id+3];
    }

    __syncthreads();

    if (thread_id == 0) {
      relSE3(ti, qi, tj, qj, tij, qij);
    }
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];

  // jacobians
  float Jx[12];
  float Jz;

  float* Ji = &Jx[0];
  float* Jj = &Jx[6];

  // hessians
  // 12*(12+1)/2 = 78
  // These are the upper triangular elements of the hessian matrix contributed by the point to the camera blocks.
  // H = [Hii Hij; Hji Hjj], where Hii and Hjj are symmetric. The actual hessian is the summation of all these blocks from the points.
  float hij[12*(12+1)/2]; // Upper triangular part of the Hessian only.

  float vi[6], vj[6];

  // Init hij's values to zero...
  int l;
  for (l=0; l<12*(12+1)/2; l++) {
    hij[l] = 0;
  }

  // Init vi, vj to zero.
  for (int n=0; n<6; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }

  __syncthreads();

  // AKA: for (size_t k = threadIdx.x; k<n; k += blockDim.x)
  GPU_1D_KERNEL_LOOP(k, ht*wd) { // AKA: loops over all pixels in the image by dividing chunks of pixels to threads.

    const int i = k / wd;
    const int j = k % wd;

    // We are at pixel (u,v) in the image of keyframe ix, and flow goes from ix to jx
    const float u = static_cast<float>(j);
    const float v = static_cast<float>(i);
    
    // homogenous coordinates
    Xi[0] = (u - cx) / fx;
    Xi[1] = (v - cy) / fy;
    Xi[2] = 1;
    Xi[3] = disps[ix][i][j]; // ix is the keyframe index.

    // transform homogenous point from keyframe ix to keyframe jx, for pixel (u,v)
    actSE3(tij, qij, Xi, Xj);

    const float x = Xj[0];
    const float y = Xj[1];
    const float h = Xj[3];

    const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];
    const float d2 = d * d;

    // Get the weight of the pixel (u,v)==(j,i) for the flow measurement number `block_id'
    // TODO: why the 0.001 !? Shouldn't it be Xj[3]?
    float weight_u = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[M][0][i][j];
    float weight_v = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[M][1][i][j];

    // Calculate the difference between the expected and current flow
    const float residual_u = target[M][0][i][j] - (fx * d * x + cx);
    const float residual_v = target[M][1][i][j] - (fy * d * y + cy);

    // Populate hessian and information vectors with jacobians
    // [H  Eii/Eij]
    // [   Cii]
    ////////////// x - coordinate 

    // x - coordinate of D(p')/D(psi_j)

    // Jacobian of reprojection function of u-coord of pixel wrt depth Z = 1/d
    // 1/d = Z; d2 = 1/Z^2; x = X; tij[0] = tx; tij[2] = tz;
    // Jz = fx * 1/Z * tx - fx * X/Z^2 * tz
    Jz = fx * (tij[0] * d - tij[2] * (x * d2)); // this is a scalar...

    // The C block is stored as a MxHW matrix, where M is the number of measurements, and HW is the number of pixels.
    Cii[M][k] = weight_u * Jz * Jz;
    bz[M][k] = weight_u * residual_u * Jz;

    if (ix == jx) weight_u = 0; // for stereo

    Jj[0] = fx * (h*d);
    Jj[1] = fx * 0.0;
    Jj[2] = fx * (-x*h*d2);
    Jj[3] = fx * (-x*y*d2);
    Jj[4] = fx * (1.0 + x*x*d2);
    Jj[5] = fx * (-y*d);

    adjSE3(tij, qij, Jj, Ji);
    for (int n=0; n<6; n++) Ji[n] *= -1.0;

    // if cam_T_body:
    adjSE3(cam_T_body_tij, cam_T_body_qij, Jj, Jj);
    adjSE3(cam_T_body_tij, cam_T_body_qij, Ji, Ji);

    // To get right jacobians wrt world_T_body
    for (int n=0; n<6; n++) Jj[n] *= -1.0;
    for (int n=0; n<6; n++) Ji[n] *= -1.0;

    // To get [wx wy wz tx ty tz] (gtsam) instead of [tx ty tz wx wy wz] (droid)
    float Jj_copy[6];
    for (int n=0; n<6; n++) Jj_copy[n] = Jj[n];
    Jj[0] = Jj_copy[3];
    Jj[1] = Jj_copy[4];
    Jj[2] = Jj_copy[5];
    Jj[3] = Jj_copy[0];
    Jj[4] = Jj_copy[1];
    Jj[5] = Jj_copy[2];
    float Ji_copy[6];
    for (int n=0; n<6; n++) Ji_copy[n] = Ji[n];
    Ji[0] = Ji_copy[3];
    Ji[1] = Ji_copy[4];
    Ji[2] = Ji_copy[5];
    Ji[3] = Ji_copy[0];
    Ji[4] = Ji_copy[1];
    Ji[5] = Ji_copy[2];

    // Compute contributions to the Hessian for the pose to pose block
    l=0;
    for (int n=0; n<12; n++) {
      for (int m=0; m<=n; m++) { // we only compute the upper diagonal, so stop at m at n...
        hij[l] += weight_u * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<6; n++) {
      // weight_u = 0 for stereo
      vi[n] += weight_u * residual_u * Ji[n];
      vj[n] += weight_u * residual_u * Jj[n];

      Eiz[M][n][k] = weight_u * Jz * Ji[n]; // Jz is a scalar
      Ejz[M][n][k] = weight_u * Jz * Jj[n];
    }

    ////////////// y - coordinate of D(p')/D(psi_j)

    // TODO: Parallelize the calculations above and below this line...

    // y - coordinate of D(p')/D(psi_j)
    Jz = fy * (tij[1] * d - tij[2] * (y * d2));
    Cii[M][k] += weight_v * Jz * Jz;
    bz[M][k] += weight_v * residual_v * Jz;

    if (ix == jx) weight_v = 0; // For stereo

    Jj[0] = fy * 0;
    Jj[1] = fy * (h*d);
    Jj[2] = fy * (-y*h*d2);
    Jj[3] = fy * (-1 - y*y*d2);
    Jj[4] = fy * (x*y*d2);
    Jj[5] = fy * (x*d);

    adjSE3(tij, qij, Jj, Ji);
    for (int n=0; n<6; n++) Ji[n] *= -1.0;

    // if cam_T_body:
    adjSE3(cam_T_body_tij, cam_T_body_qij, Jj, Jj);
    adjSE3(cam_T_body_tij, cam_T_body_qij, Ji, Ji);

    // To get right jacobians wrt world_T_body
    for (int n=0; n<6; n++) Jj[n] *= -1.0;
    for (int n=0; n<6; n++) Ji[n] *= -1.0;

    // To get [wx wy wz tx ty tz] (gtsam) instead of [tx ty tz wx wy wz] (droid)
    for (int n=0; n<6; n++) Jj_copy[n] = Jj[n];
    Jj[0] = Jj_copy[3];
    Jj[1] = Jj_copy[4];
    Jj[2] = Jj_copy[5];
    Jj[3] = Jj_copy[0];
    Jj[4] = Jj_copy[1];
    Jj[5] = Jj_copy[2];
    for (int n=0; n<6; n++) Ji_copy[n] = Ji[n];
    Ji[0] = Ji_copy[3];
    Ji[1] = Ji_copy[4];
    Ji[2] = Ji_copy[5];
    Ji[3] = Ji_copy[0];
    Ji[4] = Ji_copy[1];
    Ji[5] = Ji_copy[2];

    // Compute contributions to the Hessian
    l=0;
    for (int n=0; n<12; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += weight_v * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<6; n++) {
      // weight_v = 0 for stereo
      vi[n] += weight_v * residual_v * Ji[n];
      vj[n] += weight_v * residual_v * Jj[n];

      // Cross-contributions to the Hessian for the pose and depth blocks
      // H = [B, E; Et, C], these are the E contributions for pixel (u,v) of flow from i to j.
      // The size of the matrices are M*D*HW, where M are the number of measurements, D the dimensionality of the pose,
      Eiz[M][n][k] += weight_v * Jz * Ji[n];
      Ejz[M][n][k] += weight_v * Jz * Jj[n];
    }

    // Done with per-pixel update?
  }

  __syncthreads();

  __shared__ float sdata[THREADS];
  for (int n=0; n<6; n++) {
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      vs[0][M][n] = sdata[0];
    }

    __syncthreads();

    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      vs[1][M][n] = sdata[0];
    }

  }

  l=0;
  for (int n=0; n<12; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);

      if (threadIdx.x == 0) {
        if (n<6 && m<6) {
          Hs[0][M][n][m] = sdata[0];
          Hs[0][M][m][n] = sdata[0];
        }
        else if (n >=6 && m<6) {
          Hs[1][M][m][n-6] = sdata[0];
          Hs[2][M][n-6][m] = sdata[0];
        }
        else {
          Hs[3][M][n-6][m-6] = sdata[0];
          Hs[3][M][m-6][n-6] = sdata[0];
        }
      }

      l++;
    }
  }
}


__global__ void projmap_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> valid)
{

  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx;

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    ix = static_cast<int>(ii[block_id]);
    jx = static_cast<int>(jj[block_id]);
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();

  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = poses[ix][thread_id];
    tj[thread_id] = poses[jx][thread_id];
  }

  if (thread_id < 4) {
    qi[thread_id] = poses[ix][thread_id+3];
    qj[thread_id] = poses[jx][thread_id+3];
  }

  __syncthreads();

  if (thread_id == 0) {
    relSE3(ti, qi, tj, qj, tij, qij);
  }

  //points 
  float Xi[4];
  float Xj[4];

  __syncthreads();

  GPU_1D_KERNEL_LOOP(k, ht*wd) {
    const int i = k / wd;
    const int j = k % wd;

    const float u = static_cast<float>(j);
    const float v = static_cast<float>(i);
    
    // homogenous coordinates
    Xi[0] = (u - cx) / fx;
    Xi[1] = (v - cy) / fy;
    Xi[2] = 1;
    Xi[3] = disps[ix][i][j];

    // transform homogenous point
    actSE3(tij, qij, Xi, Xj);

    coords[block_id][i][j][0] = u;
    coords[block_id][i][j][1] = v;

    if (Xj[2] > 0.01) {
      coords[block_id][i][j][0] = fx * (Xj[0] / Xj[2]) + cx;
      coords[block_id][i][j][1] = fy * (Xj[1] / Xj[2]) + cy;
    }

    valid[block_id][i][j][0] = (Xj[2] > MIN_DEPTH) ? 1.0 : 0.0;

  }
}

__global__ void frame_distance_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> dist,
    const float beta) {

  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx;

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    ix = static_cast<int>(ii[block_id]);
    jx = static_cast<int>(jj[block_id]);
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();


  //points 
  float Xi[4];
  float Xj[4];

  __shared__ float accum[THREADS]; accum[thread_id] = 0;
  __shared__ float valid[THREADS]; valid[thread_id] = 0;
  __shared__ float total[THREADS]; total[thread_id] = 0;

  __syncthreads();

  for (int n=0; n<1; n++) {

    if (thread_id < 3) {
      ti[thread_id] = poses[ix][thread_id];
      tj[thread_id] = poses[jx][thread_id];
    }

    if (thread_id < 4) {
      qi[thread_id] = poses[ix][thread_id+3];
      qj[thread_id] = poses[jx][thread_id+3];
    }

    __syncthreads();


    relSE3(ti, qi, tj, qj, tij, qij);

    float d, du, dv;

    GPU_1D_KERNEL_LOOP(k, ht*wd) {
      const int i = k / wd;
      const int j = k % wd;

      const float u = static_cast<float>(j);
      const float v = static_cast<float>(i);


      // if (disps[ix][i][j] < 0.01) {
      //   continue;
      // }
      
      // homogenous coordinates
      Xi[0] = (u - cx) / fx;
      Xi[1] = (v - cy) / fy;
      Xi[2] = 1;
      Xi[3] = disps[ix][i][j];

      // transform homogenous point
      actSE3(tij, qij, Xi, Xj);

      du = fx * (Xj[0] / Xj[2]) + cx - u;
      dv = fy * (Xj[1] / Xj[2]) + cy - v;
      d = sqrtf(du*du + dv*dv);

      total[threadIdx.x] += beta;
      
      if (Xj[2] > MIN_DEPTH) {
        accum[threadIdx.x] += beta * d;
        valid[threadIdx.x] += beta;
      }

      Xi[0] = (u - cx) / fx;
      Xi[1] = (v - cy) / fy;
      Xi[2] = 1;
      Xi[3] = disps[ix][i][j];

      Xj[0] = Xi[0] + Xi[3] * tij[0];
      Xj[1] = Xi[1] + Xi[3] * tij[1];
      Xj[2] = Xi[2] + Xi[3] * tij[2];

      du = fx * (Xj[0] / Xj[2]) + cx - u;
      dv = fy * (Xj[1] / Xj[2]) + cy - v;
      d = sqrtf(du*du + dv*dv);

      total[threadIdx.x] += (1 - beta);
      
      if (Xj[2] > MIN_DEPTH) {
        accum[threadIdx.x] += (1 - beta) * d;
        valid[threadIdx.x] += (1 - beta);
      }
    }

    if (threadIdx.x == 0) {
      int tmp = ix;
      ix = jx;
      jx = tmp;
    }

    __syncthreads();

  }
  __syncthreads(); blockReduce(accum);
  __syncthreads(); blockReduce(total);
  __syncthreads(); blockReduce(valid);

  __syncthreads();

  if (thread_id == 0) {
    dist[block_id] = (valid[0] / (total[0] + 1e-8) < 0.75) ? 1000.0 : accum[0] / valid[0];
  }
}



__global__ void depth_filter_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> thresh,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> counter)
{

  const int block_id = blockIdx.x;
  const int neigh_id = blockIdx.y;
  const int index = blockIdx.z * blockDim.x + threadIdx.x;

  // if (threadIdx.x == 0) {
  //   printf("%d %d %d %d\n", blockIdx.x, blockIdx.y, blockDim.x, threadIdx.x);
  // }

  const int num = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx;

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  if (threadIdx.x == 0) {
    ix = static_cast<int>(inds[block_id]);
    jx = (neigh_id < 3) ? ix - neigh_id - 1 : ix + neigh_id;
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();

  if (jx < 0 || jx >= num) {
    return;
  }

  const float t = thresh[block_id];

  // load poses from global memory
  if (threadIdx.x < 3) {
    ti[threadIdx.x] = poses[ix][threadIdx.x];
    tj[threadIdx.x] = poses[jx][threadIdx.x];
  }

  if (threadIdx.x < 4) {
    qi[threadIdx.x] = poses[ix][threadIdx.x+3];
    qj[threadIdx.x] = poses[jx][threadIdx.x+3];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    relSE3(ti, qi, tj, qj, tij, qij);
  }

  //points 
  float Xi[4];
  float Xj[4];

  __syncthreads();

  if (index < ht*wd) {
    const int i = index / wd;
    const int j = index % wd;

    const float ui = static_cast<float>(j);
    const float vi = static_cast<float>(i);
    const float di = disps[ix][i][j];
    
    // homogenous coordinates
    Xi[0] = (ui - cx) / fx;
    Xi[1] = (vi - cy) / fy;
    Xi[2] = 1;
    Xi[3] = di;

    // transform homogenous point
    actSE3(tij, qij, Xi, Xj);

    const float uj = fx * (Xj[0] / Xj[2]) + cx;
    const float vj = fy * (Xj[1] / Xj[2]) + cy;
    const float dj = Xj[3] / Xj[2];

    const int u0 = static_cast<int>(floor(uj));
    const int v0 = static_cast<int>(floor(vj));

    if (u0 >= 0 && v0 >= 0 && u0 < wd-1 && v0 < ht-1) {
      const float wx = ceil(uj) - uj;
      const float wy = ceil(vj) - vj;

      const float d00 = disps[jx][v0+0][u0+0];
      const float d01 = disps[jx][v0+0][u0+1];
      const float d10 = disps[jx][v0+1][u0+0];
      const float d11 = disps[jx][v0+1][u0+1];

      // err, dj_hat are not used, but it seems the idea was to compare the re-projected depth from i to j,
      // with the bilinear interpolated depth from pose j... And if they are closer than the thresh, than add +1
      // for the depth at pixel (i,j) and camera [block_id]
      const float dj_hat = wy*wx*d00 + wy*(1-wx)*d01 + (1-wy)*wx*d10 + (1-wy)*(1-wx)*d11;

      const float err = abs(1.0/dj - 1.0/dj_hat);

      // But then here it seems it decided to just count the number of neighbours close to the reprojected depth.
      if       (abs(1.0/dj - 1.0/d00) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d01) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d10) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d11) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
    }
  }
}



__global__ void iproj_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> points)

{

  const int block_id = blockIdx.x;
  const int index = blockIdx.y * blockDim.x + threadIdx.x;


  const int num = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float t[3];
  __shared__ float q[4];

  if (threadIdx.x == 0) {
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();


  // load poses from global memory
  if (threadIdx.x < 3) {
    t[threadIdx.x] = poses[block_id][threadIdx.x];
  }

  if (threadIdx.x < 4) {
    q[threadIdx.x] = poses[block_id][threadIdx.x+3];
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];

  if (index < ht*wd) {
    const int i = index / wd;
    const int j = index % wd;

    const float ui = static_cast<float>(j);
    const float vi = static_cast<float>(i);
    const float di = disps[block_id][i][j];
    
    // homogenous coordinates
    Xi[0] = (ui - cx) / fx;
    Xi[1] = (vi - cy) / fy;
    Xi[2] = 1;
    Xi[3] = di;

    // transform homogenous point
    actSE3(t, q, Xi, Xj);

    points[block_id][i][j][0] = Xj[0] / Xj[3];
    points[block_id][i][j][1] = Xj[1] / Xj[3];
    points[block_id][i][j][2] = Xj[2] / Xj[3];

  }
}



__global__ void accum_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> inps,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ptrs,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> idxs,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> outs)
{
  
  const int block_id = blockIdx.x;
  const int D = inps.size(2);

  const int start = ptrs[block_id];
  const int end = ptrs[block_id+1];

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    float x = 0;
    for (int i=start; i<end; i++) {
      x += inps[idxs[i]][k];
    }
    outs[block_id][k] = x;
  }  
}


__device__ void
retrSE3(const float *xi, const float* t, const float* q, float* t1, float* q1) {
  // retraction on SE3 manifold

  float dt[3] = {0, 0, 0};
  float dq[4] = {0, 0, 0, 1};
  
  expSE3(xi, dt, dq);

  q1[0] = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
  q1[1] = dq[3] * q[1] + dq[1] * q[3] + dq[2] * q[0] - dq[0] * q[2];
  q1[2] = dq[3] * q[2] + dq[2] * q[3] + dq[0] * q[1] - dq[1] * q[0];
  q1[3] = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];

  actSO3(dq, t, t1);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];
}


__global__ void pose_retr_kernel(
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dx,
    const int kf0, const int kf1) 
{

  for (int k=kf0+threadIdx.x; k<kf1; k+=blockDim.x) {
    float xi[6], q[4], q1[4], t[3], t1[3];

    t[0] = poses[k][0];
    t[1] = poses[k][1];
    t[2] = poses[k][2];

    q[0] = poses[k][3];
    q[1] = poses[k][4];
    q[2] = poses[k][5];
    q[3] = poses[k][6];
    
    for (int n=0; n<6; n++) {
      xi[n] = dx[k-kf0][n];
    }

    retrSE3(xi, t, q, t1, q1);

    poses[k][0] = t1[0];
    poses[k][1] = t1[1];
    poses[k][2] = t1[2];

    poses[k][3] = q1[0];
    poses[k][4] = q1[1];
    poses[k][5] = q1[2];
    poses[k][6] = q1[3];
  }
}

__global__ void disp_retr_kernel(
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dz,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> inds) 
{
  const int i = inds[blockIdx.x];
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  for (int k=threadIdx.x; k<ht*wd; k+=blockDim.x) {
    float d = disps[i][k/wd][k%wd] + dz[blockIdx.x][k];
    disps[i][k/wd][k%wd] = d;
  }
}

torch::Tensor accum_cuda(torch::Tensor data, torch::Tensor ix, torch::Tensor jx) {
  torch::Tensor ix_cpu = ix.to(torch::kCPU);
  torch::Tensor jx_cpu = jx.to(torch::kCPU);
  torch::Tensor inds = torch::argsort(ix_cpu);

  long* ix_data = ix_cpu.data_ptr<long>();
  long* jx_data = jx_cpu.data_ptr<long>();
  long* kx_data = inds.data_ptr<long>();

  int count = jx.size(0);
  std::vector<int> cols;

  torch::Tensor ptrs_cpu = torch::zeros({count+1}, 
    torch::TensorOptions().dtype(torch::kInt64));
  
  long* ptrs_data = ptrs_cpu.data_ptr<long>();
  ptrs_data[0] = 0;

  int i = 0;
  for (int j=0; j<count; j++) {
    while (i < ix.size(0) && ix_data[kx_data[i]] <= jx_data[j]) {
      if (ix_data[kx_data[i]] == jx_data[j])
        cols.push_back(kx_data[i]);
      i++;
    }
    ptrs_data[j+1] = cols.size();
  }

  torch::Tensor idxs_cpu = torch::zeros({long(cols.size())}, 
    torch::TensorOptions().dtype(torch::kInt64));

  long* idxs_data = idxs_cpu.data_ptr<long>();

  for (int i=0; i<cols.size(); i++) {
    idxs_data[i] = cols[i];
  }

  torch::Tensor ptrs = ptrs_cpu.to(torch::kCUDA);
  torch::Tensor idxs = idxs_cpu.to(torch::kCUDA);

  torch::Tensor out = torch::zeros({jx.size(0), data.size(1)},
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  accum_kernel<<<count, THREADS>>>(
    data.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ptrs.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    idxs.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    out.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  return out;
}


__global__ void EEt6x6_kernel(
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> E,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> S)
{

  // indicices
  const int ix = idx[blockIdx.x][0];
  const int jx = idx[blockIdx.x][1];
  const int kx = idx[blockIdx.x][2];

  const int D = E.size(2);

  float dS[6][6];
  float ei[6];
  float ej[6];

  for (int i=0; i<6; i++) {
    for (int j=0; j<6; j++) {
      dS[i][j] = 0;
    }
  }

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    const float q = Q[kx][k];
      
    // coalesced memory read
    for (int n=0; n<6; n++) {
      ei[n] = E[ix][n][k] * q;
      ej[n] = E[jx][n][k];
    }

    // block EEt
    for (int n=0; n<6; n++) {
      for (int m=0; m<6; m++) {
        dS[n][m] += ei[n] * ej[m];
      }
    }
  }

  __syncthreads();
  __shared__ float sdata[THREADS];

  for (int n=0; n<6; n++) {
    for (int m=0; m<6; m++) {
      sdata[threadIdx.x] = dS[n][m];

      blockReduce(sdata);

      if (threadIdx.x == 0) {
        S[blockIdx.x][n][m] = sdata[0];
      }
    }
  }
}


__global__ void Ev6x1_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> E,
    const torch::PackedTensorAccessor32<float, 2,torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> w,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> v)
{
  const int D = E.size(2);
  const int kx = idx[blockIdx.x][0];

  float b[6];
  for (int n=0; n<6; n++) {
    b[n] = 0.0;
  }

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    const float q_w = Q[kx][k] * w[kx][k];

    for (int n=0; n<6; n++) {
      b[n] += q_w * E[blockIdx.x][n][k];
    }
  }

  __syncthreads();
  __shared__ float sdata[THREADS];

  for (int n=0; n<6; n++) {
    sdata[threadIdx.x] = b[n];
    blockReduce(sdata);

    if (threadIdx.x == 0) {
      v[blockIdx.x][n] += sdata[0];
    }
  }
}

// EvT6x1_kernel<<<ix.size(0), THREADS>>>(
__global__ void EvT6x1_kernel(
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> E,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> x,
  const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> idx,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> w)
{

  // blockIdx.x = index of the block, which in this case has size N+ii.size(0) (aka number of keyframes + number of measurements)

  const int HW = E.size(2);
  const int ix = idx[blockIdx.x];

  if (idx[blockIdx.x] <= 0 || idx[blockIdx.x] >= x.size(0))
    return;

  // Here, ix and block.Idx.x are fixed.
  // threadIdx.x = index of the thread in the block
  // blockDim.x = number of threads per block (aka 256 here)
  for (int k=threadIdx.x; k<HW; k+=blockDim.x) { 
    float dw = 0;
    for (int n=0; n<6; n++) { // D=6 for SE(3) poses
      dw += E[blockIdx.x][n][k] * x[ix][n]; // Et * delta_x
    }
    w[blockIdx.x][k] = dw;
  }
}

class SparseBlock {
  public:

    Eigen::SparseMatrix<double> A;
    Eigen::VectorX<double> b;

    SparseBlock(int N, int M) : N(N), M(M) {
      A = Eigen::SparseMatrix<double>(N*M, N*M);
      b = Eigen::VectorXd::Zero(N*M);
    }

    SparseBlock(Eigen::SparseMatrix<double> const& A, Eigen::VectorX<double> const& b, 
        int N, int M) : A(A), b(b), N(N), M(M) {}

    void update_lhs(torch::Tensor As, torch::Tensor ii, torch::Tensor jj) {
      auto As_cpu = As.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);
      auto jj_cpu = jj.to(torch::kCPU).to(torch::kInt64);

      // An accessor allows you to index (and access) the tensor...
      auto As_accessor = As_cpu.accessor<double,3>();
      auto ii_accessor = ii_cpu.accessor<long,1>();
      auto jj_accessor = jj_cpu.accessor<long,1>();

      // Build Sparse Matrix from triplets (col,row,value)
      // Builds Sparse Hessian from Dense Hessian
      std::vector<T> tripletList;
      for (int n = 0; n < ii.size(0); n++) {
        const int i = ii_accessor[n];
        const int j = jj_accessor[n];

        if (i >= 0 && j >= 0) { // this is because the (i,j) that are negative are the ones that are not optimized over (aka fixed poses)
          for (int k = 0; k < M; k++) { // M=6
            for (int l = 0; l < M; l++) { // M=6
              double val = As_accessor[n][k][l];
              // (col, row, value)
              tripletList.push_back(T(M * i + k, M * j + l, val));
            }
          }
        }
      }
      A.setFromTriplets(tripletList.begin(), tripletList.end());
    }

    void update_rhs(torch::Tensor bs, torch::Tensor ii) {
      auto bs_cpu = bs.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);

      auto bs_accessor = bs_cpu.accessor<double,2>();
      auto ii_accessor = ii_cpu.accessor<long,1>();

      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_accessor[n];
        if (i >= 0) {
          for (int j=0; j<M; j++) {
            b(i*M + j) += bs_accessor[n][j];
          }
        }
      }
    }

    SparseBlock operator-(const SparseBlock& S) {
      return SparseBlock(A - S.A, b - S.b, N, M);
    }

    std::tuple<torch::Tensor, torch::Tensor> get_dense() {
      Eigen::MatrixXd Ad = Eigen::MatrixXd(A);

      torch::Tensor H = torch::from_blob(Ad.data(), {N*M, N*M}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

      torch::Tensor v = torch::from_blob(b.data(), {N*M, 1}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

      return std::make_tuple(H, v);

    }

    torch::Tensor solve(const float lm=0.0001, const float ep=0.1) {

      torch::Tensor dx;

      Eigen::SparseMatrix<double> L(A);
      L.diagonal().array() += ep + lm * L.diagonal().array();

      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(L);

      if (solver.info() == Eigen::Success) {
        Eigen::VectorXd x = solver.solve(b);
        dx = torch::from_blob(x.data(), {N, M}, torch::TensorOptions()
          .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);
      }
      else {
        printf("ERROR solving Cholesky!!\n");
        dx = torch::zeros({N, M}, torch::TensorOptions()
          .device(torch::kCUDA).dtype(torch::kFloat32));
      }
      
      return dx;
    }

  private:
    const int N;
    const int M;

};


SparseBlock schur_block(torch::Tensor E,
                        torch::Tensor Q,
                        torch::Tensor w,
                        torch::Tensor ii,
                        torch::Tensor jj,
                        torch::Tensor kk,
                        const int kf0,
                        const int kf1)
{

  torch::Tensor ii_cpu = ii.to(torch::kCPU);
  torch::Tensor jj_cpu = jj.to(torch::kCPU);
  torch::Tensor kk_cpu = kk.to(torch::kCPU);

  const int P = kf1 - kf0;
  const long* ii_data = ii_cpu.data_ptr<long>();
  const long* jj_data = jj_cpu.data_ptr<long>();
  const long* kk_data = kk_cpu.data_ptr<long>();

  std::vector<std::vector<long>> graph(P);
  std::vector<std::vector<long>> index(P);

  for (int n=0; n<ii_cpu.size(0); n++) {
    const int j = jj_data[n];
    const int k = kk_data[n];

    if (j >= kf0 && j <= kf1) {
      const int t = j - kf0;
      graph[t].push_back(k);
      index[t].push_back(n);
    }
  }

  std::vector<long> ii_list, jj_list, idx, jdx;

  for (int i=0; i<P; i++) {
    for (int j=0; j<P; j++) {
      for (int k=0; k < graph[i].size(); k++) {
        for (int l=0; l < graph[j].size(); l++) {
          if (graph[i][k] == graph[j][l]) {
            ii_list.push_back(i);
            jj_list.push_back(j);

            idx.push_back(index[i][k]);
            idx.push_back(index[j][l]);
            idx.push_back(graph[i][k]);
          }
        }
      }
    }
  }

  torch::Tensor ix_cuda = torch::from_blob(idx.data(), {long(idx.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).to(torch::kCUDA).view({-1, 3});

  torch::Tensor jx_cuda = torch::stack({kk_cpu}, -1)
    .to(torch::kCUDA).to(torch::kInt64);

  torch::Tensor ii2_cpu = torch::from_blob(ii_list.data(), {long(ii_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor jj2_cpu = torch::from_blob(jj_list.data(), {long(jj_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor S = torch::zeros({ix_cuda.size(0), 6, 6}, 
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  torch::Tensor v = torch::zeros({jx_cuda.size(0), 6},
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  EEt6x6_kernel<<<ix_cuda.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ix_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
    S.packed_accessor32<float,3,torch::RestrictPtrTraits>());

  Ev6x1_kernel<<<jx_cuda.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    w.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    jx_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
    v.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  // schur block
  SparseBlock A(P, 6);
  A.update_lhs(S, ii2_cpu, jj2_cpu);
  A.update_rhs(v, jj_cpu - kf0);

  return A;
}


std::vector<torch::Tensor> ba_cuda(
    torch::Tensor poses,
    torch::Tensor body_poses, // SEND IDENTITY, this is not supposed to work otw
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor extrinsics,
    torch::Tensor disps_sens,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii,
    torch::Tensor jj,
    const int kf0,
    const int kf1,
    const int iterations,
    const float lm,
    const float ep,
    const bool motion_only)
{
  auto opts = poses.options();
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor ts = torch::arange(kf0, kf1).to(torch::kCUDA);
  torch::Tensor ii_expanded = torch::cat({ts, ii}, 0);
  torch::Tensor jj_expanded = torch::cat({ts, jj}, 0);

  std::tuple<torch::Tensor, torch::Tensor> ii_unique = 
    torch::_unique(ii_expanded, true, true);

  torch::Tensor kx = std::get<0>(ii_unique);
  torch::Tensor kk_exp = std::get<1>(ii_unique);
    
  torch::Tensor dx;
  torch::Tensor dz;

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num, 6, 6}, opts);
  torch::Tensor vs = torch::zeros({2, num, 6}, opts);
  torch::Tensor Eii = torch::zeros({num, 6, ht*wd}, opts);
  torch::Tensor Eij = torch::zeros({num, 6, ht*wd}, opts);
  torch::Tensor Cii = torch::zeros({num, ht*wd}, opts);
  torch::Tensor wi = torch::zeros({num, ht*wd}, opts);

  for (int itr=0; itr<iterations; itr++) {

    projective_transform_kernel<<<num, THREADS>>>(
      targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      body_poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
      extrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
      ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      vs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Eii.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Eij.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cii.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      wi.packed_accessor32<float,2,torch::RestrictPtrTraits>());


    // pose x pose block
    SparseBlock A(kf1 - kf0, 6); // (N, M=6) -> A = (N*M, N*M), b = (N*M)

    A.update_lhs(Hs.reshape({-1, 6, 6}), 
        torch::cat({ii, ii, jj, jj}) - kf0, 
        torch::cat({ii, jj, ii, jj}) - kf0);

    A.update_rhs(vs.reshape({-1, 6}), 
        torch::cat({ii, jj}) - kf0);

    if (motion_only) {
      dx = A.solve(lm, ep);

      // update poses
      pose_retr_kernel<<<1, THREADS>>>(
        poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(), kf0, kf1);
    } else {
      // add depth residual if there are depth sensor measurements
      const float alpha = 0.05; // THIS should be a huber weight per measurement, not a constant
      torch::Tensor m = (disps_sens.index({kx, "..."}) > 0).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd});
      torch::Tensor C = accum_cuda(Cii, ii, kx) + m * alpha + (1 - m) * eta.view({-1, ht*wd}); 
      torch::Tensor w = accum_cuda(wi, ii, kx) - m * alpha * (disps.index({kx, "..."}) - disps_sens.index({kx, "..."})).view({-1, ht*wd});
      torch::Tensor Q = 1.0 / C; 

      torch::Tensor Ei = accum_cuda(Eii.view({num, 6*ht*wd}), ii, ts).view({kf1-kf0, 6, ht*wd});
      torch::Tensor E = torch::cat({Ei, Eij}, 0);

      SparseBlock S = schur_block(E, Q, w, ii_expanded, jj_expanded, kk_exp, kf0, kf1);

      // Return (A-S), the reduced camera matrix, and solve it via GTSAM together with IMU

      dx = (A - S).solve(lm, ep);

      // Given dx, solve for dz

      torch::Tensor ix = jj_expanded - kf0;
      torch::Tensor dw = torch::zeros({ix.size(0), ht*wd}, opts);

      EvT6x1_kernel<<<ix.size(0), THREADS>>>(
        E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        dw.packed_accessor32<float,2,torch::RestrictPtrTraits>());

      dz = Q * (w - accum_cuda(dw, ii_expanded, kx));

      // update poses
      pose_retr_kernel<<<1, THREADS>>>(
        poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(), kf0, kf1);

      // update disparity maps
      disp_retr_kernel<<<kx.size(0), THREADS>>>(
        disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dz.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        kx.packed_accessor32<long,1,torch::RestrictPtrTraits>());
    }

  }

  return {dx, dz};
}



torch::Tensor frame_distance_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    const float beta)
{
  auto opts = poses.options();
  const int num = ii.size(0);

  torch::Tensor dist = torch::zeros({num}, opts);

  frame_distance_kernel<<<num, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    dist.packed_accessor32<float,1,torch::RestrictPtrTraits>(), beta);

  return dist;
}


std::vector<torch::Tensor> projmap_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj)
{
  auto opts = poses.options();
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor coords = torch::zeros({num, ht, wd, 3}, opts);
  torch::Tensor valid = torch::zeros({num, ht, wd, 1}, opts);

  projmap_kernel<<<num, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    valid.packed_accessor32<float,4,torch::RestrictPtrTraits>());

  return {coords, valid};
}


torch::Tensor depth_filter_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh)
{
  const int num = ix.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor counter = torch::zeros({num, ht, wd}, disps.options());

  dim3 blocks(num, 6, NUM_BLOCKS(ht * wd));

  depth_filter_kernel<<<blocks, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    thresh.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    counter.packed_accessor32<float,3,torch::RestrictPtrTraits>());

  return counter;
}


torch::Tensor iproj_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics)
{

  const int nm = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  auto opts = disps.options();
  torch::Tensor points = torch::zeros({nm, ht, wd, 3}, opts);

  dim3 blocks(nm, NUM_BLOCKS(ht * wd));

  iproj_kernel<<<blocks, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    points.packed_accessor32<float,4,torch::RestrictPtrTraits>());

  return points;

}


// Computes reduced camera matrix:  H_c/H_p
// Aka the schur complement of the cameras wrt points
// Eigen::MatrixXd is a dynamic 2D matrix of doubles.
std::vector<torch::Tensor>
reduced_camera_matrix_cuda(
    torch::Tensor poses,
    torch::Tensor body_poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor extrinsics,
    torch::Tensor disps_sens,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii, // Contains active AND inactive keyframes
    torch::Tensor jj, // Contains active AND inactive keyframes
    const int kf0,    // Min active keyframe
    const int kf1)    // Max active keyframe
{
  auto opts = poses.options();
  const int M = ii.size(0); // number of measurements
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor ts = torch::arange(kf0, kf1).to(torch::kCUDA);
  torch::Tensor ii_expanded = torch::cat({ts, ii}, 0);
  torch::Tensor jj_expanded = torch::cat({ts, jj}, 0);

  std::tuple<torch::Tensor, torch::Tensor> ii_unique = 
    torch::_unique(ii_expanded, true, true);

  torch::Tensor ii_kf_ids = std::get<0>(ii_unique);
  torch::Tensor kk_exp = std::get<1>(ii_unique); // same size as ii_expanded, ii_expanded[i] == ii_kf_ids[kk_exp[i]]
    
  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, M, 6, 6}, opts);
  torch::Tensor vs = torch::zeros({2, M, 6}, opts);
  torch::Tensor Eiz = torch::zeros({M, 6, ht*wd}, opts);
  torch::Tensor Ejz = torch::zeros({M, 6, ht*wd}, opts);
  torch::Tensor Cii = torch::zeros({M, ht*wd}, opts);
  torch::Tensor wi = torch::zeros({M, ht*wd}, opts);

  // Here we should iterate...
  // By iterating on the whole function, we are re-allocating buffers all the time...
  projective_transform_kernel<<<M, THREADS>>>(
    targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    body_poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    extrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    vs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Eiz.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Ejz.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Cii.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    wi.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  // pose x pose block
  SparseBlock A(kf1 - kf0, 6); // (N, M=6) -> A = (N*M, N*M), b = (N*M)

  A.update_lhs(Hs.reshape({-1, 6, 6}), 
      torch::cat({ii, ii, jj, jj}) - kf0, 
      torch::cat({ii, jj, ii, jj}) - kf0);

  A.update_rhs(vs.reshape({-1, 6}), 
      torch::cat({ii, jj}) - kf0);

  // add depth residual if there are depth sensor measurements
  const float alpha = 0.05;
  torch::Tensor m = (disps_sens.index({ii_kf_ids, "..."}) > 0).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd});
  torch::Tensor C = accum_cuda(Cii, ii, ii_kf_ids) + m * alpha + (1 - m) * eta.view({-1, ht*wd}); // add alpha if sensed depth, eta if not.
  torch::Tensor w = accum_cuda(wi, ii, ii_kf_ids)  - m * alpha * (disps.index({ii_kf_ids, "..."}) - disps_sens.index({ii_kf_ids, "..."})).view({-1, ht*wd});
  torch::Tensor Q = 1.0 / C;

  // Accumulate all contributions from the different flows to get the Eii block, that constraints the i-th camera depth
  torch::Tensor Ei = accum_cuda(Eiz.view({M, 6*ht*wd}), ii, ts).view({kf1-kf0, 6, ht*wd}); 
  torch::Tensor E = torch::cat({Ei, Ejz}, 0);

  SparseBlock S = schur_block(E, Q, w, ii_expanded, jj_expanded, kk_exp, kf0, kf1);

  // Return (A-S), the reduced camera matrix, and solve it via GTSAM together with IMU
  SparseBlock rcm = (A - S);//dx = (A - S).solve(lm, ep);
  // We also want the inverse of rcm! Because it is our marginal covariance!
  // But we rather would take the covariance of the rcm with the IMU...
  auto tuple = rcm.get_dense();
  return {std::get<0>(tuple), std::get<1>(tuple), Q, E, w};
}

// Given dx, calc dz.
// With both dx and dz, retract poses and depths.
void solve_depth_cuda(
    torch::Tensor dx,
    torch::Tensor disps,
    torch::Tensor Q,
    torch::Tensor E,
    torch::Tensor w,
    torch::Tensor ii,
    torch::Tensor jj,
    const int kf0,
    const int kf1)
{
    auto opts = disps.options();
    const int ht = disps.size(1);
    const int wd = disps.size(2);

    // Given dx, solve for dz
    torch::Tensor dz;


    torch::Tensor ts = torch::arange(kf0, kf1).to(torch::kCUDA);
    torch::Tensor ii_expanded = torch::cat({ts, ii}, 0);
    torch::Tensor jj_expanded = torch::cat({ts, jj}, 0); // expanded because it contains ii->jj, prepended with 0->0,...N->N, aka self-loops, because E, is cat(Eii, Eij)

    std::tuple<torch::Tensor, torch::Tensor> ii_unique = 
      torch::_unique(ii_expanded, true, true);

    torch::Tensor ii_kf_ids = std::get<0>(ii_unique);
    torch::Tensor kk_exp = std::get<1>(ii_unique);

    torch::Tensor ix = jj_expanded - kf0; // Size of P + ii.size(0), aka P + N
    torch::Tensor dw = torch::zeros({ix.size(0), ht*wd}, opts);

    //  <<<numBlocks, threadsPerBlock>>>
    // THREADS = 256
    EvT6x1_kernel<<<ix.size(0), THREADS>>>(
      E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      dw.packed_accessor32<float,2,torch::RestrictPtrTraits>());

    dz = Q * (w - accum_cuda(dw, ii_expanded, ii_kf_ids));

    // TODO: remove since gtsam does the retraction itself...
    // update poses
    // pose_retr_kernel<<<1, THREADS>>>(
    //   poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    //   droid_dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(), kf0, kf1);

    // update disparity maps
    disp_retr_kernel<<<ii_kf_ids.size(0), THREADS>>>(
      disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      dz.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      ii_kf_ids.packed_accessor32<long,1,torch::RestrictPtrTraits>());
}

// void solve_cuda(
//     torch::Tensor A,
//     torch::Tensor S,
//     const float lm,
//     const float ep)
// {
//   dx = (A - S).solve(lm, ep); // Unfortunately, A & S are SchurBlocks, not tensors...
//   return dx;
// }
// 
void solve_poses_cuda(
    torch::Tensor poses,
    torch::Tensor dx,
    const int kf0,
    const int kf1)
{
  pose_retr_kernel<<<1, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    kf0, kf1);
}
