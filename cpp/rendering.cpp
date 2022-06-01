#include <torch/extension.h>
#include "rendering.h"

void launch_rendering(float* image,
                      const float* direction,
                      const float* origin,
                      const float* tsdf,
                      int B,
                      int H,
                      int W,
                      int vH,
                      int vW,
                      int times,
                      float voxel_size,
                      float xmin,
                      float xmax,
                      float ymin,
                      float ymax,
                      float zmin,
                      float zmax);

void rendering(torch::Tensor &image,
               const torch::Tensor &direction,
               const torch::Tensor &origin,
               const torch::Tensor &tsdf,
               int64_t B,
               int64_t H,
               int64_t W,
               int64_t vH,
               int64_t vW,
               int64_t times,
               double voxel_size,
               double xmin,
               double xmax,
               double ymin,
               double ymax,
               double zmin,
               double zmax
               ){
    launch_rendering((float *) image.data_ptr(),
                   (const float *) direction.data_ptr(),
                   (const float *) origin.data_ptr(),
                   (const float *) tsdf.data_ptr(),
                   B, H, W, vH, vW, times, voxel_size,
                   xmin, xmax, ymin, ymax, zmin, zmax);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rendering",
          &rendering,
          "render B images with HxW resolution");
}

TORCH_LIBRARY(rendering, m) {
    m.def("rendering", rendering);
}
