#include <torch/extension.h>
#include "applyAttn.h"

void launch_applyAttn(float* output,
                      const float* attnMap,
                      const float* v,
                      int B,
                      int Heads,
                      int winh,
                      int winw,
                      int C,
                      int H,
                      int W);
void applyAttn(torch::Tensor &output,
                       const torch::Tensor &attnMap,
                       const torch::Tensor &v,
                       int64_t B,
                       int64_t Heads,
                       int64_t winh,
                       int64_t winw,
                       int64_t C,
                       int64_t H,
                       int64_t W
                       ){
    launch_applyAttn((float *)output.data_ptr(),
                   (const float *)attnMap.data_ptr(),
                   (const float *)v.data_ptr(),
                   B, Heads, winh, winw, C, H, W);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("applyAttn",
          &applyAttn,
          "applyAttn");
}

TORCH_LIBRARY(applyAttn, m) {
    m.def("applyAttn", applyAttn);
}
