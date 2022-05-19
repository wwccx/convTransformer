#include <torch/extension.h>
#include "convAttn.h"
void launch_convAttn(float* attnMap,
                     const float* q,
                     const float* k,
                     int B,
                     int Heads,
                     int winh,
                     int winw,
                     int C,
                     int H,
                     int W);
void convAttn(torch::Tensor &attnMap,
                       const torch::Tensor &q,
                       const torch::Tensor &k,
                       int64_t B,
                       int64_t Heads,
                       int64_t winh,
                       int64_t winw,
                       int64_t C,
                       int64_t H,
                       int64_t W
                       ){
    launch_convAttn((float *)attnMap.data_ptr(),
                (const float *)q.data_ptr(),
                (const float *)k.data_ptr(),
                B, Heads, winh, winw, C, H, W);
}







PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convAttn",
          &convAttn,
          "convAttn");
}

TORCH_LIBRARY(convAttn, m) {
    m.def("convAttn", convAttn);
}



