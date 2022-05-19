#include <torch/extension.h>
#include "convAttnBackward.h"
void launch_convAttnBackward(float* gQ,
                     float* gK,
                     const float* attn,
                     const float* Q,
                     const float* K,
                     int B,
                     int Heads,
                     int winh,
                     int winw,
                     int C,
                     int H,
                     int W);

void convAttnBackward(torch::Tensor &gQ,
                            torch::Tensor &gK,
                            const torch::Tensor &attn,
                            const torch::Tensor &Q,
                            const torch::Tensor &K,
                            int64_t B,
                            int64_t Heads,
                            int64_t winh,
                            int64_t winw,
                            int64_t C,
                            int64_t H,
                            int64_t W
                       ){
    launch_convAttnBackward((float *)gQ.data_ptr(),
                    (float *)gK.data_ptr(),
                    (const float *)attn.data_ptr(),
                    (const float *)Q.data_ptr(),
                    (const float *)K.data_ptr(),
                    B, Heads, winh, winw, C, H, W);
}







PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convAttnBackward",
          &convAttnBackward,
          "convAttnBackward");
}

TORCH_LIBRARY(convAttnBackward, m) {
    m.def("convAttnBackward", convAttnBackward);
}



