#include <torch/extension.h>
#include "applyAttnBackward.h"
void launch_applyAttnBackward(float* gAttn,
                             float* gV,
                             const float* gX,
                             const float* Attn,
                             const float* V,
                             int B,
                             int Heads,
                             int win,
                             int C,
                             int H,
                             int W);

void applyAttnBackward(torch::Tensor &gAttn,
                            torch::Tensor &gV,
                            const torch::Tensor &gX,
                            const torch::Tensor &Attn,
                            const torch::Tensor &V,
                            int64_t B,
                            int64_t Heads,
                            int64_t win,
                            int64_t C,
                            int64_t H,
                            int64_t W
                       ){
    launch_applyAttnBackward((float *)gAttn.data_ptr(),
                    (float *)gV.data_ptr(),
                    (const float *)gX.data_ptr(),
                    (const float *)Attn.data_ptr(),
                    (const float *)V.data_ptr(),
                    B, Heads, win, C, H, W);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("applyAttnBackward",
          &applyAttnBackward,
          "applyAttnBackward");
}

TORCH_LIBRARY(applyAttnBackward, m) {
    m.def("applyAttnBackward", applyAttnBackward);
}