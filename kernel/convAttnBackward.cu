__global__ void convAttnBackward(float* gQ,
                                 float* gK,
                                 const float* gAttn,
                                 const float* Q,
                                 const float* K,
                                 int B,
                                 int Heads,
                                 int winh,
                                 int winw,
                                 int C,
                                 int H,
                                 int W) {
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int dimIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int batchIdx = blockIdx.z * blockDim.z + threadIdx.z;
    int embedDim = C / Heads;
    if (pixelIdx < H * W && dimIdx < embedDim && batchIdx < B * Heads) {

        int resoQ = H * W;
        int resoK = resoQ + (winh - 1) * W + (winw - 1) * H + (winh - 1) * (winw - 1);  // (H + win - 1) * (W + win - 1)
        // int resoK = (H + win - 1) * (W + win - 1);
        int u = pixelIdx / W;
        int v = pixelIdx % W;

        int gAttnIdx = batchIdx * winh * winw *resoQ + pixelIdx;
        int kIdx = (batchIdx * embedDim + dimIdx) * resoK + pixelIdx + u * (winw - 1);
        int qIdx = (batchIdx * embedDim + dimIdx) * resoQ + pixelIdx;
        float sumQ = 0;
        float sumK = 0;
        int uq = 0;
        int vq = 0;
        int pixelBias = 0;
        for (int i = 0; i < winh * winw; i++) {
            sumQ += gAttn[gAttnIdx + i * resoQ] * K[kIdx + (i / winw) * (W + winw - 1) + i % winw];
            uq = u + winh / 2 - (i / winw);
            vq = v + winw / 2 - (i % winw);
            if (uq >= 0 && uq < H && vq >= 0 && vq < W) {
                pixelBias = (winh / 2 - (i / winw)) * W + winw / 2 - (i % winw);
                sumK += gAttn[gAttnIdx + i * resoQ + pixelBias] * Q[qIdx + pixelBias];
            }
        }
        gQ[qIdx] = sumQ;
        gK[kIdx + (winw / 2) * (H + winh - 1) + winw / 2] = sumK;
    }
}

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
                             int W) {
    dim3 grid((H * W + 31)/32, C / Heads, (B * Heads + 31) / 32);
    dim3 block(32, 1, 32);
    convAttnBackward<<<grid, block>>>(gQ, gK, attn, Q, K, B, Heads, winh, winw, C, H, W);
}



