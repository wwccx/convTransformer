__global__ void convAttnBackward(float* gQ,
                                 float* gK,
                                 const float* gAttn,
                                 const float* Q,
                                 const float* K,
                                 int B,
                                 int Heads,
                                 int win,
                                 int C,
                                 int H,
                                 int W) {
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int dimIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int batchIdx = blockIdx.z * blockDim.z + threadIdx.z;
    int embedDim = C / Heads;
    if (pixelIdx < H * W && dimIdx < embedDim && batchIdx < B * Heads) {

        int resoQ = H * W;
        int resoK = resoQ + (win - 1) * (H + W + win - 1);  // (H + win - 1) * (W + win - 1)
        // int resoK = (H + win - 1) * (W + win - 1);
        int u = pixelIdx / W;
        int v = pixelIdx % W;

        int gAttnIdx = batchIdx * win * win *resoQ + pixelIdx;
        int kIdx = (batchIdx * embedDim + dimIdx) * resoK + pixelIdx + u * (win - 1);
        int qIdx = (batchIdx * embedDim + dimIdx) * resoQ + pixelIdx;
        float sumQ = 0;
        float sumK = 0;
        int uq = 0;
        int vq = 0;
        int pixelBias = 0;
        for (int i = 0; i < win * win; i++) {
            sumQ += gAttn[gAttnIdx + i * resoQ] * K[kIdx + (i / win) * (W + win - 1) + i % win];
            uq = u + win / 2 - (i / win);
            vq = v + win / 2 - (i % win);
            if (uq >= 0 && uq < H && vq >= 0 && vq < W) {
                pixelBias = (win / 2 - (i / win)) * W + win / 2 - (i % win);
                sumK += gAttn[gAttnIdx + i * resoQ + pixelBias] * Q[qIdx + pixelBias];
            }
        }
        gQ[qIdx] = sumQ;
        gK[kIdx + (win / 2) * (H + win - 1) + win / 2] = sumK;
    }
}

void launch_convAttnBackward(float* gQ,
                             float* gK,
                             const float* attn,
                             const float* Q,
                             const float* K,
                             int B,
                             int Heads,
                             int win,
                             int C,
                             int H,
                             int W) {
    dim3 grid((H * W + 31)/32, C / Heads, (B * Heads + 31) / 32);
    dim3 block(32, 1, 32);
    convAttnBackward<<<grid, block>>>(gQ, gK, attn, Q, K, B, Heads, win, C, H, W);
}



