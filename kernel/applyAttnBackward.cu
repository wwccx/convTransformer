__global__ void applyAttnBackward(float* gAttn,
                                 float* gV,
                                 const float* gX,
                                 const float* Attn,
                                 const float* V,
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

        int resoX = H * W;
        int resoV = resoX + (win - 1) * (H + W + win - 1);  // (H + win - 1) * (W + win - 1)
        // int resoK = (H + win - 1) * (W + win - 1);
        int u = pixelIdx / W;
        int v = pixelIdx % W;
        int attnIdx = batchIdx * win * win * resoX + pixelIdx;
        int xIdx = (batchIdx * embedDim + dimIdx) * resoX + pixelIdx;
        int vIdx = (batchIdx * embedDim + dimIdx) * resoV + pixelIdx + u * (win - 1);
        float sumV = 0;
        int uq = 0;
        int vq = 0;
        int pixelBias = 0;
        for (int i = 0; i < win * win; i++) {
//             gAttn[attnIdx + i * resoX] += gX[xIdx] * V[vIdx + (i / win) * (W + win - 1) + (i % win)];
            uq = u + win / 2 - (i / win);
            vq = v + win / 2 - (i % win);
            if (uq >= 0 && uq < H && vq >= 0 && vq < W) {
               pixelBias = (win / 2 - i / win) * W + (win / 2 - i % win);
               sumV += gX[xIdx + pixelBias] * Attn[attnIdx + i * resoX + pixelBias];
            }
        }
        gV[vIdx + (win / 2) * (H + win - 1) + win / 2] = sumV;
    }
}

__global__ void applyAttnBackwardAttn(float* gAttn,
                                 const float* gX,
                                 const float* Attn,
                                 const float* V,
                                 int B,
                                 int Heads,
                                 int win,
                                 int C,
                                 int H,
                                 int W) {
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int winPixelIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int batchIdx = blockIdx.z * blockDim.z + threadIdx.z;

    if (pixelIdx < H * W && winPixelIndex < win * win && batchIdx < B * Heads) {
        int embedDim = C / Heads;

        int resoX = H * W;
        int resoV = resoX + (win - 1) * (H + W + win - 1);  // (H + win - 1) * (W + win - 1)
        int u = pixelIdx / W;
        int v = pixelIdx % W;

        float sumAttn = 0;
        int xIdx = batchIdx * embedDim * resoX + pixelIdx;
        int vIdx = batchIdx * embedDim * resoV + (u + winPixelIndex / win) * (W + win - 1) + v + winPixelIndex % win;

        for (int i = 0; i < embedDim; i++) {
            sumAttn = sumAttn + gX[xIdx + i * resoX] * V[vIdx + i * resoV];
        }
        gAttn[(batchIdx * win * win + winPixelIndex) * resoX + pixelIdx] = sumAttn;

    }
}

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
                             int W) {
    dim3 gridV((H * W + 31) / 32, C / Heads, (B * Heads + 31) / 32);
    dim3 blockV(32, 1, 32);
    applyAttnBackward<<<gridV, blockV>>>(gAttn, gV, gX, Attn, V, B, Heads, win, C, H, W);

    dim3 gridAttn((H * W + 31) / 32, win*win, (B * Heads + 31) / 32);
    dim3 blockAttn(32, 1, 32);
    applyAttnBackwardAttn<<<gridAttn, blockAttn>>>(gAttn, gX, Attn, V, B, Heads, win, C, H, W);
}



