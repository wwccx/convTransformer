__global__ void applyAttnBackward(float* gAttn,
                                 float* gV,
                                 const float* gX,
                                 const float* Attn,
                                 const float* V,
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

        int resoX = H * W;
        int resoV = resoX + (winh - 1) * W + (winw - 1) * H + (winh - 1) * (winw - 1);  // (H + win - 1) * (W + win - 1)
        // int resoK = (H + win - 1) * (W + win - 1);
        int u = pixelIdx / W;
        int v = pixelIdx % W;
        int attnIdx = batchIdx * winh * winw * resoX + pixelIdx;
        int xIdx = (batchIdx * embedDim + dimIdx) * resoX + pixelIdx;
        int vIdx = (batchIdx * embedDim + dimIdx) * resoV + pixelIdx + u * (winw - 1);
        //(batchIdx * embedDim + dimIdx) * resoV + u * (W + winw - 1) + v;
        float sumV = 0;
        int uq = 0;
        int vq = 0;
        int pixelBias = 0;
        for (int i = 0; i < winh * winw; i++) {
//             gAttn[attnIdx + i * resoX] += gX[xIdx] * V[vIdx + (i / win) * (W + win - 1) + (i % win)];
            uq = u + winh / 2 - (i / winw);
            vq = v + winw / 2 - (i % winw);
            if (uq >= 0 && uq < H && vq >= 0 && vq < W) {
               pixelBias = (winh / 2 - i / winw) * W + (winw / 2 - i % winw);
               sumV += gX[xIdx + pixelBias] * Attn[attnIdx + i * resoX + pixelBias];
            }
        }
        gV[vIdx + (winw / 2) * (H + winh - 1) + winw / 2] = sumV;
        // (i / winw) * (W + winw - 1) + i % win
    }
}

__global__ void applyAttnBackwardAttn(float* gAttn,
                                 const float* gX,
                                 const float* Attn,
                                 const float* V,
                                 int B,
                                 int Heads,
                                 int winh,
                                 int winw,
                                 int C,
                                 int H,
                                 int W) {
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int winPixelIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int batchIdx = blockIdx.z * blockDim.z + threadIdx.z;

    if (pixelIdx < H * W && winPixelIndex < winh * winw && batchIdx < B * Heads) {
        int embedDim = C / Heads;

        int resoX = H * W;
        int resoV = resoX + (winh - 1) * W + (winw - 1) * H + (winh - 1) * (winw - 1);  // (H + win - 1) * (W + win - 1)
        int u = pixelIdx / W;
        int v = pixelIdx % W;

        float sumAttn = 0;
        int xIdx = batchIdx * embedDim * resoX + pixelIdx;
        int vIdx = batchIdx * embedDim * resoV + (u + winPixelIndex / winw) * (W + winw - 1) + v + winPixelIndex % winw;

        for (int i = 0; i < embedDim; i++) {
            sumAttn = sumAttn + gX[xIdx + i * resoX] * V[vIdx + i * resoV];
        }
        gAttn[(batchIdx * winh * winw + winPixelIndex) * resoX + pixelIdx] = sumAttn;

    }
}

void launch_applyAttnBackward(float* gAttn,
                             float* gV,
                             const float* gX,
                             const float* Attn,
                             const float* V,
                             int B,
                             int Heads,
                             int winh,
                             int winw,
                             int C,
                             int H,
                             int W) {
    dim3 gridV((H * W + 31) / 32, C / Heads, (B * Heads + 31) / 32);
    dim3 blockV(32, 1, 32);
    applyAttnBackward<<<gridV, blockV>>>(gAttn, gV, gX, Attn, V, B, Heads, winh, winw, C, H, W);

    dim3 gridAttn((H * W + 31) / 32, win*win, (B * Heads + 31) / 32);
    dim3 blockAttn(32, 1, 32);
    applyAttnBackwardAttn<<<gridAttn, blockAttn>>>(gAttn, gX, Attn, V, B, Heads, winh, winw, C, H, W);
}



