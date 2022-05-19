__global__ void convAttn(float* attnMap,
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
    int winPixelIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int batchIdx = blockIdx.z * blockDim.z + threadIdx.z;
    if (pixelIdx < H * W && winPixelIdx < winh * winw && batchIdx < B * Heads) {
        int embedDim = C / Heads;
        int resoQ = H * W;
        int resoK = resoQ + (winh - 1) * W + (winw - 1) * H + (winh - 1) * (winw - 1);  // (H + win - 1) * (W + win - 1)
        // int resoK = (H + win - 1) * (W + win - 1);
        int u = pixelIdx / W;
        int v = pixelIdx % W;

        int qIdx = batchIdx * embedDim * resoQ + pixelIdx;
//         int kIdx = qIdx + u * (win - 1) + (winPixelIdx / win) * (W + win - 1) + winPixelIdx % win;
        int kIdx = batchIdx * embedDim * resoK + (u + winPixelIdx / winw) * (W + winw - 1) + v + winPixelIdx % winw;

        float sum = 0;
        for (int i = 0; i < embedDim; i++) {
            sum += Q[qIdx + i * resoQ] * K[kIdx + i * resoK];
        }
        attnMap[(batchIdx * winh * winw + winPixelIdx) * resoQ + pixelIdx] = sum;
    }
}

void launch_convAttn(float* attnMap,
                     const float* q,
                     const float* k,
                     int B,
                     int Heads,
                     int winh,
                     int winw,
                     int C,
                     int H,
                     int W) {
    dim3 grid((H * W + 31)/32, win*win, (B * Heads + 31) / 32);
    dim3 block(32, 1, 32);
    convAttn<<<grid, block>>>(attnMap, q, k, B, Heads, winh, winw, C, H, W);
}



