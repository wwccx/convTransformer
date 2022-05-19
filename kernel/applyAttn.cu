__global__ void applyAttn(float* output,
                            const float* attnMap,
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

        int resoMap = H * W;
        int resoV = resoMap + (winh - 1) * W + (winw - 1) * H + (winh - 1) * (winw - 1);  // (H + win - 1) * (W + win - 1)
        // int resoK = (H + win - 1) * (W + win - 1);
        int u = pixelIdx / W;
        int v = pixelIdx % W;

        int mapIdx = batchIdx * winh * winw * resoMap + pixelIdx;
        int vIdx = (batchIdx * embedDim + dimIdx) * resoV + u * (W + winw - 1) + v;

        float sum = 0;
        for (int i = 0; i < winh * winw; i++) {
            sum += attnMap[mapIdx + i * resoMap] * V[vIdx + (i / winw) * (W + winw - 1) + i % win];
//             sum += V[vIdx + i * resoV + (i / win) * (W + win - 1) + i % win];
        }
        output[(batchIdx * embedDim + dimIdx) * resoMap + pixelIdx] = sum;
    }
}
void launch_applyAttn(float* output,
                      const float* attnMap,
                      const float* v,
                      int B,
                      int Heads,
                      int winh,
                      int winw,
                      int C,
                      int H,
                      int W) {
    dim3 grid((H * W + 31)/32, C / Heads, (B * Heads + 31) / 32);
    dim3 block(32, 1, 32);
    applyAttn<<<grid, block>>>(output, attnMap, v, B, Heads, winh, winw, C, H, W);
}