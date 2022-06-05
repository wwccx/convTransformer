__global__ void rendering(float* image,
                          const float* direction,
                          const float* origin,
                          const float* tsdf,
                          int B,
                          int H,
                          int W,
                          int vH,
                          int vW,
                          int times,
                          float voxel_size,
                          float xmin,
                          float xmax,
                          float ymin,
                          float ymax,
                          float zmin,
                          float zmax) {
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
//     int dimIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int batchIdx = blockIdx.z * blockDim.z + threadIdx.z;
    // image B H W
    // direction B H W 3
    // origin B 3
    if (pixelIdx < H * W && batchIdx < B) {
        float3 ray_dir = make_float3(direction[batchIdx * H * W * 6 + pixelIdx * 6 + 0],
                                 direction[batchIdx * H * W * 6 + pixelIdx * 6 + 1],
                                 direction[batchIdx * H * W * 6 + pixelIdx * 6 + 2]);
        float3 ray_origin = make_float3(direction[batchIdx * H * W * 6 + pixelIdx * 6 + 3],
                                     direction[batchIdx * H * W * 6 + pixelIdx * 6 + 4],
                                     direction[batchIdx * H * W * 6 + pixelIdx * 6 + 5]);
//         float3 ray_origin = make_float3(origin[batchIdx * H * W * 3 + pixelIdx * 3 + 0],
//                                      origin[batchIdx * H * W * 3 + pixelIdx * 3 + 1],
//                                      origin[batchIdx * H * W * 3 + pixelIdx * 3 + 2]);
        int x = 0;
        int y = 0;
        int z = 0;
        float last_val = 1.0f;
        float tsdf_val = 1.0f;
        float x_cur = ray_origin.x;
        float y_cur = ray_origin.y;
        float z_cur = ray_origin.z;
        float sums = 0.0f;
        for (int i = 0; i < times; i++) {
            x_cur += ray_dir.x;
            y_cur += ray_dir.y;
            z_cur += ray_dir.z;
//             image[batchIdx * H * W + pixelIdx] = ray_origin.y + 0.00769406f;
//             continue;
            if (x_cur > xmin && x_cur < xmax && z_cur > zmin && z_cur < zmax && y_cur > ymin && y_cur < ymax) {
//                 if (i == times / 2) {
//                    image[batchIdx * H * W + pixelIdx] =  0.1f;
//                 }
//                 image[batchIdx * H * W + pixelIdx] =  1.0f/(i + 2);

                x = floor((x_cur - xmin) / voxel_size);
                y = floor((y_cur - ymin) / voxel_size);
                z = floor((z_cur - zmin) / voxel_size);
                tsdf_val = tsdf[x * vH * vW + y * vW + z];
                if (tsdf_val > 0.0f) {
                    last_val = tsdf_val;
//                     image[batchIdx * H * W + pixelIdx] = 1.444f;
                }
                else {
                    image[batchIdx * H * W * 4 + pixelIdx * 4 + 3] = (i + last_val / (last_val - tsdf_val)) * voxel_size;
                    image[batchIdx * H * W * 4 + pixelIdx * 4 + 0] = origin[x * vH * vW*3 + y * vW*3 + z*3 + 0];
                    image[batchIdx * H * W * 4 + pixelIdx * 4 + 1] = origin[x * vH * vW*3 + y * vW*3 + z*3 + 1];
                    image[batchIdx * H * W * 4 + pixelIdx * 4 + 2] = origin[x * vH * vW*3 + y * vW*3 + z*3 + 2];
                       break;
//                        x += 0;
                }

            }
//             else{
//                 origin_ = make_float3(origin_.x + ray_dir.x,
//                                       origin_.y + ray_dir.y,
//                                       origin_.z + ray_dir.z);
// //                 x_cur += ray_dir.x;
// //                 y_cur += ray_dir.y;
// //                 z_cur += ray_dir.z;
//
// //                 image[batchIdx * H * W + pixelIdx] = 1.333f;
//             }
        }
//         image[batchIdx * H * W + pixelIdx] = 1;
    }
}

void launch_rendering(float* image,
                      const float* direction,
                      const float* origin,
                      const float* tsdf,
                      int B,
                      int H,
                      int W,
                      int vH,
                      int vW,
                      int times,
                      float voxel_size,
                      float xmin,
                      float xmax,
                      float ymin,
                      float ymax,
                      float zmin,
                      float zmax){
    dim3 grid((H * W + 31)/32, 1, (B + 31) / 32);
    dim3 block(32, 1, 32);
    rendering<<<grid, block>>>(image, direction, origin, tsdf, B, H, W, vH, vW, times, voxel_size, xmin, xmax, ymin, ymax, zmin, zmax);
}