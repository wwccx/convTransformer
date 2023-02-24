import torch


def transformer(input_img, theta, out_size, **kwargs):
    '''
    传入图像U： B C H W
    Parameters
    ----------
    input_size   :   b c h w   source_img
    theta:   单应性矩阵H_mat   b 3 3      M^-1 H M
    out_size:   (h w)
    kwargs:

    Returns: output        返回经过theta（B 3 3）变换后的 resampling_grid B H W 2
    -------
    '''

    def _repeat(x, n_repeats):
        '''
        Returns: batch_indices    b*h*w
        '''

        rep = torch.ones([n_repeats, ]).unsqueeze(0)  # size: 1 h*w
        rep = rep.int()
        x = x.int()

        x = torch.matmul(x.reshape([-1, 1]), rep)  # b h*w   其实就是dataset里的batch_indices
        return x.reshape([-1])

    def _interpolate(im, x, y, out_size, scale_h):
        '''
        Parameters
        ----------
        im: real_A   b c h w
        x:  x_grid_flatten     b*h*w              registered_img_A 对应到real_a的坐标场 grid_x
        y:  y_grid_flatten     b*h*w     -1 ~ 1
        out_size:    (h, w)
        scale_h:     True

        Returns
        -------
        '''

        num_batch, num_channels, height, width = im.size()

        height_f = height
        width_f = width
        out_height, out_width = out_size[0], out_size[1]

        zero = 0
        max_y = height - 1
        max_x = width - 1
        if scale_h:
            # 将 x_grid_flatten  0 ~ h
            x = (x + 1.0) * (width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        dim2 = torch.from_numpy(np.array(width))  # w
        dim1 = torch.from_numpy(np.array(width * height))  # h*w

        base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width)  # b*h*w       batch_indices
        if torch.cuda.is_available():
            dim2 = dim2.cuda()
            dim1 = dim1.cuda()
            y0 = y0.cuda()
            y1 = y1.cuda()
            x0 = x0.cuda()
            x1 = x1.cuda()
            base = base.cuda()
        base_y0 = base + y0 * dim2  # base是batch层面，  yo是相对于一张图片而言的位置
        base_y1 = base + y1 * dim2
        # registered_imgA对应到real_A图像上的点， 临近四点 a b c d的index
        idx_a = base_y0 + x0  # b*h*w
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # channels dim
        im = im.permute(0, 2, 3, 1)  # b h w c
        im_flat = im.reshape([-1, num_channels]).float()  # b*h*w  c      [-1, 1]像素值

        im_flat = (im_flat + 1.0) / 2.0 * 255.0

        idx_a = idx_a.unsqueeze(-1).long()  # b*h*w 1
        idx_a = idx_a.expand(height * width * num_batch, num_channels)  # b*h*w c
        Ia = torch.gather(im_flat, 0, idx_a)  # b*h*w c   从源图像real_A中取出图像a点对应的全部像素值Ia

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(height * width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)  # b点的全部像素值

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(height * width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)  # c点的全部像素值

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(height * width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)  # d点的全部像素值      b*h*w c

        x0_f = x0.float()  # b*h*w
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)  # a位置的权重系数  b*h*w 1
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id  # bilinear法      registered_real_A: b*h*w c

        output = (output / 255.0) * 2.0 - 1

        return output  # registered_real_A  b*h*w  c       如果registered_real_A对应到real_A的点（x,y）超出了real_A的边界，  根据双边插值后全为0

    def _meshgrid(height, width, scale_h):

        if scale_h:
            x_t = torch.matmul(torch.ones([height, 1]),
                               torch.transpose(torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 1), 1, 0))  # p_h  p_w
            y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1),
                               torch.ones([1, width]))
        else:
            x_t = torch.matmul(torch.ones([height, 1]),
                               torch.transpose(torch.unsqueeze(torch.linspace(0.0, width.float(), width), 1), 1, 0))
            y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height.float(), height), 1),
                               torch.ones([1, width]))

        x_t_flat = x_t.reshape((1, -1)).float()  # 1  p_h*p_w
        y_t_flat = y_t.reshape((1, -1)).float()  # 1  p_h*p_w

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)  # grid 就是类比F.affine_grid  3 p_h*p_w   就是初始化的grid
        if torch.cuda.is_available():
            grid = grid.cuda()
        return grid

    def _transform(theta, input_img, out_size, scale_h):
        '''
        theta: b 3 3  M^-1*H-1*M       input_dim: b c h w    input image  (real_A)   source_img
        out_size: h w   scale_h:  True
        Returns: resampling_grid    B H W 2
        -------

        '''
        num_batch, num_channels, height, width = input_img.size()
        #  Changed
        theta = theta.reshape([-1, 3, 3]).float()

        out_height, out_width = out_size[0], out_size[1]
        grid = _meshgrid(out_height, out_width,
                         scale_h)  # 初始化的grid  3 h*w  每一列(x y 1)    比如第一列  (-1, -1, 1)        将源图像(source_img)的xy坐标归一化到-1 ~ 1
        grid = grid.unsqueeze(0).reshape([1, -1])  # 1  3*h*w
        shape = grid.size()
        grid = grid.expand(num_batch, shape[1])  # b 3*h*w
        grid = grid.reshape([num_batch, 3, -1])  # b 3 h*w

        T_g = torch.matmul(theta,
                           grid)  # b 3 h*w       H_mat(-1 ~ 1坐标之间的变换矩阵)  matmul  init_grid    得到  registered_imgA对应到real_A的总的grid场 每一列 (x_s y_s t_s)     T_g 类比于 nemar里的 resampling_grid
        x_s = T_g[:, 0, :]  # b  h*w
        y_s = T_g[:, 1, :]
        t_s = T_g[:, 2, :]

        t_s_flat = t_s.reshape([-1])

        # smaller
        small = 1e-7
        smallers = 1e-6 * (1.0 - torch.ge(torch.abs(t_s), small).float())

        t_s = t_s + smallers
        # smallers = 1e-6*(1.0 - torch.ge(torch.abs(t_s_flat), small).float())
        #
        # t_s_flat = t_s_flat + smallers
        # condition = torch.sum(torch.gt(torch.abs(t_s_flat), small).float())
        # Ty changed
        x_s_flat = x_s.reshape([-1]) / t_s_flat  # x_s -> x_1    (x y s) -> (x y 1)      b*h*w
        y_s_flat = y_s.reshape([-1]) / t_s_flat  # y_s_flat: 应该是registered_imgA对应到real_A的 y_grid场

        # input_transformed = _interpolate( input_dim, x_s_flat, y_s_flat,out_size,scale_h)       # registered_imgA的像素值由real_A周围的像素进行插值得到    b*h*w  c

        # output = input_transformed.reshape([num_batch, out_height, out_width, num_channels ])   # b h w c

        # resampling_grid = torch.stack([x_s / t_s, y_s / t_s], dim = 1).view(num_batch, -1, out_height,
        #                                                                     out_width).permute(0, 2, 3, 1)  # b h w 2

        input_transformed = _interpolate(input_img, x_s_flat, y_s_flat, out_size,
                                         scale_h)  # registered_imgA的像素值由real_A周围的像素进行插值得到    b*h*w  c
        output = input_transformed.reshape([num_batch, out_height, out_width, num_channels])  # b h w c
        output = output.permute(0, 3, 1, 2)
        # return resampling_grid
        return output

    scale_h = True
    # resampling_grid = _transform(theta, input_size, out_size, scale_h)
    resampling_image = _transform(theta, input_img, out_size, scale_h)
    return resampling_image
