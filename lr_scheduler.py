import torch
from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler(config, optimier, n_iter_per_epoch):
    num_steps = config.n_epochs * n_iter_per_epoch
    warmup_steps = int(20 * n_iter_per_epoch)
    decay_steps = int(30 * n_iter_per_epoch)
    lr_scheduler = CosineLRScheduler(
        optimier,
        t_initial=num_steps,
        t_mul=1,
        lr_min=1.25e-6,
        warmup_lr_init=1e-7,
        warmup_t=warmup_steps,
        cycle_limit=1,
        decay_rate=1,
        t_in_epochs=False,
    )

    return lr_scheduler


if __name__ == '__main__':
    from convTrans import convTransformer
    import numpy as np
    l = np.array([])
    c = convTransformer(B=1)
    o = torch.optim.AdamW(c.parameters(), lr=5e-4)
    lrS = build_scheduler(2, o, 400)
    for i in range(300):
        for j in range(400):
            lrS.step_update((i * 400 + j))
            l = np.append(l, o.param_groups[0]['lr'])
        print(i)
    from matplotlib import pyplot as plt
    plt.plot(l)
    plt.show()
