import numpy as np


def norm_sub(cell_est_list: list, user_num=None, tolerance=1):
    np_cell_est_list = np.array(cell_est_list)
    estimates = np.copy(np_cell_est_list)

    while (np.fabs(sum(estimates) - user_num) > tolerance) or (estimates < 0).any():
        if (estimates <= 0).all():
            estimates[:] = user_num / estimates.size
            break
        estimates[estimates < 0] = 0
        total = sum(estimates)
        mask = estimates > 0
        diff = (user_num - total) / sum(mask)
        estimates[mask] += diff

    return estimates

