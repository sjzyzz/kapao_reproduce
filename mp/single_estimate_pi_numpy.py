from multiprocessing import Pool
import random
import time

import numpy as np


def estimate_nbr_points_in_quarter_circle(nbr_samples):
    np.random.seed()
    xs = np.random.uniform(0, 1, int(nbr_samples))
    ys = np.random.uniform(0, 1, int(nbr_samples))
    esitmate_inside_quarter_unit_circle = (xs * xs + ys * ys) <= 1.0
    return np.sum(esitmate_inside_quarter_unit_circle)


if __name__ == "__main__":
    nbr_samples_in_total = 1E8
    nbr_parallel_blocks = 4
    pool = Pool(processes=nbr_parallel_blocks)
    nbr_samples_per_worker = nbr_samples_in_total / nbr_parallel_blocks
    print(f"Making {nbr_samples_per_worker} samples per worker")
    nbr_trails_per_process = [nbr_samples_per_worker] * nbr_parallel_blocks
    t1 = time.time()
    nbr_in_unit_circles = pool.map(
        estimate_nbr_points_in_quarter_circle, nbr_trails_per_process
    )
    pi_estimate = sum(nbr_in_unit_circles) * 4 / nbr_samples_in_total
    print(f'Estimated pi: {pi_estimate}')
    print(f'Delta: {time.time() - t1}')