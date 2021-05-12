# -*- coding: utf-8 -*-
# @Time    : 2021/5/11 18:49
# @Author  : Zeqi@@
# @FileName: time_estimation.py
# @Software: PyCharm

import time

def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    print("Execution time:", time.perf_counter() - start_time)