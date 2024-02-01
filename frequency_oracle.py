import numpy as np
import math
import xxhash
import random
from enum import Enum


class FOProtocol(Enum):
    GRR = 1
    OUE = 2
    ADAPTIVE = 3

class OUE:  # OUE has the same variance as OLH
    def __init__(self, domain_size=None,
                 epsilon=None,
                 sampling_factor=1,
                 args=None):
        self.args = args
        self.domain = domain_size
        if epsilon is None:
            self.epsilon = self.args.epsilon
        else:
            self.epsilon = epsilon

        self.group_user_num = 0
        self.perturbed_count = np.zeros(self.domain, dtype=int)
        self.aggregated_count = np.zeros(self.domain, dtype=int)   # after correction
        self.p = 0.5


        self.q = 1.0 / (math.exp(self.epsilon) + 1)
        self.sampling_factor = sampling_factor

    def operation_perturb(self, real_value=None):
        self.perturbed_count[real_value] += 1

    def operation_aggregate(self):
        perturbed_count_1 = np.copy(self.perturbed_count)
        s = sum(perturbed_count_1)
        est_count = np.random.binomial(perturbed_count_1, self.p)
        perturbed_count_0 = self.group_user_num - np.copy(self.perturbed_count)
        est_count += np.random.binomial(perturbed_count_0, self.q)


        a = 1.0 / (self.p - self.q)
        b = self.group_user_num * self.q / (self.p - self.q)
        est_count = a * est_count - b

        self.aggregated_count = est_count / self.group_user_num * self.args.user_num

        return

class OLH:
    def __init__(self, domain_size=None, epsilon=None, sampling_factor=1, args=None):
        self.args = args
        if domain_size is None:
            self.domain_size = self.args.domain_size
        else:
            self.domain_size = domain_size
        if epsilon is None:
            self.epsilon = self.args.epsilon
        else:
            self.epsilon = epsilon

        self.group_user_num = 0
        self.perturbed_count = np.zeros(self.domain_size, dtype=int)
        self.aggregated_count = np.zeros(self.domain_size, dtype=int)  # after correction
        self.sampling_factor = sampling_factor
        self.ee = np.exp(self.epsilon)
        self.g = int(round(self.ee)) + 1
        self.p = self.ee / (self.ee + self.g - 1)
        self.q = 1 / self.g
        self.var = 4 * self.ee / (self.ee - 1) ** 2
        self.user_real_val_list = []

    def operation_perturb(self, real_value=None):
        self.perturbed_count[real_value] += 1
        self.user_real_val_list.append(real_value)

    def operation_aggregate(self):

        assert self.group_user_num == len(self.user_real_val_list)
        samples_one = np.random.random_sample(self.group_user_num)
        hash_function_seeds_list = np.random.randint(0, self.group_user_num, self.group_user_num)
        hashed_val_list = np.zeros(self.group_user_num, dtype= int)
        reported_val_list = np.zeros(self.group_user_num, dtype= int)
        est_count = np.zeros(self.domain_size, dtype= int)

        for i in range(self.group_user_num):
            tmp_real_val = self.user_real_val_list[i]
            hashed_val_list[i] = (xxhash.xxh32(str(tmp_real_val), seed=hash_function_seeds_list[i]).intdigest()) % self.g

            if samples_one[i] > self.g:
                tmp_report_val = np.random.randint(0, self.g - 1)
                if tmp_report_val >= hashed_val_list[i]:
                    tmp_report_val += 1
            else:
                tmp_report_val = hashed_val_list[i]

            reported_val_list[i] = tmp_report_val

        for j in range(self.domain_size):
            for i in range(self.group_user_num):
                hashed_j = (xxhash.xxh32(str(j), seed=hash_function_seeds_list[i]).intdigest()) % self.g
                if hashed_j == reported_val_list[i]:
                    est_count[j] += 1

        a = 1.0 * self.g / (self.p * self.g - 1)
        b = 1.0 * self.group_user_num / (self.p * self.g - 1)
        est_count = a * est_count - b
        self.aggregated_count = est_count / self.group_user_num * self.args.user_num


class GRR:

    def __init__(self, domain_size=None,
                 epsilon=None,
                 user_num=None,
                 args=None):
        self.args = args
        self.domain = domain_size
        self.user_num = user_num
        if epsilon is None:
            self.epsilon = self.args.epsilon
        else:
            self.epsilon = epsilon

        self.group_user_num = 0
        self.perturbed_count = np.zeros(self.domain, dtype=int)
        self.aggregated_count = np.zeros(self.domain, dtype=int)

        ee = np.exp(self.epsilon)
        self.p = ee / (ee + self.domain - 1)
        self.q = 1 / (ee + self.domain - 1)

    def operation_perturb(self, real_value=None):
        self.perturbed_count[real_value] += 1

    def operation_aggregate(self):
        n = self.group_user_num  # number of users
        perturbed_data = np.zeros(n, dtype=int)
        counter = 0
        for k, v in enumerate(self.perturbed_count):
            for _ in range(v):
                y = x = k
                p_sample = np.random.random_sample()

                if p_sample > self.p:
                    y = np.random.randint(0, self.domain - 1)
                    if y >= x:
                        y += 1
                perturbed_data[counter] = y
                counter += 1

        est = np.zeros(self.domain)
        unique, counts = np.unique(perturbed_data, return_counts=True)
        for i in range(len(unique)):
            est[unique[i]] = counts[i]

        a = 1.0 / (self.p - self.q)
        b = n * self.q / (self.p - self.q)
        est = a * est - b
        self.aggregated_count = est / self.group_user_num * self.args.user_num
        return
