import math
import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint


kOUE = "OUE"
kGRR = "GRR"

class UserDistOptmizer:
    def __init__(self, rs_list, ls_list, grids_weights, args):
        self.rs_list = rs_list
        self.ls_list = ls_list
        self.grids_weights = grids_weights
        self.args = args

    def var_grr_1d(self, n, r, l_vec):
        ee = math.exp(self.args.epsilon)
        l = l_vec[0]
        nue = (0.7 / l) ** 2
        ne = ((l * r * (ee + l - 2)) / (n * ((ee - 1) ** 2)))
        return nue + ne

    def var_oue_1d(self, n, r, l_vec):
        ee = math.exp(self.args.epsilon)
        l = l_vec[0]
        return (0.7 / l) ** 2 + ((4 * l * r * ee) / (n * (ee - 1) ** 2))

    def var_general_1d(self, n, r, l_vec):
        ee = math.exp(self.args.epsilon)
        l = l_vec[0]
        if 3*ee + 2 > l:
            return self.var_grr_1d(n, r, l_vec)
        return self.var_oue_1d(n, r, l_vec)

    def var_grr_nd(self, n, rx_vec, l_vec):
        ee = math.exp(self.args.epsilon)
        sum_rl = 0
        for i in range(len(rx_vec)):
            sum_rl += rx_vec[i] * l_vec[i]

        prod_rl = rx_vec[0] * l_vec[0]
        for i in range(1, len(rx_vec)):
            prod_rl *= rx_vec[i] * l_vec[i]

        prod_l = l_vec[0]
        for i in range(1, len(rx_vec)):
            prod_l *= l_vec[i]

        d = len(l_vec)
        nue = (((2**(d-1) * 0.03 * sum_rl) / prod_l) ** 2)
        ne = ((prod_rl * (ee + prod_l - 2)) / (n * ((ee - 1) ** 2)))
        return nue + ne

    def var_oue_nd(self, n, rx_vec, l_vec):
        ee = math.exp(self.args.epsilon)
        sum_rl = 0
        for i in range(len(rx_vec)):
            sum_rl += rx_vec[i] * l_vec[i]

        prod_rl = rx_vec[0] * l_vec[0]
        for i in range(1, len(rx_vec)):
            prod_rl *= rx_vec[i] * l_vec[i]

        prod_l = l_vec[0]
        for i in range(1, len(rx_vec)):
            prod_l *= l_vec[i]

        return ((len(rx_vec) * 0.03 * (sum_rl)) / (prod_l)) ** 2 + (4 * prod_rl * ee) / (n * (ee - 1) ** 2)


    def var_general_nd(self, n, rx_vec, l_vec):
        ee = math.exp(self.args.epsilon)
        prod_l = l_vec[0]
        for i in range(1, len(rx_vec)):
            prod_l *= l_vec[i]
        if 3*ee + 2 > prod_l:
            return self.var_grr_nd(n, rx_vec, l_vec)
        return self.var_oue_nd(n, rx_vec, l_vec)

    def find_user_dist(self):
        def fobj(ns):
            cost = 0
            for i in range(len(ns)):
                var_error = 0
                if len(self.ls_list[i]) == 1:
                    var_error = self.var_general_1d(ns[i], self.args.dimension_query_volume, self.ls_list[i])
                else:
                    rx_vec = [self.args.dimension_query_volume] * len(self.ls_list[i])
                    var_error = self.var_general_nd(ns[i], rx_vec, self.ls_list[i])

                cost += (self.grids_weights[len(self.ls_list[i])-1]**2) * var_error

            return cost

        uniform_d = int(self.args.user_num / len(self.ls_list))
        ns0 = [uniform_d] * len(self.ls_list)

        b = (0, (self.args.user_num//7))
        bnds = [b] * len(self.ls_list)

        def constr_fun(x):
            return np.array(sum(x))

        nlc = NonlinearConstraint(constr_fun, 0, self.args.user_num+1)
        sol = differential_evolution(fobj,
                                     bounds=bnds,
                                     popsize=100,
                                     strategy="rand1bin",
                                     mutation=(0.3, 1.0),
                                     recombination=0.7,
                                     tol=1e-8,
                                     #maxiter=2000,
                                     #updating="deferred",
                                     polish=False,
                                     constraints=nlc)


        user_dist = []
        for s in sol.x:
            user_dist.append(int(s))

        for i in range(len(user_dist)):
            w = self.grids_weights[i]
            if w == 0:
                user_dist[i] = 0

        sum_dist = sum(user_dist)
        over = sum_dist - self.args.user_num
        if over > 0:
            remove_per_group = over // len(self.ls_list)
            if remove_per_group > 0:
                for i in range(len(user_dist)):
                    if user_dist[i] - remove_per_group > remove_per_group:
                        user_dist[i] -= remove_per_group
                        if user_dist[i] < 0:
                            raise Exception("users_dist_opt - Negative correction")

            sum_dist = sum(user_dist)
            over_residual = sum_dist - self.args.user_num

            if over_residual > 0:
                high = 0
                high_i = 0
                for j in range(len(user_dist)):
                    if user_dist[j] > high:
                        high = user_dist[j]
                        high_i = j
                user_dist[high_i] -= over_residual

        if over < 0:
            over = over * -1
            non_zero = 0
            for udi in user_dist:
                if udi > 0:
                    non_zero += 1
            add_per_group = over // non_zero
            for i in range(len(user_dist)):
                if user_dist[i] > 0:
                    user_dist[i] += add_per_group

            sum_dist = sum(user_dist)
            under_residual = self.args.user_num - sum_dist

            if under_residual > 0:
                user_dist[0] += under_residual

        return user_dist
