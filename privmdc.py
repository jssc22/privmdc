import copy
import numpy as np
import math
import grid_generator as GridGen
import generic_grids as GenericGridsGen
from estimate_method import EstimateMethod
import frequency_oracle as FO
from frequency_oracle import FOProtocol
import itertools
import new_grid_size_finder
from correlation_identifier import CorrIdentifier
import os
import groups_finder as GF
from users_dist_opt import UserDistOptmizer
import bin_packing_v2 as bin_packing
from generate_query import AttrType
import warnings
warnings.filterwarnings("ignore")

kOUE = "OUE"
kGRR = "GRR"


class PrivMDC:
    def __init__(self, phase_one_dataset_path, domain_list=None, attr_type_list=None, args=None,
                 protocol=None, alpha1=0.7, alpha2=0.03, default=True,
                 grid_weights=None, selectivities=None):
        self.user_mapping = None
        self.attr_name_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6,
                              'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12,
                              'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18,
                              't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}
        self.args = args
        self.grid_weights = grid_weights
        self.group_attr_num = 2
        self.group_num = 0
        self.domain_list = domain_list
        self.attr_type_list = attr_type_list
        self.attr_group_list = []
        self.grid_set = []
        self.legacy_grid_sets = []
        self.legacy_groups = []
        self.answer_list = []
        self.weighted_update_answer_list = []
        self.default = default
        self.selectivities = selectivities

        self.user_group_ldp_mech_list = []
        self.group_id2grid_size = {}
        self.legacy_group_id2grid_size = {}
        self.group2d_id2mapping_lens = {}
        self.group1d_id2cell_lens = {}
        self.protocol = protocol
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.group_id2proto = {}
        self.gs_list = {}
        for i in range(len(domain_list)):
            self.gs_list[i] = [0, 0, 0, 0, 0, 0, 0]

        self.legacy_gs_list = {}
        for i in range(len(domain_list)):
            self.legacy_gs_list[i] = [0, 0, 0, 0, 0, 0, 0]

        self.attr2cell_lens = {}
        self.legacy_attr2cell_lens = {}
        self.phase_one_dataset_path = phase_one_dataset_path
        self.grids_weights = None

    def generate_attr_group(self):


        corr_file = './test_dataset/gs/gs_{}_nc{}_sr{}_bn{}_t{}_d{}_e{}.txt'.format(self.args.dataset, self.args.n_code,
                                                                                 self.args.split_ratio,
                                                                                self.args.bn_degree,
                                                                   self.args.data_type,
                                                                   self.args.attr_num, self.args.epsilon)
        if not os.path.exists(corr_file):
            ci = CorrIdentifier(self.phase_one_dataset_path, bn_degree=self.args.bn_degree)
            bn = ci.get_bayesian_network(epsilon=self.args.epsilon)
            print(bn)
            for c, parents in bn:
                if len(parents) == 0:
                    self.attr_group_list.append([self.attr_name_map[c]])
                else:
                    for p in parents:
                        a1 = self.attr_name_map[c]
                        a2 = self.attr_name_map[p]
                        l = [a1, a2]
                        l.sort()
                        self.attr_group_list.append(l)

            for i in range(len(self.attr_type_list)):
                a_t = self.attr_type_list[i]
                if a_t == AttrType.numerical:
                    self.attr_group_list.append([i])

            self.group_num = len(self.attr_group_list)
            self.legacy_groups = self.attr_group_list.copy()
            self.args.group_num = self.group_num

            # find grids
            if self.args.kd:
                self.attr_group_list = GF.find_grids(self.attr_group_list, range(self.args.attr_num))
                self.args.group_num = self.group_num = len(self.attr_group_list)

            with open(corr_file, "w") as file:
                for g in self.attr_group_list:
                    str1 = ""
                    for i in range(len(g)):
                        str1 += str(g[i])

                        if i < len(g) - 1:
                            str1 += " "
                    file.write(str1)
                    file.write("\n")

        else:
            with open(corr_file, "r") as file:
                for line in file:
                    line = line.strip()
                    line = line.split()
                    group = list(map(int, line))
                    combinations = list(itertools.combinations(group, 2))
                    for c in combinations:
                        self.legacy_groups.append(list(c))
                    if len(group) == 1:
                        self.legacy_groups.append(group)
                    self.attr_group_list.append(group)

                self.group_num = len(self.attr_group_list)
                self.args.group_num = self.group_num

                if not self.args.kd:
                    self.group_num = len(self.legacy_groups)
                    self.args.group_num = self.group_num
                    self.attr_group_list = self.legacy_groups

        return self.attr_group_list

    def define_initial_cell_list_lens(self, domain, gran):
        remaining_domain = domain
        remaining_g = gran
        cell_length_list = []
        while remaining_g > 0:
            cur_cell_len = remaining_domain // remaining_g
            cell_length_list.append(cur_cell_len)
            remaining_domain -= cur_cell_len
            remaining_g -= 1
        return cell_length_list

    def construct_gs_list(self, attr_type, domain, attr_id, legacy=False):
        gs_cell_len_list = []
        if attr_type == AttrType.numerical:
            for i in range(6):
                if legacy:
                    gs_cell_len_list.append(self.define_initial_cell_list_lens(domain, self.legacy_gs_list[attr_id][i]))
                else:
                    gs_cell_len_list.append(self.define_initial_cell_list_lens(domain, self.gs_list[attr_id][i]))
        elif attr_type == AttrType.categorical:
            g1_cell_len_list = [1] * domain
            g2_cell_len_list = [1] * domain
        else:
            raise Exception("Invalid attr type")
        return gs_cell_len_list

    def setup_all_grid_formatting(self, legacy=False):

        # make the matching
        for attr_id in range(len(self.domain_list)):
            d = self.domain_list[attr_id]
            t = self.attr_type_list[attr_id]

            find = True
            increase_1d = True
            decrease_2d = False

            gs_cell_len_list = self.construct_gs_list(t, d, attr_id, legacy)


            attr2cell_lens = self.attr2cell_lens
            if legacy:
                attr2cell_lens = self.legacy_attr2cell_lens

            if attr_id not in attr2cell_lens:
                attr2cell_lens[attr_id] = []
            attr2cell_lens[attr_id].append([gs_cell_len_list[0], [], {}])

            for z in range(1, len(gs_cell_len_list)):
                if not gs_cell_len_list[z]:
                    if attr_id not in attr2cell_lens:
                        attr2cell_lens[attr_id] = []
                    attr2cell_lens[attr_id].append([])
                    continue

                l3 = []
                for item in gs_cell_len_list[0]:
                    l3.append(int(item))

                l4 = []
                for item in gs_cell_len_list[z]:
                    l4.append(int(item))

                g2index2g1_lens_int_index = None
                g2index2g1_lens_sum = None
                g2index2g1_lens_list = None
                g1_len_list = None

                try:
                    if t == AttrType.numerical:
                        g2index2g1_lens_int_index, \
                        g2index2g1_lens_sum, \
                        g2index2g1_lens_list, \
                        g1_len_list = bin_packing.solve_bin_packing(l3, l4)
                    else:
                        g1_len_list = l3

                except Exception as ex:
                    print("Error bin_packing attr_id: {}".format(attr_id))
                if len(g1_len_list) == len(l3):
                    find = False
                else:
                    print(g1_len_list)
                    print(gs_cell_len_list)
                    raise Exception("[setup_all_grid_formatting] Need to change gs_list content")

                if attr_id not in attr2cell_lens:
                    attr2cell_lens[attr_id] = []

                attr2cell_lens[attr_id].append([g1_len_list, l4, g2index2g1_lens_int_index])
        return

    def calculate_inital_grid_kd_size(self, index, legacy=False):

        group = None
        if legacy:
            group = self.legacy_groups[index]
        else:
            group = self.attr_group_list[index]

        size = new_grid_size_finder.calculate_grid_size(eps=self.args.epsilon, dataset=self.args.dataset)

        if legacy:
            self.legacy_group_id2grid_size[index] = size
        else:
            self.group_id2grid_size[index] = size

        size_per_dimension = new_grid_size_finder.size_per_dim(size=size, order=len(group), dataset=self.args.dataset)
        for i in range(len(size_per_dimension)):
            a = group[i]
            d = self.domain_list[a]
            if size_per_dimension[i] > d:
                size_per_dimension[i] = d

        for i in range(len(group)):
            if legacy:
                self.legacy_gs_list[group[i]][len(group)-1] = size_per_dimension[i]
            else:
                self.gs_list[group[i]][len(group)-1] = size_per_dimension[i]
        return

    def calculate_inital_grid_1d_size(self, attr_id, legacy=False):
        code = new_grid_size_finder.calculate_grid_size(eps=self.args.epsilon, dataset=self.args.dataset)
        size = new_grid_size_finder.size_per_dim(size=code, order=1, dataset=self.args.dataset)[0]

        if self.domain_list[attr_id] < size:
            size = self.domain_list[attr_id]

        if self.default:
            self.group_id2grid_size[attr_id] = size
        if legacy:
            self.legacy_gs_list[attr_id][0] = size
        else:
            self.gs_list[attr_id][0] = size

        return

    def construct_grid_set(self):

        for i in range(len(self.domain_list)):
            self.calculate_inital_grid_1d_size(i)

        for i in range(self.group_num):
            group = self.attr_group_list[i]
            if len(group) > 1:
                self.calculate_inital_grid_kd_size(i)

        self.setup_all_grid_formatting()

        tmp_grid = None
        for i in range(self.group_num):
            group = self.attr_group_list[i]
            group_domain = []
            for attr in group:
                group_domain.append(self.domain_list[attr])
            if len(self.attr_group_list[i]) == 1:  # 1D grid

                valid_list = 0
                for k in range(len(self.attr2cell_lens[attr])):
                    if self.attr2cell_lens[attr][k]:
                        valid_list = k
                        break
                d1_len_list = self.attr2cell_lens[attr][valid_list][0]
                grid_attr_types = [self.attr_type_list[group[0]]]
                gran_list = [self.gs_list[group[0]][0]]
                tmp_grid = GridGen.UniformGrid(index=i,
                                               attr_set=group,
                                               domain_list=group_domain,
                                               attr_types_list=grid_attr_types,
                                               grid_size=gran_list,
                                               cell_length_d1_list=d1_len_list,
                                               args=self.args)
            else:
                type_dims = []
                kd_len_list = []
                gran_list = []
                for g in group:
                    type_dims.append(self.attr_type_list[g])

                    kd_len_list.append(self.attr2cell_lens[g][len(group)-1][1])
                    gran_list.append(self.gs_list[g][len(group)-1])

                tmp_grid = GenericGridsGen.GenericUniformGrid(index=i,
                                               attr_set=group,
                                               domain_list=group_domain,
                                               attr_types_list=type_dims,
                                               grid_size=gran_list,
                                               kd_cell_lens_list=kd_len_list,
                                               args=self.args)
            tmp_grid.generate_grid()
            self.grid_set.append(tmp_grid)


        for i in range(len(self.domain_list)):
            self.calculate_inital_grid_1d_size(i, legacy=True)

        for i in range(len(self.legacy_groups)):
            group = self.legacy_groups[i]
            if len(group) > 1:
                self.calculate_inital_grid_kd_size(i, legacy=True)

        self.setup_all_grid_formatting(legacy=True)


        tmp_grid = None
        for i in range(len(self.legacy_groups)):
            group = self.legacy_groups[i]
            group_domain = []
            for attr in group:
                group_domain.append(self.domain_list[attr])
            if len(self.legacy_groups[i]) == 1:  # 1D grid

                valid_list = 0
                for k in range(len(self.legacy_attr2cell_lens[attr])):
                    if self.legacy_attr2cell_lens[attr][k]:
                        valid_list = k
                d1_len_list = self.legacy_attr2cell_lens[attr][valid_list][0]

                grid_attr_types = [self.attr_type_list[group[0]]]
                gran_list = [self.legacy_gs_list[group[0]][0]]
                tmp_grid = GridGen.UniformGrid(index=i,
                                               attr_set=group,
                                               domain_list=group_domain,
                                               attr_types_list=grid_attr_types,
                                               grid_size=gran_list,
                                               cell_length_d1_list=d1_len_list,
                                               args=self.args)
            else:
                type_dims = []
                kd_len_list = []
                gran_list = []
                for g in group:
                    type_dims.append(self.attr_type_list[g])
                    kd_len_list.append(self.legacy_attr2cell_lens[g][len(group)-1][1])
                    gran_list.append(self.legacy_gs_list[g][len(group)-1])

                tmp_grid = GenericGridsGen.GenericUniformGrid(index=i,
                                               attr_set=group,
                                               domain_list=group_domain,
                                               attr_types_list=type_dims,
                                               grid_size=gran_list,
                                               kd_cell_lens_list=kd_len_list,
                                               args=self.args)
            tmp_grid.generate_grid()
            self.legacy_grid_sets.append(tmp_grid)

        return

    def get_user_record_in_attr_group(self, user_record_i,
                                      attr_group: int = None):
        user_record_in_attr_group = []
        for tmp in self.attr_group_list[attr_group]:
            user_record_in_attr_group.append(user_record_i[tmp])
        return user_record_in_attr_group

    def define_user_mapping(self):
        def user_map(user_dist):
            user_m = {}
            for i in range(self.group_num):
                user_m[i] = []

            cursor = 0
            for j in range(len(user_dist)):
                for i in range(cursor, cursor+user_dist[j]):
                    user_m[j].append(i)
                cursor += user_dist[j]
            return user_m

        rs_list = []
        for i in range(len(self.attr_group_list)):
            group = self.attr_group_list[i]
            g_list = []
            for g in group:
                g_list.append(self.selectivities[g])
            rs_list.append(g_list)

        ls_list = []
        for i in range(len(self.attr_group_list)):
            lss = []
            group = self.attr_group_list[i]
            for g in group:
                lss.append(self.gs_list[g][len(group)-1])
            ls_list.append(lss)

        user_dist = []
        dist_file = './cache/dist_ds_{}_nc{}_sr{}_d{}_e{}_bn{}.txt'.format(self.args.dataset, self.args.n_code,
                                                                                self.args.split_ratio,
                                                                                self.args.attr_num,
                                                                                self.args.epsilon, self.args.bn_degree)

        if not os.path.exists(dist_file):
            udo = UserDistOptmizer(rs_list=rs_list, ls_list=ls_list, grids_weights=self.grids_weights, args=self.args)
            user_dist = udo.find_user_dist()
            with open(dist_file, "w") as file:
                for i in range(len(user_dist)):
                    str1 = str(user_dist[i])
                    file.write(str1)
                    file.write("\n")

        else:

            with open(dist_file, "r") as file:
                for line in file:
                    user_dist.append(int(line))

        self.user_mapping = user_map(user_dist)
        return

    def calculate_grids_weights(self, query_list):
        if not self.args.optimize_udist:
            return np.ones((len(self.attr_group_list),), dtype=int)
        self.grids_weights = np.zeros((len(self.attr_group_list),), dtype=int)
        for query in query_list:
            combinations = list(itertools.combinations(query.selected_attr_index_list, 2))
            for c in combinations:
                pair = list(c)

                g_index = self.attr_group_list.index([pair[0]])
                self.grids_weights[g_index] += 1
                g_index = self.attr_group_list.index([pair[1]])
                self.grids_weights[g_index] += 1

                for g in self.attr_group_list:
                    if set(pair).issubset(g):
                        g_index = self.attr_group_list.index(g)
                        self.grids_weights[g_index] += 1
                        break
        return


    def run_ldp(self, user_record):
        self.define_user_mapping()
        self.user_group_ldp_mech_list = []
        for j in range(self.group_num):
            tmp_grid = self.grid_set[j]
            tmp_domain_size = len(tmp_grid.cell_list)

            if tmp_domain_size > 3*math.exp(self.args.epsilon) + 2:
                tmp_ldr = FO.OUE(domain_size=tmp_domain_size,
                                 epsilon=self.args.epsilon,
                                 args=self.args)
            else:
                tmp_ldr = FO.GRR(domain_size=tmp_domain_size,
                                 epsilon=self.args.epsilon,
                                 args=self.args)

            self.user_group_ldp_mech_list.append(tmp_ldr)



        for group_index_of_user, users in self.user_mapping.items():
            for i in users:
                j = group_index_of_user
                self.user_group_ldp_mech_list[j].group_user_num += 1
                tmp_grid = self.grid_set[j]
                tmp_real_cell_index = 0
                record = user_record[i]
                user_record_in_attr_group_j = self.get_user_record_in_attr_group(record, j)
                tmp_real_cell_index = tmp_grid.get_cell_index_from_attr_value_set(
                    user_record_in_attr_group_j)
                tmp_ldp_mechanism = self.user_group_ldp_mech_list[j]
                tmp_ldp_mechanism.operation_perturb(tmp_real_cell_index)



        for j in range(self.group_num):
            tmp_ldp_mechanism = self.user_group_ldp_mech_list[j]
            if tmp_ldp_mechanism.group_user_num > 0:
                tmp_ldp_mechanism.operation_aggregate()
                tmp_grid = self.grid_set[j]  # the j-th Grid
                for k in range(len(tmp_grid.cell_list)):
                    aux = tmp_ldp_mechanism.aggregated_count[k]
                    tmp_grid.cell_list[k].perturbed_count = aux
            else:
                tmp_grid = self.grid_set[j]
                for k in range(len(tmp_grid.cell_list)):
                    tmp_grid.cell_list[k].perturbed_count = 0

        return


    def get_attr_group_list(self, selected_attr_list):

        def judge_sub_attr_list_in_attr_group(sub_attr_list, attr_group):
            flag = True
            for sub_attr in sub_attr_list:
                if sub_attr not in attr_group:
                    flag = False
                    break
            return flag

        attr_group_index_list = []
        attr_group_list = []
        for grid in self.grid_set:
            # note that here we judge if tmp_Grid.attr_set belongs to selected_attr_list
            if judge_sub_attr_list_in_attr_group(grid.attr_set, selected_attr_list):
                attr_group_index_list.append(grid.index)
                attr_group_list.append(grid.attr_set)

        return attr_group_index_list, attr_group_list

    def pmdc_answer_query(self, query):
        t_grid_ans = []
        attr_group_index_list, attr_group_list = self.get_attr_group_list(query.selected_attr_index_list)

        for k in attr_group_index_list:
            grid = self.grid_set[k]
            grid_query_attr_node_list = []
            for attr in grid.attr_set:
                grid_query_attr_node_list.append(query.attr_node_list[attr])

            t_grid_ans.append(grid.answer_query_with_wu_matrix(grid_query_attr_node_list))
        if query.query_dimension == self.group_attr_num:  # answer the 2-way marginal
            tans_weighted_update = t_grid_ans[0]
        else:
            et = EstimateMethod(args=self.args)
            tans_weighted_update = et.weighted_update(query, attr_group_list, t_grid_ans)

        return tans_weighted_update

    def answer_query_list(self, query_list):
        self.weighted_update_answer_list = []
        for query in query_list:
            tans_weighted_update = self.pmdc_answer_query(query)
            self.weighted_update_answer_list.append(tans_weighted_update)
        return

    def get_t_a_a(self, sub_attr_value=None, sub_attr=None,
                  relevant_attr_group_list: list = None,
                  c_reci_list=None):

        sum_c_reci_list = sum(c_reci_list)
        sum_t_v_i_a = 0
        for group_index in relevant_attr_group_list:
            t_v_i_a = 0
            grid = self.grid_set[group_index]

            if len(grid.attr_set) == 1:
                int_map = self.attr2cell_lens[sub_attr][1][2]  # g2index2g1_lens_int_index
                left_interval_1_way = int_map[sub_attr_value][0]
                right_interval_1_way = int_map[sub_attr_value][1]
                k = left_interval_1_way
                while k <= right_interval_1_way:
                    gcell = grid.cell_list[k]
                    t_v_i_a += gcell.consistent_count
                    k += 1

            else:
                sub_attr_index_in_grid = grid.attr_set.index(sub_attr)
                for gcell in grid.cell_list:
                    if gcell.cell_pos[sub_attr_index_in_grid] == sub_attr_value:
                        t_v_i_a += gcell.consistent_count
            sum_t_v_i_a += (c_reci_list[group_index] * t_v_i_a)
        t_a_a = sum_t_v_i_a / sum_c_reci_list
        return t_a_a

    def get_c_w_list(self, sub_attr=None, relevant_attr_group_list: list = None):
        c_reci_list = np.zeros(self.args.group_num)
        for group_index in relevant_attr_group_list:
            grid = self.grid_set[group_index]

            if self.args.smart_post:
                if len(grid.attr_set) == 1:
                    bl = len(self.user_mapping[group_index])
                    c_reci_list[group_index] = (len(self.user_mapping[group_index]) / self.args.user_num) / \
                                               (self.gs_list[sub_attr][0] // self.gs_list[sub_attr][1])
                else:
                    c_reci_list[group_index] = (len(self.user_mapping[group_index]) / self.args.user_num) / \
                                               self.gs_list[sub_attr][1]
            else:
                if len(grid.attr_set) == 1:
                    c_reci_list[group_index] = 1 / (self.gs_list[sub_attr][0] // self.gs_list[sub_attr][1])
                else:
                    c_reci_list[group_index] = 1 / self.gs_list[sub_attr][1]

        return c_reci_list

    def get_consistency_for_sub_attr(self, sub_attr_index=None):
        relevant_attr_group_list = []
        for i in range(self.group_num):
            if sub_attr_index in self.attr_group_list[i]:
                relevant_attr_group_list.append(i)

        sub_attr_domain = range(self.gs_list[sub_attr_index][1])
        for sub_attr_value in sub_attr_domain:
            c_reci_list = self.get_c_w_list(sub_attr_index, relevant_attr_group_list)
            t_a_a = self.get_t_a_a(sub_attr_value, sub_attr_index, relevant_attr_group_list, c_reci_list)

            for g_index in relevant_attr_group_list:
                t_v_i_a = 0
                t_v_i_c_cell_list = []
                grid = self.grid_set[g_index]
                if len(grid.attr_set) == 1:
                    int_map = self.attr2cell_lens[sub_attr_index][1][2]  # g2index2g1_lens_intervals
                    left_interval_1_way = int_map[sub_attr_value][0]
                    right_interval_1_way = int_map[sub_attr_value][1]
                    k = left_interval_1_way
                    while k <= right_interval_1_way:
                        tmp_cell = grid.cell_list[k]
                        t_v_i_c_cell_list.append(k)
                        t_v_i_a += tmp_cell.consistent_count
                        k += 1
                else:
                    sub_attr_index_in_grid = grid.attr_set.index(sub_attr_index)
                    for k in range(len(grid.cell_list)):
                        cell = grid.cell_list[k]
                        if cell.cell_pos[sub_attr_index_in_grid] == sub_attr_value:
                            t_v_i_c_cell_list.append(k)
                            t_v_i_a += cell.consistent_count

                for k in t_v_i_c_cell_list:
                    cell = grid.cell_list[k]
                    aux1 = c_reci_list[g_index]
                    aux2 = t_a_a - t_v_i_a
                    cell.consistent_count += aux2 * aux1

    def overall_consistency(self):
        for i in range(self.args.attr_num):
            self.get_consistency_for_sub_attr(i)
        return

    def get_consistent_grid_set(self):

        for grid in self.grid_set:
            grid.get_consistent_grid()

        if self.args.consist:
            self.overall_consistency()
            for i in range(self.args.consistency_iteration_num_max):
                for grid in self.grid_set:
                    grid.get_consistent_grid()
                self.overall_consistency()

            for tmp_grid in self.grid_set:
                tmp_grid.get_consistent_grid()

    def wu_iteration(self, grid_1d_list=None, grid_2d=None):


        for grid_1d in grid_1d_list:
            one_way_attr = grid_1d.attr_set[0]
            one_way_attr_index = grid_2d.attr_set.index(one_way_attr)
            for cell in grid_1d.cell_list:
                lower = cell.left_interval_list[0]
                upper = cell.right_interval_list[0] + 1
                if one_way_attr_index == 0:
                    sub_matrix = grid_2d.wu_matrix[lower:upper, :]
                    tmp_sum = np.sum(sub_matrix)
                    if tmp_sum == 0:
                        continue
                    grid_2d.wu_matrix[lower:upper, :] = grid_2d.wu_matrix[lower:upper,:] / tmp_sum * cell.consistent_count
                else:
                    sub_matrix = grid_2d.wu_matrix[:, lower:upper]
                    tmp_sum = np.sum(sub_matrix)
                    if tmp_sum == 0:
                        continue

                    grid_2d.wu_matrix[:, lower:upper] = grid_2d.wu_matrix[:,
                                                        lower:upper] / tmp_sum * cell.consistent_count

                grid_2d.wu_matrix = grid_2d.wu_matrix / np.sum(grid_2d.wu_matrix) * self.args.user_num

        for cell in grid_2d.cell_list:
            x_lower = cell.left_interval_list[0]
            x_upper = cell.right_interval_list[0] + 1
            y_lower = cell.left_interval_list[1]
            y_upper = cell.right_interval_list[1] + 1
            sub_matrix = grid_2d.wu_matrix[x_lower:x_upper, y_lower:y_upper]
            tmp_sum = np.sum(sub_matrix)
            if tmp_sum == 0:
                continue
            grid_2d.wu_matrix[x_lower:x_upper, y_lower:y_upper] = grid_2d.wu_matrix[x_lower:x_upper,
                                                                  y_lower:y_upper] / tmp_sum * cell.consistent_count
            tmp_sum = np.sum(grid_2d.wu_matrix)
            if tmp_sum == 0:
                continue

            grid_2d.wu_matrix = grid_2d.wu_matrix / tmp_sum * self.args.user_num
    def get_wu_for_2_way_group(self):

        for grid in self.grid_set:
            if len(grid.attr_set) == 2:

                if grid.attr_types_list[0] == AttrType.categorical and grid.attr_types_list[1] == AttrType.categorical:
                    continue

                grid_1_way_list = []
                for grid_1_way in self.grid_set:
                    if len(grid_1_way.attr_set) == 1 and grid_1_way.attr_set[0] in grid.attr_set:
                        grid_1_way_list.append(grid_1_way)
                domain_x = grid.domain_list[0]
                domain_y = grid.domain_list[1]
                grid.wu_matrix = np.zeros((domain_x, domain_y))
                grid.wu_matrix[:, :] = self.args.user_num / (domain_x * domain_y)

                for i in range(self.args.wu_iteration_num_max):
                    wu_matrix_before = np.copy(grid.wu_matrix)
                    self.wu_iteration(grid_1_way_list, grid)
                    wu_matrix_delta = np.sum(np.abs(grid.wu_matrix - wu_matrix_before))
                    if wu_matrix_delta < 0.1:
                        break

    def extract_grids(self):

        def copy_cell(lg_cell, n_cell):
            lg_cell.dimension = n_cell.dimension
            lg_cell.left_interval_list = n_cell.left_interval_list.copy()
            lg_cell.right_interval_list = n_cell.right_interval_list.copy()
            lg_cell.index = n_cell.index
            lg_cell.cell_pos = n_cell.cell_pos
            lg_cell.level = n_cell.level
            lg_cell.next_level_grid = n_cell.next_level_grid
            lg_cell.real_count = n_cell.real_count
            lg_cell.perturbed_count = n_cell.perturbed_count
            lg_cell.consistent_count = n_cell.consistent_count

        for legacy_grid in self.legacy_grid_sets:
            for ngrid in self.grid_set:
                if ngrid.attr_set == legacy_grid.attr_set:
                    for lg_cell, n_cell in zip(legacy_grid.cell_list, ngrid.cell_list):
                        lg_cell.perturbed_count = n_cell.perturbed_count

        for ngrid in self.grid_set:
            if ngrid.dimension > 2:
                combs = itertools.combinations(ngrid.attr_set, 2)
                for pair_grid in combs:
                    for lg in self.legacy_grid_sets:
                        if lg.attr_set == list(pair_grid):
                            ng_attr_indexes = [ngrid.attr_set.index(pair_grid[0]), ngrid.attr_set.index(pair_grid[1])]
                            xs_n = ngrid.grid_size[ng_attr_indexes[0]]
                            ys_n = ngrid.grid_size[ng_attr_indexes[1]]
                            sub_matrix_array = np.zeros([xs_n, ys_n])
                            for ncell in ngrid.cell_list:
                                pair_index = tuple([ncell.cell_pos[ng_attr_indexes[0]], ncell.cell_pos[ng_attr_indexes[1]]])
                                sub_matrix_array[pair_index] += ncell.perturbed_count


                            xs_lg = lg.grid_size[0]
                            ys_lg = lg.grid_size[1]
                            fx = int(xs_lg / xs_n)
                            fy = int(ys_lg / ys_n)
                            upscaled = np.kron(sub_matrix_array, np.ones((fx, fy)))
                            upscaled = upscaled / (fx*fy)

                            for lgc in lg.cell_list:
                                lgc.perturbed_count = upscaled[tuple(lgc.cell_pos)]


        aux_gs_list = self.gs_list.copy()
        self.gs_list = self.legacy_gs_list.copy()
        self.legacy_gs_list = aux_gs_list.copy()

        aux_grids = self.grid_set.copy()
        self.grid_set = self.legacy_grid_sets.copy()
        self.legacy_grid_sets = aux_grids.copy()
        self.group_num = self.args.group_num = len(self.grid_set)
        aux_groups = self.attr_group_list.copy()
        self.attr_group_list = self.legacy_groups.copy()
        self.legacy_groups = aux_groups.copy()

        aux_legacy_attr2cell_lens = self.attr2cell_lens.copy()
        self.attr2cell_lens = self.legacy_attr2cell_lens.copy()
        self.legacy_attr2cell_lens = aux_legacy_attr2cell_lens.copy()

        aux_legacy_group_id2grid_size = self.group_id2grid_size.copy()
        self.group_id2grid_size = self.legacy_group_id2grid_size.copy()
        self.legacy_group_id2grid_size = aux_legacy_group_id2grid_size.copy()

