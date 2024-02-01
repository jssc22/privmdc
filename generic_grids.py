import numpy as np
import consistency_method as ConMeth
from generate_query import AttrType

class GenericCell:
    def __init__(self, dimension_num=None, level=0, cell_index=None):
        assert dimension_num is not None
        self.dimension = dimension_num
        self.left_interval_list = [-1 for i in range(dimension_num)]
        self.right_interval_list = [-1 for i in range(dimension_num)]
        self.index = cell_index
        self.cell_pos = None    # for overall consistency
        self.level = level
        self.next_level_grid = None
        self.real_count = 0
        self.perturbed_count = 0
        self.consistent_count = 0


class GenericUniformGrid:
    def __init__(self, index=None, attr_set=None, domain_list=None, attr_types_list=None,
                 grid_size=None, left_interval_list=None,
                 right_interval_list=None, kd_cell_lens_list=None, hbs=None, args=None):
        self.args = args
        self.hbs = hbs
        self.attr_set = attr_set
        self.dimension = len(attr_set)
        self.attr_types_list = attr_types_list
        self.domain_list = domain_list  # saves the domain of each attribute
        self.index = index
        self.cell_pos2cell_index = {}

        if left_interval_list is None:
            self.left_interval_list = [0 for i in range(self.dimension)]
        else:
            self.left_interval_list = left_interval_list
        if right_interval_list is None:
            self.right_interval_list = [(self.domain_list[i] - 1) for i in range(self.dimension)]
        else:
            self.right_interval_list = right_interval_list

        self.grid_size = grid_size
        self.kd_cell_lens_list = kd_cell_lens_list
        self.cell_list = []
        self.wu_matrix = []
        self.hbs_res_grid = None

    def get_consistent_grid(self):
        cell_est_list = [0 for i in range(len(self.cell_list))]
        for i in range(len(self.cell_list)):
            cell_est_list[i] = self.cell_list[i].perturbed_count
        consistent_value_list = ConMeth.norm_sub(cell_est_list, user_num=self.args.user_num)
        s = sum(consistent_value_list)
        # set the consistent count to each cell
        for i in range(len(self.cell_list)):
            self.cell_list[i].consistent_count = consistent_value_list[i]
        return

    def generate_grid(self):

        total_cell_num = self.grid_size[0]
        if self.dimension > 1:
            for i in range(1, len(self.grid_size)):
                total_cell_num *= self.grid_size[i]

        coordinates = [0] * self.dimension
        cursors = [0] * self.dimension
        cell_len_k_index = [0] * self.dimension
        l1_interval = 0
        l2_interval = 0
        r1_interval = 0
        r2_interval = 0
        cell_key = ""
        column_index = 0
        for i in range(total_cell_num):
            new_cell = GenericCell(self.dimension, cell_index=i)

            if self.dimension == 1:
                new_cell.cell_pos = [i]
                self.cell_pos2cell_index["{}".format(str(i))] = i
            else:
                new_cell.cell_pos = coordinates.copy()
                cell_key = "{}".format(str(coordinates[0]))
                for d in range(1, self.dimension):
                    cell_key += ",{}".format(str(coordinates[d]))
                self.cell_pos2cell_index[cell_key] = i

            for j in range(self.dimension):
                l_interval = cursors[j]
                d_cell_lens = self.kd_cell_lens_list[j]
                k_index = cell_len_k_index[j]
                r_interval = l_interval + d_cell_lens[k_index] - 1
                new_cell.left_interval_list[j] = l_interval
                new_cell.right_interval_list[j] = r_interval

            if self.dimension > 1:
                for z in range(self.dimension):

                    if coordinates[z] < self.grid_size[z] - 1:
                        coordinates[z] += 1
                        cell_len_k_index[z] += 1
                        cursors[z] = new_cell.right_interval_list[z] + 1
                        break
                    else:
                        coordinates[z] = 0
                        cell_len_k_index[z] = 0
                        cursors[z] = 0

            self.cell_list.append(new_cell)
        return

    def add_real_user_record_to_grid(self, attr_value_set: list = None):
        tmp_cell_index = self.get_cell_index_from_attr_value_set(attr_value_set)
        self.cell_list[tmp_cell_index].real_count += 1

    def add_perturbed_user_record_to_grid(self, attr_value_set: list = None):
        tmp_cell_index = self.get_cell_index_from_attr_value_set(attr_value_set)
        self.cell_list[tmp_cell_index].perturbed_count += 1

    def get_cell_index_from_attr_value_set(self, attr_value_set: list = None):

        cells_indexes = []
        for d in range(self.dimension):
            attr_value = attr_value_set[d]
            accumulated = 0
            kd_grid_lens = self.kd_cell_lens_list[d]
            for j in range(len(kd_grid_lens)):
                j_cell_len = kd_grid_lens[j]
                accumulated += j_cell_len
                if attr_value < accumulated:
                    cells_indexes.append(j)
                    break
        cell_key = str(cells_indexes[0])
        for i in range(1, len(cells_indexes)):
            cell_key += ",{}".format(str(cells_indexes[i]))

        if cell_key not in self.cell_pos2cell_index:
            ops = 0

        return self.cell_pos2cell_index[cell_key]

    def get_answer_query_of_cell(self, cell: GenericCell = None, range_query_node_list: list = None, private_flag=0):
        query_left_interval_list = []
        query_right_interval_list = []
        for i in range(len(range_query_node_list)):
            query_left_interval_list.append(range_query_node_list[i].left_interval)
            query_right_interval_list.append(range_query_node_list[i].right_interval)
        cell_point_counts = 1

        cell_2d_left_interval_list = cell.left_interval_list
        cell_2d_right_interval_list = cell.right_interval_list



        for i in range(len(cell_2d_left_interval_list)):
            cell_length = cell_2d_right_interval_list[i] - cell_2d_left_interval_list[i]
            cell_point_counts *= (cell_length + 1)  # note here 1 must be added

        overlap_point_counts = 1
        for i in range(len(cell_2d_left_interval_list)):
            cell_length = cell_2d_right_interval_list[i] - cell_2d_left_interval_list[i]
            query_length = query_right_interval_list[i] - query_left_interval_list[i]
            min_interval = min(cell_2d_left_interval_list[i], query_left_interval_list[i])
            max_interval = max(cell_2d_right_interval_list[i], query_right_interval_list[i])
            overlap_length = cell_length + query_length - abs(max_interval - min_interval) + 1
            if overlap_length <= 0:
                overlap_point_counts = 0
                break
            else:
                overlap_point_counts *= overlap_length

        tans = overlap_point_counts / cell_point_counts * cell.consistent_count
        return tans

    def get_answer_query_of_cell_with_wu_matrix(self, cell: GenericCell = None, query_node_list: list = None):

        query_left_interval_list = []
        query_right_interval_list = []

        node1 = query_node_list[0]
        node2 = []
        if len(query_node_list) == 2:
            node2 = query_node_list[1]

        node1_type = node1.attr_type
        node2_type = None
        if len(query_node_list) == 2:
            node2_type = node2.attr_type

        cell_y0 = cell.left_interval_list[0]
        cell_y1 = cell.right_interval_list[0]

        cell_x0 = 0
        cell_x1 = 0
        if len(query_node_list) == 2:
            cell_x0 = cell.left_interval_list[1]
            cell_x1 = cell.right_interval_list[1]

        for node in query_node_list:
            query_left_interval_list.append(node.left_interval)
            query_right_interval_list.append(node.right_interval)

        cell_point_counts = 1
        for i in range(len(cell.left_interval_list)):
            cell_length = cell.right_interval_list[i] - cell.left_interval_list[i]
            cell_point_counts *= (cell_length + 1)  # note here 1 must be added

        overlap_point_counts = 1
        for i in range(len(cell.left_interval_list)):
            cell_length = cell.right_interval_list[i] - cell.left_interval_list[i]
            query_length = query_right_interval_list[i] - query_left_interval_list[i]
            min_interval = min(cell.left_interval_list[i], query_left_interval_list[i])
            max_interval = max(cell.right_interval_list[i], query_right_interval_list[i])

            overlap_length = cell_length + query_length - abs(max_interval - min_interval) + 1
            if overlap_length <= 0:
                overlap_point_counts = 0
                break
            else:
                overlap_point_counts *= overlap_length
        tans = 0
        if overlap_point_counts == cell_point_counts:
            tans = cell.consistent_count
        elif overlap_point_counts == 0:
            tans = 0
        else:
            if len(cell.left_interval_list) == 2:
                for i in range(cell.left_interval_list[0], cell.right_interval_list[0] + 1):
                    if query_left_interval_list[0] <= i <= query_right_interval_list[0]:

                        for j in range(cell.left_interval_list[1], cell.right_interval_list[1] + 1):
                            if query_left_interval_list[1] <= j <= query_right_interval_list[1]:
                                tans += self.wu_matrix[i][j]
            else:
                cell_length = cell.right_interval_list[i] - cell.left_interval_list[i]
                tans += cell.consistent_count * overlap_point_counts/cell_length
        return tans

    def answer_query_tdg(self, query_node_list):
        assert len(self.attr_set) == len(query_node_list)
        ans = 0
        for i in range(len(self.cell_list)):
            cell = self.cell_list[i]
            ans += self.get_answer_query_of_cell(cell, query_node_list)
        return ans

    def answer_query_hbs(self, clusters):
        ans = 0
        for i in range(len(self.cell_list)):
            cell = self.cell_list[i]
            ans += self.get_hbs_answer_query_of_cell(cell, clusters)
        return ans

    def get_hbs_answer_query_of_cell(self, cell: GenericCell = None, clusters: list = None):

        cell_d1_left = cell.left_interval_list[0]
        cell_d1_right = cell.right_interval_list[0]
        cell_point_counts = (cell_d1_right - cell_d1_left + 1)
        overlap_point_counts = 0
        for c in clusters:
            cd1_left = c[0]
            if cd1_left > cell_d1_right:
                break

            cd1_right = c[1]
            if cd1_right < cell_d1_left:
                continue

            cell_length = cell_d1_right - cell_d1_left
            cluster_length = cd1_right - cd1_left
            min_interval = min(cell_d1_left, cd1_left)
            max_interval = max(cell_d1_right, cd1_right)
            overlap_length = cell_length + cluster_length - abs(max_interval - min_interval) + 1
            if overlap_length <= 0:
                continue
            else:
                overlap_point_counts += overlap_length

        tans = overlap_point_counts / cell_point_counts * cell.consistent_count
        return tans

    def answer_query_with_wu_matrix(self, query_node_list):
        assert len(self.attr_set) == len(query_node_list)
        ans = 0
        for i in range(len(self.cell_list)):
            cell = self.cell_list[i]
            ans += self.get_answer_query_of_cell_with_wu_matrix(cell, query_node_list)
        return ans




