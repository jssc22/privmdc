import random
import sys
import numpy as np

import enum

class AttrType(enum.Enum):
    categorical = 1
    numerical = 2

class QueryAttrNode:
    def __init__(self, attr=-1, interval_length=-1, domain=None, attr_type=None,
                 args=None):
        self.args = args
        self.attr_domain = domain
        self.attr_index = attr
        self.attr_type = attr_type
        self.interval_length_ratio = None
        self.interval_length = None
        self.left_interval = None
        self.right_interval = None
        self.cat_value = None

        if attr_type == AttrType.numerical:
            self.interval_length_ratio = 1
            self.interval_length = interval_length
            if self.interval_length == -1:
                self.interval_length = self.attr_domain
            self.left_interval = 0
            self.right_interval = self.left_interval + self.interval_length - 1
        else:
            self.cat_value = np.random.randint(0, domain, 1)[0]
            self.left_interval = self.cat_value
            self.right_interval = self.cat_value


    def set_interval_length_ratio(self, interval_length_ratio=1.0):
        self.interval_length_ratio = interval_length_ratio
        window_size = int(np.floor(self.attr_domain * self.interval_length_ratio))
        self.left_interval = random.randint(0, self.attr_domain - window_size)
        self.right_interval = self.left_interval + window_size
        if self.right_interval >= self.attr_domain:
            self.right_interval = self.attr_domain - 1


class Query:
    def __init__(self, query_dimension=-1, attr_num=-1, domains_list=None,
                 attr_types_list=None,
                 args=None):
        self.args = args
        self.attr_types_list = attr_types_list
        self.domains_list = domains_list
        self.query_dimension = query_dimension
        self.attr_num = attr_num
        self.selected_attr_index_list = []
        self.attr_index_list = [i for i in range(self.attr_num)]
        self.attr_node_list = []
        assert self.query_dimension <= self.attr_num
        self.real_answer = None
        self.domain_ratio = 0
        self.attr_index2letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd',  4: 'e', 5: 'f', 6: 'g',
                              7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm',
                              13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's',
                              19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
        self.initialize_query()
        self.set_selected_attr_list()



    def initialize_query(self):
        # for each attr in the dataset
        for i, d, t, in zip(range(self.attr_num), self.domains_list, self.attr_types_list):
            self.attr_node_list.append(QueryAttrNode(i, domain=d, attr_type=t, args=self.args))

    def workload0(self):
        self.selected_attr_index_list = np.random.choice(self.args.attr_num, self.args.query_dimension, replace=False).tolist()


    def set_selected_attr_list(self):
        self.workload0()
        self.query_dimension = len(self.selected_attr_index_list)

    def define_values_for_selected_attrs(self, selecivity_list):

        for i in self.selected_attr_index_list:
            node = self.attr_node_list[i]
            if node.attr_type == AttrType.numerical:

                node.set_interval_length_ratio(selecivity_list[i])


    def print_query_answer(self, file_out=None):
        file_out.write(str(self.real_answer) + "\n")

    def print_query(self, file_out=None):

        len_attr = len(self.selected_attr_index_list)
        it = 0
        line = ""
        for i in self.selected_attr_index_list:
            qn = self.attr_node_list[i]
            if qn.attr_type == AttrType.numerical:
                line += str(qn.left_interval) + "<=" + self.attr_index2letter[qn.attr_index] + "<=" + str(qn.right_interval)
            elif qn.attr_type == AttrType.categorical:
                line += self.attr_index2letter[qn.attr_index] + "=" + str(qn.cat_value)
            else:
                raise Exception("Invalid attr type")
            it += 1
            if it < len_attr:
                line += " and "
        file_out.write(line + "\n")


class QueryList:
    def __init__(self,
                 query_dimension=-1,
                 attr_num=-1,
                 query_num=-1,
                 dimension_query_volume_list=None,
                 attr_types_list=None,
                 args=None, domains_list=None):
        self.args = args
        self.attr_types_list = attr_types_list
        self.domains_list = domains_list
        self.query_dimension = query_dimension
        self.query_num = query_num
        self.attr_num = attr_num
        if self.attr_num == -1:
            self.attr_num = self.args.attr_num
        self.query_list = []
        self.real_answer_list = []
        self.dimension_query_volume_list = dimension_query_volume_list
        self.direct_multiply_MNAE = None
        self.max_entropy_MNAE = None
        self.weight_update_MNAE = None
        assert self.query_dimension <= self.attr_num and self.query_num > 0

    def generate_query_list(self):
        for i in range(self.query_num):
            query = Query(self.query_dimension,
                          self.attr_num,
                          domains_list=self.domains_list,
                          attr_types_list=self.attr_types_list,
                          args=self.args)
            query.define_values_for_selected_attrs(self.dimension_query_volume_list)
            self.query_list.append(query)

    def generate_real_answer_list(self, user_record):

        for iq in range(len(self.query_list)):
            query = self.query_list[iq]
            count = 0
            for user_i in range(self.args.user_num):
                flag = True
                for attr_index in query.selected_attr_index_list:
                    attr_node = query.attr_node_list[attr_index]
                    real_value = user_record[user_i][attr_index]
                    if attr_node.attr_type == AttrType.numerical:
                        if attr_node.left_interval <= real_value <= attr_node.right_interval:
                            continue
                        else:
                            flag = False
                            break
                if flag:
                    count += 1

            query.real_answer = count
            if query.real_answer == 0:
                del self.query_list[iq]
                self.real_answer_list = []
                return True
            else:
                self.real_answer_list.append(count)
        return 0 in self.real_answer_list

    def print_query_list(self, file_out=None):
        for i in range(len(self.query_list)):
            tmp_query = self.query_list[i]
            file_out.write("select count(*) from foo where ")
            tmp_query.print_query(file_out)

    def print_query_answers(self, file_out=None):
        for i in range(len(self.query_list)):
            tmp_query = self.query_list[i]
            tmp_query.print_query_answer(file_out)

