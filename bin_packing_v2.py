
from ortools.linear_solver import pywraplp


def solve_bin_packing(g1_list, g2_list):
    data = {}
    data['weights'] = g1_list
    data['values'] = [1] * len(g1_list)
    assert len(data['weights']) == len(data['values'])
    data['num_items'] = len(data['weights'])
    data['all_items'] = range(data['num_items'])

    data['bin_capacities'] = g2_list
    data['num_bins'] = len(data['bin_capacities'])
    data['all_bins'] = range(data['num_bins'])


    solver = pywraplp.Solver.CreateSolver('SCIP')
    if solver is None:
        print('SCIP solver unavailable.')
        return

    x = {}
    for i in data['all_items']:
        for b in data['all_bins']:
            var_name = "x_{}_{}".format(i, b)
            x[i, b] = solver.BoolVar(var_name)



    for i in data['all_items']:
        aux = sum(x[i, b] for b in data['all_bins']) <= 1
        solver.Add(aux)


    for b in data['all_bins']:
        aux = sum(x[i, b] * data['weights'][i] for i in data['all_items']) <= data['bin_capacities'][b]
        solver.Add(aux)


    objective = solver.Objective()
    for i in data['all_items']:
        for b in data['all_bins']:
            objective.SetCoefficient(x[i, b], data['values'][i])
    objective.SetMaximization()

    status = solver.Solve()

    g2index2g1_lens_list = {}
    g1_len_list = []
    g2_index = 0
    if status == pywraplp.Solver.OPTIMAL:

        total_weight = 0
        for b in data['all_bins']:
            rect_width_list_by_bin = []
            bin_weight = 0
            bin_value = 0
            for i in data['all_items']:
                if x[i, b].solution_value() > 0:
                    rect_width_list_by_bin.append(data['weights'][i])
                    g1_len_list.append(data['weights'][i])
                    bin_weight += data['weights'][i]
                    bin_value += data['values'][i]
            g2index2g1_lens_list[g2_index] = rect_width_list_by_bin
            g2_index += 1
            total_weight += bin_weight

        g2index2g1_lens_int_index = {}
        cursor = 0
        for g2_id, g1_lens in g2index2g1_lens_list.items():
            summ = 0
            for l in g1_lens:
                summ += 1
            g2index2g1_lens_int_index[g2_id] = [cursor, cursor+summ-1]
            cursor += summ

        g2index2g1_lens_sum = {}
        cursor = 0
        for g2_id, g1_lens in g2index2g1_lens_list.items():
            summ = 0
            for l in g1_lens:
                summ += l
            g2index2g1_lens_sum[g2_id] = summ
            cursor += summ

    else:
        raise Exception('The problem does not have an optimal solution.')


    return g2index2g1_lens_int_index, g2index2g1_lens_sum, g2index2g1_lens_list, g1_len_list
