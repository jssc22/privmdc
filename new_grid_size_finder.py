
import gsizes_log.adult as adult

def calculate_grid_size(eps, dataset):

    if dataset == "adult6":
        return adult.calculate_grid_size(eps)

def size_per_dim(size, order, dataset):
    if dataset == "adult6":
        return adult.size_per_dim(size, order)

