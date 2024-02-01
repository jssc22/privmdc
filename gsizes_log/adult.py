
def calculate_grid_size(eps):
    if eps <= 0.2:
        return 1
    if eps <= 0.6:
        return 2
    elif eps <= 1.0:
        return 3
    elif eps <= 1.5:
        return 4
    elif eps <= 2.0:
        return 5
    else:
        return 5
def size_per_dim(size, order):
    if order == 1:
        if size == 1:
            return [2]
        elif size == 2:
            return [12]
        elif size == 3:
            return [16]
        elif size == 4:
            return [68]
        elif size == 5:
            return [70]
        else:
            return [256]

    elif order == 2:
        if size == 1:
            return [1, 1]
        elif size == 2:
            return [2, 2]
        elif size == 3:
            return [2, 2]
        elif size == 4:
            return [2, 2]
        elif size == 5:
            return [2, 2]
        else:
            return [4, 4]
    elif order == 3:
        if size == 1:
            return [1, 1, 1]
        elif size == 2:
            return [2, 2, 2]
        elif size == 3:
            return [2, 2, 2]
        elif size == 4:
            return [2, 2, 2]
        elif size == 5:
            return [2, 2, 2]
        else:
            return [2, 2, 2]
    elif order == 4:
        if size == 1:
            return [1, 1, 1, 1]
        elif size == 2:
            return [2, 2, 2, 2]
        elif size == 3:
            return [2, 2, 2, 2]
        elif size == 4:
            return [2, 2, 2, 2]
        elif size == 5:
            return [2, 2, 2, 2]
        else:
            return [2, 2, 2, 2]

    elif order == 5:
        if size == 1:
            return [1, 1, 1, 1, 1]
        elif size == 2:
            return [2, 2, 2, 2, 2]
        elif size == 3:
            return [2, 2, 2, 2, 2]
        elif size == 4:
            return [2, 2, 2, 2, 2]
        elif size == 5:
            return [2, 2, 2, 2, 2]
        else:
            return [4, 4, 4, 4, 4]

    elif order == 6:
        if size == 1:
            return [1, 1, 1, 1, 1, 1]
        elif size == 2:
            return [1, 1, 1, 2, 2, 2]
        elif size == 3:
            return [2, 2, 2, 2, 2, 2]
        elif size == 4:
            return [2, 2, 2, 2, 2, 2]
        elif size == 5:
            return [2, 2, 2, 2, 2, 2]
        else:
            return [4, 4, 4, 4, 4, 4]

    elif order == 7:
        if size == 1:
            return [1, 1, 1, 1, 1, 1, 1]
        elif size == 2:
            return [2, 2, 2, 2, 2, 2, 2]
        elif size == 3:
            return [2, 2, 2, 2, 2, 2, 2]
        elif size == 4:
            return [2, 2, 2, 2, 2, 2, 2]
        elif size == 5:
            return [2, 2, 2, 2, 2, 2, 2]
        else:
            return [2, 2, 2, 2, 2, 2, 2]