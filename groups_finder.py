import itertools

def find_grids(correlations, attributes):
    grids = []
    for d in range(len(attributes), 0, -1):
        attrr_combs = itertools.combinations(attributes, d)

        for attr_c in attrr_combs:
            stuff = list(attr_c)
            combos = []
            for subset in itertools.combinations(stuff, 2):
                l = list(subset)
                l.sort()
                combos.append(l)
            result = all(elem in correlations for elem in combos)

            if result:
                grids.append(stuff)
                for item in combos:
                    correlations.remove(item)
    return grids