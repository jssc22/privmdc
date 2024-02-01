import argparse


def generate_args():
    parser = argparse.ArgumentParser()

    # int type
    parser.add_argument("--domain_size", type=int, default=50000, help="domain")
    parser.add_argument("--user_num", type=int, default=50000, help="the number of users")
    parser.add_argument("--attr_num", type=int, default=8, help="the number of attrs")
    parser.add_argument("--domain_for_all", type=int, default=16, help="the domain size of each attr")
    parser.add_argument("--query_num", type=int, default=10, help="the number of queries")
    parser.add_argument("--query_dimension", type=int, default=3, help="the query dimension")
    parser.add_argument("--consistency_iteration_num_max", type=int, default=200, help="the maximum number of iterations in consitency operation")
    parser.add_argument("--wu_iteration_num_max", type=int, default=200, help="the maximum number of iterations in weighted update process")
    parser.add_argument("--bn_degree", type=int, default=2, help="bn_degree")
    parser.add_argument("--split_ratio", type=int, default=0.2, help="split_ratio")
    parser.add_argument("--dataset", type=str, default="adult", help="dataset")
    parser.add_argument("--data_type", type=str, default="num", help="data_type")
    parser.add_argument("--n_code", type=str, default="1m", help="n_code")
    parser.add_argument("--smart_post", type=bool, default=True, help="smart post")
    parser.add_argument("--consist", type=bool, default=True, help="consistency")
    parser.add_argument("--kd", type=bool, default=False, help="kd")
    parser.add_argument("--optimize_udist", type=bool, default=True, help="optimize_udist")
    parser.add_argument("--boost_dist", type=bool, default=True, help="boost_dist")
    parser.add_argument("--user_alpha", type=float, default=0.8, help="user_alpha")
    #parser.add_argument('-w', '--weights', nargs='+', help='<Required> Set flag', required=False, type=int)

    # float type
    parser.add_argument("--epsilon", type=float, default=1.0, help="the privacy budget")
    parser.add_argument("--dimension_query_volume", type=float, default=0.5, help="the dimensional query volume")

    # str type
    parser.add_argument("--algorithm_name", type=str, default="", help="choose the algorithm: TDG or HDG")

    # bool
    parser.add_argument("--log_g", type=bool, default=False, help="log_g")
    parser.add_argument("--selectivity", type=bool, default=True, help="selectivity")
    parser.add_argument("--rx", type=float, default=0.5, help="rx")
    parser.add_argument("--ry", type=float, default=0.5, help="ry")


    args = parser.parse_args()
    return args

