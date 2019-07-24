import  argparse
def get_arg():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', type=str, help='CUB data_path', default='/home/ws/datasets/CUB_200_2011/')
    argparser.add_argument('--val_support_num', type=int, default=1)
    argparser.add_argument('--support_num', type=int, default=5)
    args = argparser.parse_args()
    return args
