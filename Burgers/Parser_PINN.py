import argparse
import torch
import math
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--load_path', default='./result/PINNlb206.pth'
    )
    parser.add_argument(
        '--is_load', default=False
    )
    parser.add_argument(
        '--train_plot', default=True
    )
    # 训练信息
    parser.add_argument(
        '--epochs', default=10000, type=int
    )
    parser.add_argument(
        '--N_f', default=700, type=int
    )
    parser.add_argument(
        '--N_b', default=1000, type=int
    )
    parser.add_argument(
        '--N_u', default=2000, type=int
    )
    parser.add_argument(
        '--N_p', default=200, type=int
    )
    parser.add_argument(
        '--w_u', default=1.0, type=float
    )
    parser.add_argument(
        '--w_f', default=1.0, type=float
    )
    parser.add_argument(
        '--w_b', default=1.0, type=float
    )
    parser.add_argument(
        '--nu', default=0.01/math.pi)

    parser.add_argument(
        '--lr', default=0.001, type=float
    )
    parser.add_argument(
        '--criterion', default=torch.nn.MSELoss()
    )
    parser.add_argument(
        '--optimizer', default=torch.optim.Adam
    )
    parser.add_argument(
        '--optimizer_sigma', default=torch.optim.Adam
    )
    # 网络信息
    parser.add_argument(
        '--seq_net', default=[2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    )

    parser.add_argument(
        '--activation', default=torch.tanh
    )
    parser.add_argument(
        '--active_name', default='tanh'
    )
    parser.add_argument(
        '--save_name', default='./result/Sirenlb.pkl'
    )
    parser.add_argument(
        '--save_lossfile', default='./data/Sirenlbloss1.csv'
    )
    parser.add_argument(
        '--save_sigmafile', default='./data/Sirensigma.csv'
    )
    parser.add_argument(
        '--save_omegafile', default='./data/Sirenomega.csv'
    )
    parser.add_argument(
        '--x_left', default=-1., type=float
    )
    parser.add_argument(
        '--x_right', default=1., type=float
    )
    parser.add_argument(
        '--t_left', default=0., type=float
    )
    parser.add_argument(
        '--t_right', default=1., type=float
    )
    return parser
