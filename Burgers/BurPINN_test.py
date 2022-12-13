from net import Net, Siren
import torch
import os
#from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from Parser_PINN import get_parser
from PINN_losses import loss_u, loss_f, loss_b, gradients
from burgers_All_plots import Exact_plot, Pred_plot, aberror_plot, residual_plot, pred_plot, test_plot, loss_plot, losscv_plot, sigmma_plot, omega_plot
import csv
import scipy.io
import numpy as np
import sys
sys.path.append("..")

def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.cuda('cpu')
    PINN = Net(seq_net=args.seq_net, activation=args.activation).to(device)
    #PINN.load_state_dict(torch.load(args.load_path))
    optimizer= args.optimizer(PINN.parameters(), args.lr)


   


    # data
    data = scipy.io.loadmat('./burgers_shock.mat')
    Exact = np.real(data['usol']).T
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))# 在水平方向上平铺(25600, 2)
    X_star = X_star.astype(np.float32)
    X_star = torch.from_numpy(X_star).cuda().requires_grad_(True)
    u_star = Exact.flatten()[:, None]
    u_star = u_star.astype(np.float32)
    u_star = torch.from_numpy(u_star).cuda().requires_grad_(True)

    MSE_PDE = []
    MSE_BC = []
    MSE_U = []
    MSE_All = []


    if args.train_plot:
        for epoch in range(args.epochs):
            optimizer.zero_grad()
          
            # N_u Noiseles Data
            noise = 0.0
            idx = np.random.choice(X_star.shape[0], args.N_u, replace=False)
            X_u_train = X_star[idx, :].requires_grad_(True)  # (2000, 2)
            x_u = X_u_train[:, 0:1].requires_grad_(True)
            
            t_u = X_u_train[:, 1:2].requires_grad_(True)
            u_train = u_star[idx, :]  # (2000, 1)

            u_f = PINN(torch.cat([x_u, t_u], dim=1))

            

            mse_u = loss_u(u_f, u_train)
            f, mse_f = loss_f(u_f, x_u, t_u, args)

            #boundary
            x_1 = ((args.x_left + args.x_right) / 2 + (args.x_right - args.x_left) *
                   (torch.rand(size=(args.N_b, 1), dtype=torch.float, device=device) - 0.5)
                   ).requires_grad_(True)
            t_1 = ((args.t_left + args.t_right) / 2 + (args.t_right - args.t_left) *
                   (torch.rand(size=(args.N_b, 1), dtype=torch.float, device=device) - 0.5)
                   ).requires_grad_(True)
            t_2 = ((args.t_left + args.t_right) / 2 + (args.t_right - args.t_left) *
                   (torch.rand(size=(args.N_b, 1), dtype=torch.float, device=device) - 0.5)
                   ).requires_grad_(True)

            b_0 = (args.x_left * torch.ones_like(x_1)
                   ).requires_grad_(True)
            b_1 = (args.x_right * torch.ones_like(x_1)
                   ).requires_grad_(True)
            b_2 = (args.t_left * torch.ones_like(t_1)
                   ).requires_grad_(True)

            u_b_1 = PINN(torch.cat([x_1, b_2], dim=1))  # u(x,0)
            u_b_2 = PINN(torch.cat([b_0, t_1], dim=1))  # u(-1,t)
            u_b_3 = PINN(torch.cat([b_1, t_2], dim=1))  # u(1,t)

            mse_b = loss_b(u_b_1, u_b_2, u_b_3, x_1)

            # loss
            loss = (args.w_b * mse_b + args.w_u * mse_u + args.w_f * mse_f)*1e3
            


            if ((epoch + 1) % 500 == 0):
                print(
                    'epoch:{:05d}, myf: {:.06e}, myBC: {:.06e}, myu: {:.06e},myloss: {:.06e}'.format(
                        epoch, mse_f.item(), mse_b.item(), mse_u.item(), loss.item()
                    )
                )
          

            MSE_PDE.append(mse_f.item())
            MSE_BC.append(mse_b.item())
            MSE_U.append(mse_u.item())
            MSE_All.append(loss.item())

           

            loss.backward()
            optimizer.step()
            

    #         if ((epoch + 1) % 10000 == 0):
    #             #predict
    #             x_pred = X_star[:, 0:1]
    #             t_pred = X_star[:, 1:2]
    #             u_pred = PINN(torch.cat([x_pred, t_pred], dim=1))
    #
    #             loss_test = loss_u(u_pred, u_star)
    #             r_f, r_f_test = loss_f(u_pred, x_pred, t_pred, args)
    #
    #             rf_pred = griddata(X_star.cpu().detach().numpy(), r_f.cpu().detach().numpy().flatten(), (X, T),
    #                                method='cubic')
    #             #pred_plot(epoch, X_star, X, T, Exact, u_pred, rf_pred)
    #
    #             print('epoch:{},test loss:{}'.format(epoch + 1, loss_test))
    #             Exact_plot(epoch, Exact)
    #             Pred_plot(epoch, X_star, X, T, u_pred)
    #             aberror_plot(epoch, X_star, X, T, Exact, u_pred)
    #             residual_plot(epoch, X_star, X, T, r_f)
    #             # test_plot(epoch, x, Exact, X_star, X, T, u_pred)
    #
    #
    #             x_25 = x.astype(np.float32)
    #             x_25 = torch.from_numpy(x_25).cuda().requires_grad_(True)
    #             t_25 = (0.25 * torch.ones_like(x_25)).requires_grad_(True)
    #             u_25 = PINN(torch.cat([x_25, t_25], dim=1))
    #             t_50 = (0.5 * torch.ones_like(x_25)).requires_grad_(True)
    #             u_50 = PINN(torch.cat([x_25, t_50], dim=1))
    #             t_75 = (0.75 * torch.ones_like(x_25)).requires_grad_(True)
    #             u_75 = PINN(torch.cat([x_25, t_75], dim=1))
    #             test_plot(epoch, x, Exact, u_25, u_50, u_75)
    #
    # torch.save(PINN.state_dict(),args.load_path)
    # loss_plot(args.epochs, MSE_PDE, MSE_BC, MSE_U)
    # losscv_plot(args.epochs, MSE_PDE, MSE_BC, MSE_U)
    # with open(args.save_lossfile, 'a+') as csvFile:
    #     # 写入多行用writerows
    #     writer = csv.writer(csvFile)
    #     writer.writerow(MSE_PDE)
    #     writer.writerow(MSE_BC)
    #     writer.writerow(MSE_U)
    #     writer.writerow(MSE_All)
   





if __name__ == '__main__':
    parser_PINN = get_parser()
    args = parser_PINN.parse_args()

    import time

    time_start = time.time()  # 开始计时
    train(args)
    time_end = time.time()  # 结束计时

    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')
