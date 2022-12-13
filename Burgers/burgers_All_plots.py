import torch
import sys
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("..")
def Exact_plot(epoch, Exact):
    fig, ax = plt.subplots()
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    h0 = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow',
                      extent=[0.0, 1.0, -1.0, 1.0],
                      origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h0, cax=cax)
    ax.set_xlabel('$t$', font2)
    ax.set_ylabel('$x$', font2)
    # ax[0].set_aspect('equal', 'box')
    ax.set_title('Exact u(t,x)', fontsize=16)
    # fig = plt.gcf()
    # fig.savefig('./Pred_plot/Epoch_({})_Exact_plot.pdf'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    # fig = plt.gcf()
    # fig.savefig('./Pred_plot/Epoch_({})_Exact_plot.png'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    # plt.show()
    fig = plt.gcf()
    fig.savefig('./lbPred_plot/Epoch_({})_Exact_plot.pdf'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    fig = plt.gcf()
    fig.savefig('./lbPred_plot/Epoch_({})_Exact_plot.png'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    plt.show()

def Pred_plot(epoch, xy_1, X1, Y1, U_p):
    fig, ax = plt.subplots()
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    u1_pred = griddata(xy_1.cpu().detach().numpy(), U_p.cpu().detach().numpy().flatten(), (X1, Y1),
                       method='cubic')
    h0 = ax.imshow(u1_pred.T, interpolation='nearest', cmap='rainbow',
                      extent=[0.0, 1.0, -1.0, 1.0],
                      origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h0, cax=cax)
    ax.set_xlabel('$t$', font2)
    ax.set_ylabel('$x$', font2)
    # ax[0].set_aspect('equal', 'box')
    ax.set_title('Predicted u(t,x)', fontsize=16)
    # fig = plt.gcf()
    # fig.savefig('./Pred_plot/Epoch_({})_Pred_plot.pdf'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    # fig = plt.gcf()
    # fig.savefig('./Pred_plot/Epoch_({})_Pred_plot.png'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    # plt.show()
    fig = plt.gcf()
    fig.savefig('./lbPred_plot/Epoch_({})_Pred_plot.pdf'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    fig = plt.gcf()
    fig.savefig('./lbPred_plot/Epoch_({})_Pred_plot.png'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    plt.show()

def pred_plot(epoch, X_star, X, T, Exact, u_pred, rf_pred):
    u1_pred = griddata(X_star.cpu().detach().numpy(), u_pred.cpu().detach().numpy().flatten(), (X, T),
                       method='cubic')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    h0 = ax[0].imshow(Exact.T, interpolation='nearest', cmap='rainbow',
                      extent=[0.0, 1.0, -1.0, 1.0],
                      origin='lower', aspect='auto')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h0, cax=cax)
    ax[0].set_xlabel('$t$')
    ax[0].set_ylabel('$x$')
    # ax[0].set_aspect('equal', 'box')
    ax[0].set_title('Exact u(t,x)', fontsize=15)

    h1 = ax[1].imshow(u1_pred.T, interpolation='nearest', cmap='rainbow',
                      extent=[0.0, 1.0, -1.0, 1.0],
                      origin='lower', aspect='auto')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h1, cax=cax)
    ax[1].set_xlabel('$t$')
    ax[1].set_ylabel('$x$')
    # ax[1].set_aspect('equal', 'box')
    ax[1].set_title('Predicted u(t,x)', fontsize=15)
    # line = np.linspace(x.min(), x.max(), 2)[:, None]
    # ax[1].plot(t[25] * np.ones((2, 1)), line, 'w--', linewidth=2)
    # ax[1].plot(t[50] * np.ones((2, 1)), line, 'w--', linewidth=2)
    # ax[1].plot(t[75] * np.ones((2, 1)), line, 'w--', linewidth=2)
    # plt.tight_layout(pad=1.5)
    fig.tight_layout()
    plt.subplots_adjust(wspace=2.5)

    # fig = plt.gcf()
    # fig.savefig('./Pred_plot/Epoch_({})_Predplot'.format((epoch + 1)))
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    # print("Absolute error", np.abs(Exact.T - u1_pred.T))
    h0 = ax[0].imshow(np.abs(Exact.T - u1_pred.T), interpolation='nearest', cmap='rainbow',
                      extent=[0.0, 1.0, -1.0, 1.0],
                      origin='lower', aspect='auto')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h0, cax=cax)
    ax[0].set_xlabel('$t$')
    ax[0].set_ylabel('$x$')
    # ax[0].set_aspect('equal', 'box')
    ax[0].set_title(' Absolute error', fontsize=15)

    h1 = ax[1].imshow(rf_pred.T, interpolation='nearest', cmap='rainbow',
                      extent=[0.0, 1.0, -1.0, 1.0],
                      origin='lower', aspect='auto')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h1, cax=cax)
    ax[1].set_xlabel('$t$')
    ax[1].set_ylabel('$x$')
    # ax[1].set_aspect('equal', 'box')
    ax[1].set_title('Residual prediction', fontsize=15)
    fig.tight_layout()
    plt.subplots_adjust(wspace=2.5)

    # plt.tight_layout(pad=1.5)

    # fig = plt.gcf()
    # fig.savefig('./Pred_plot/Epoch_({})_Predplot'.format((epoch + 1)))
    plt.show()

def aberror_plot(epoch, xy_1, X1, Y1, Exact, U_p):
    fig, ax = plt.subplots()
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    u1_pred = griddata(xy_1.cpu().detach().numpy(), U_p.cpu().detach().numpy().flatten(), (X1, Y1),
                       method='cubic')
    h0 = ax.imshow(np.abs(Exact.T - u1_pred.T), interpolation='nearest', cmap='rainbow',
                      extent=[0.0, 1.0, -1.0, 1.0],
                      origin='lower', aspect='auto', vmin=0.0, vmax=0.25)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h0, cax=cax)
    ax.set_xlabel('$t$', font2)
    ax.set_ylabel('$x$', font2)
    # ax[0].set_aspect('equal', 'box')
    ax.set_title(' Absolute error', fontsize=16)
    # fig = plt.gcf()
    # fig.savefig('./Pred_plot/Epoch_({})_error_plot.pdf'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    # fig = plt.gcf()
    # fig.savefig('./Pred_plot/Epoch_({})_error_plot.png'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    # plt.show()
    fig = plt.gcf()
    fig.savefig('./lbPred_plot/PINNEpoch_({})_error_plot.pdf'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    fig = plt.gcf()
    fig.savefig('./lbPred_plot/PINNEpoch_({})_error_plot.png'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    plt.show()

def residual_plot(epoch, xy_1, X1, Y1, pred_f):
    rf_pred = griddata(xy_1.cpu().detach().numpy(), pred_f.cpu().detach().numpy().flatten(), (X1, Y1),
                       method='cubic')
    fig, ax = plt.subplots()
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    h0 = ax.imshow(rf_pred.T, interpolation='nearest', cmap='rainbow',
                      extent=[0.0, 1.0, -1.0, 1.0],
                      origin='lower', aspect='auto', vmin=-0.6, vmax=1.0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h0, cax=cax)
    ax.set_xlabel('$t$', font2)
    ax.set_ylabel('$x$', font2)
    # ax[0].set_aspect('equal', 'box')
    ax.set_title('Residual prediction', fontsize=16)
    # fig = plt.gcf()
    # fig.savefig('./Pred_plot/Epoch_({})_residual_plot.pdf'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    # fig = plt.gcf()
    # fig.savefig('./Pred_plot/Epoch_({})_residual_plot.png'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    # plt.show()
    fig = plt.gcf()
    fig.savefig('./lbPred_plot/PINNEpoch_({})_residual_plot.pdf'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    fig = plt.gcf()
    fig.savefig('./lbPred_plot/PINNEpoch_({})_residual_plot.png'.format(epoch + 1), bbox_inches='tight', pad_inches=0.02)
    plt.show()

def test_plot(epoch, x, Exact, u_25, u_50, u_75):
    fig = plt.figure(figsize=(16, 4))
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    ax = fig.add_subplot(131)

    # u_25, u_25_coords = burgers_siren(torch.cat([x_25, t_25], dim=1))
    ax.plot(x, Exact[25, :], 'b-', linewidth=2)
    ax.plot(x, u_25.reshape((-1, 1)).detach().cpu().numpy(), 'r--', linewidth=2)
    ax.set_xlabel('x', font2)
    ax.set_ylabel('u(t,x)', font2)
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('t = 0.25', fontsize=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    # ax = plt.subplot(132)
    ax = fig.add_subplot(132)
    # u_50, u_50_coords = burgers_siren(torch.cat([x_25, t_50], dim=1))
    ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_50.reshape((-1, 1)).detach().cpu().numpy(), 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('x', font2)
    ax.set_ylabel('u(t,x)', font2)
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('t = 0.50', fontsize=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(133)
    # u_75, u_75_coords = burgers_siren(torch.cat([x_25, t_75], dim=1))
    ax.plot(x, Exact[75, :], 'b-', linewidth=2)
    ax.plot(x, u_75.reshape((-1, 1)).detach().cpu().numpy(), 'r--', linewidth=2)
    ax.set_xlabel('x', font2)
    ax.set_ylabel('u(t,x)', font2)
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('t = 0.75', fontsize=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    # fig = plt.gcf()
    # fig.savefig('./Pred_plot/Epoch_({})_testplot.pdf'.format((epoch + 1)), bbox_inches='tight', pad_inches=0.02)
    # fig = plt.gcf()
    # fig.savefig('./Pred_plot/Epoch_({})_testplot.png'.format((epoch + 1)), bbox_inches='tight', pad_inches=0.02)
    # plt.show()
    # fig = plt.gcf()
    # fig.savefig('./lbPred_plot/Epoch_({})_testplot.pdf'.format((epoch + 1)), bbox_inches='tight', pad_inches=0.02)
    # fig = plt.gcf()
    # fig.savefig('./lbPred_plot/Epoch_({})_testplot.png'.format((epoch + 1)), bbox_inches='tight', pad_inches=0.02)
    # plt.show()
    fig = plt.gcf()
    fig.savefig('./lbPred_plot/Siren_({})_testplot.pdf'.format((epoch + 1)), bbox_inches='tight', pad_inches=0.02)
    fig = plt.gcf()
    fig.savefig('./lbPred_plot/Siren_({})_testplot.png'.format((epoch + 1)), bbox_inches='tight', pad_inches=0.02)
    plt.show()


def loss_plot(epoch, MSE_PDE, MSE_BC, MSE_U):
    fig = plt.figure()
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    ax = fig.add_subplot(111)
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    ax.plot(MSE_PDE, 'b-', linewidth=2, label='$Loss_{PDE}$')
    ax.plot(MSE_BC, 'r-', linewidth=2, label='$Loss_{BC}$')
    ax.plot(MSE_U, 'g-', linewidth=2, label='$Loss_{Data}$')
    ax.set_xlabel('iterations', font1)
    ax.set_ylabel('Loss', font1)
    plt.yscale('log')
    plt.ylim(1e-6, 10)
    ax.legend(loc='best', prop=font1)
    # fig = plt.gcf()
    # fig.savefig('./Loss_plot/All_loss{}.pdf'.format(epoch), bbox_inches='tight', pad_inches=0.02)
    # fig = plt.gcf()
    # fig.savefig('./Loss_plot/All_loss{}.png'.format(epoch), bbox_inches='tight', pad_inches=0.02)
    # plt.show()
    fig = plt.gcf()
    fig.savefig('./lbLoss_plot/All_loss{}.pdf'.format(epoch), bbox_inches='tight', pad_inches=0.02)
    fig = plt.gcf()
    fig.savefig('./lbLoss_plot/All_loss{}.png'.format(epoch), bbox_inches='tight', pad_inches=0.02)
    plt.show()

def converge(norm, num_epoch):
    min_norm = np.min(norm, axis=0)
    best_index = norm.index(min_norm)
    min_con = [np.min(norm[:i]) for i in range(1, num_epoch + 1)]
    return min_norm, min_con, best_index


def losscv_plot(epoch, MSE_PDE, MSE_I, MSE_p):
    fig = plt.figure()
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    ax = fig.add_subplot(111)
    min_PDE, res_PDE, x_PDE = converge(MSE_PDE, epoch)
    min_I, res_I, x_I = converge(MSE_I, epoch)
    min_p, res_p, x_p = converge(MSE_p, epoch)
    ax.plot(res_PDE, 'b-', linewidth=2, label='$Loss_{PDE}$')
    ax.plot(res_I, 'r-', linewidth=2, label='$Loss_{BC}$')
    ax.plot(res_p, 'g-', linewidth=2, label='$Loss_{Data}$')
    ax.set_xlabel('iterations', font1)
    ax.set_ylabel('Loss', font1)
    plt.yscale('log')
    plt.ylim(1e-3, 10)
    ax.legend(loc='best', prop=font1)
    # fig = plt.gcf()
    # fig.savefig('./Loss_plot/losscv_{}.pdf'.format(epoch), bbox_inches='tight', pad_inches=0.02)
    # fig = plt.gcf()
    # fig.savefig('./Loss_plot/losscv_{}.png'.format(epoch), bbox_inches='tight', pad_inches=0.02)
    # plt.show()
    fig = plt.gcf()
    fig.savefig('./lbLoss_plot/losscv_{}.pdf'.format(epoch), bbox_inches='tight', pad_inches=0.02)
    fig = plt.gcf()
    fig.savefig('./lbLoss_plot/losscv_{}.png'.format(epoch), bbox_inches='tight', pad_inches=0.02)
    plt.show()

def sigmma_plot(epoch, Sigmma1, Sigmma2, Sigmma3):
    fig = plt.figure()
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    ax = fig.add_subplot(111)
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    ax.plot(Sigmma1, 'b-', linewidth=2, label='$\epsilon_{f}$')
    ax.plot(Sigmma2, 'r-', linewidth=2, label='$\epsilon_{b}$')
    ax.plot(Sigmma3, 'g-', linewidth=2, label='$\epsilon_{d}$')
    ax.set_xlabel('iterations', font2)
    ax.set_ylabel('$\epsilon$', font1)
    plt.yscale('log')
    plt.ylim(1e-3, 10)
    ax.legend(loc='best', prop=font1)

    fig = plt.gcf()
    fig.savefig('./lbLoss_plot/Epsilon_{}.pdf'.format(epoch), bbox_inches='tight', pad_inches=0.02)
    fig = plt.gcf()
    fig.savefig('./lbLoss_plot/Epsilon_{}.png'.format(epoch), bbox_inches='tight', pad_inches=0.02)
    plt.show()

def omega_plot(epoch, Weights1, Weights2, Weights3):
    fig = plt.figure()
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    ax = fig.add_subplot(111)

    ax.plot(Weights1, 'b-', linewidth=2, label='$\omega_{f}$')
    ax.plot(Weights2, 'r-', linewidth=2, label='$\omega_{b}$')
    ax.plot(Weights3, 'g-', linewidth=2, label='$\omega_{d}$')
    ax.set_xlabel('iterations', font2)
    ax.set_ylabel('$\omega$', font1)
    plt.yscale('log')
    plt.ylim(1e-1, 1e5)
    ax.legend(loc='best', prop=font1)

    fig = plt.gcf()
    fig.savefig('./lbLoss_plot/omega_{}.pdf'.format(epoch), bbox_inches='tight', pad_inches=0.02)
    fig = plt.gcf()
    fig.savefig('./lbLoss_plot/omega_{}.png'.format(epoch), bbox_inches='tight', pad_inches=0.02)
    plt.show()

