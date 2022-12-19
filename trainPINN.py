import sys
import torch
import argparse
from torch.autograd import grad
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PINN import *
# Run ex) python3 trainPINN.py --train_data_path training_data.csv --lr 1e-3 --save_path exps/exp1

parser = argparse.ArgumentParser(description = "MAS480(C) PINN Training");

## Training details
parser.add_argument('--train_data_path',     type=str,   default="training_data.csv",   help='Absolute path to the training dataset');
parser.add_argument('--test_data_path',     type=str,   default="testing_data.csv",   help='Absolute path to the testing dataset');
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for prediction and losses');
parser.add_argument('--lr',      type=float, default=0.001,  help='Learning rate');

args = parser.parse_args();

device = torch.device('cuda:1') # This is the GPU ID given for my eelab server. 

def mysplit(lines):
    for i, line in enumerate(lines):
        if i==0:
            t = line.split(",")
            for j in range(len(t)):
                t[j] = float(t[j])
        elif i==1:
            S = line.split(",")
            for j in range(len(S)):
                S[j] = float(S[j])
        elif i==2:
            I = line.split(",")
            for j in range(len(I)):
                I[j] = float(I[j])
        elif i==3:
            R = line.split(",")
            for j in range(len(R)):
                R[j] = float(R[j])
        elif i==4:
            V = line.split(",")
            for j in range(len(V)):
                V[j] = float(V[j])
        elif i==5:
            D = line.split(",")
            for j in range(len(D)):
                D[j] = float(D[j])
        else:
            raise Exception("Data is wrong")
    data = [t, S, I, R, V, D]
    return data

def myplot_loss(losses):
    plt.plot(losses, color = 'teal')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig(args.save_path + "Losses.png")
    return

def myplot_prediction(label, prediction):

    [S_pred_list, I_pred_list, R_pred_list, V_pred_list, D_pred_list] = prediction
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

    ax.plot(label[0], label[1], 'pink', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(label[0], S_pred_list[0].detach().cpu().numpy(), 'red', alpha=0.9, lw=2, label='Susceptible Prediction', linestyle='dashed')

    ax.plot(label[0], label[2], 'violet', alpha=0.5, lw=2, label='Infected')
    ax.plot(label[0], I_pred_list[0].detach().cpu().numpy(), 'dodgerblue', alpha=0.9, lw=2, label='Infected Prediction', linestyle='dashed')

    ax.plot(label[0], label[3], 'darkgreen', alpha=0.5, lw=2, label='Recovered')
    ax.plot(label[0], R_pred_list[0].detach().cpu().numpy(), 'green', alpha=0.9, lw=2, label='Recovered Prediction', linestyle='dashed')

    ax.plot(label[0], label[4], 'blue', alpha=0.5, lw=2, label='Vaccinated')
    ax.plot(label[0], V_pred_list[0].detach().cpu().numpy(), 'teal', alpha=0.9, lw=2, label='Vaccinated Prediction', linestyle='dashed')

    ax.plot(label[0], label[5], 'cyan', alpha=0.5, lw=2, label='Dead')
    ax.plot(label[0], D_pred_list[0].detach().cpu().numpy(), 'orange', alpha=0.9, lw=2, label='Dead Prediction', linestyle='dashed')

    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(visible=True, which='major', c='black', lw=0.2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()
    plt.savefig(args.save_path + "Prediction.png")
    return

def myplot_prediction2(label, prediction):

    [S_pred_list, I_pred_list, R_pred_list, V_pred_list, D_pred_list] = prediction
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

    ax.plot(label[0], label[2], 'violet', alpha=0.5, lw=2, label='Infected')
    ax.plot(label[0], I_pred_list[0].detach().cpu().numpy(), 'dodgerblue', alpha=0.9, lw=2, label='Infected Prediction', linestyle='dashed')

    ax.plot(label[0], label[3], 'darkgreen', alpha=0.5, lw=2, label='Recovered')
    ax.plot(label[0], R_pred_list[0].detach().cpu().numpy(), 'green', alpha=0.9, lw=2, label='Recovered Prediction', linestyle='dashed')

    ax.plot(label[0], label[5], 'cyan', alpha=0.5, lw=2, label='Dead')
    ax.plot(label[0], D_pred_list[0].detach().cpu().numpy(), 'orange', alpha=0.9, lw=2, label='Dead Prediction', linestyle='dashed')

    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(visible=True, which='major', c='black', lw=0.2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()
    plt.savefig(args.save_path + "Prediction2.png")
    return

def myplot_prediction_test(label, prediction):

    [S_pred_list, I_pred_list, R_pred_list, V_pred_list, D_pred_list] = prediction
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

    ax.plot(label[0], label[1], 'pink', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(label[0], S_pred_list.detach().cpu().numpy(), 'red', alpha=0.9, lw=2, label='Susceptible Prediction', linestyle='dashed')

    ax.plot(label[0], label[2], 'violet', alpha=0.5, lw=2, label='Infected')
    ax.plot(label[0], I_pred_list.detach().cpu().numpy(), 'dodgerblue', alpha=0.9, lw=2, label='Infected Prediction', linestyle='dashed')

    ax.plot(label[0], label[3], 'darkgreen', alpha=0.5, lw=2, label='Recovered')
    ax.plot(label[0], R_pred_list.detach().cpu().numpy(), 'green', alpha=0.9, lw=2, label='Recovered Prediction', linestyle='dashed')

    ax.plot(label[0], label[4], 'blue', alpha=0.5, lw=2, label='Vaccinated')
    ax.plot(label[0], V_pred_list.detach().cpu().numpy(), 'teal', alpha=0.9, lw=2, label='Vaccinated Prediction', linestyle='dashed')

    ax.plot(label[0], label[5], 'cyan', alpha=0.5, lw=2, label='Dead')
    ax.plot(label[0], D_pred_list.detach().cpu().numpy(), 'orange', alpha=0.9, lw=2, label='Dead Prediction', linestyle='dashed')

    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(visible=True, which='major', c='black', lw=0.2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()
    plt.savefig(args.save_path + "Prediction_test.png")
    return

def myplot_prediction_test2(label, prediction):

    [S_pred_list, I_pred_list, R_pred_list, V_pred_list, D_pred_list] = prediction
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

    ax.plot(label[0], label[2], 'violet', alpha=0.5, lw=2, label='Infected')
    ax.plot(label[0], I_pred_list.detach().cpu().numpy(), 'dodgerblue', alpha=0.9, lw=2, label='Infected Prediction', linestyle='dashed')

    ax.plot(label[0], label[3], 'darkgreen', alpha=0.5, lw=2, label='Recovered')
    ax.plot(label[0], R_pred_list.detach().cpu().numpy(), 'green', alpha=0.9, lw=2, label='Recovered Prediction', linestyle='dashed')

    ax.plot(label[0], label[5], 'cyan', alpha=0.5, lw=2, label='Dead')
    ax.plot(label[0], D_pred_list.detach().cpu().numpy(), 'orange', alpha=0.9, lw=2, label='Dead Prediction', linestyle='dashed')

    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(visible=True, which='major', c='black', lw=0.2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()
    plt.savefig(args.save_path + "Prediction_test2.png")
    return

def relative_l2_error(label, prediction):
    label_sirvd = label[1:]
    prediction_sirvd = prediction
    err_s = torch.linalg.norm(label_sirvd[0] - prediction_sirvd[0][0].cpu()) / torch.linalg.norm(label_sirvd[0])
    err_i = torch.linalg.norm(label_sirvd[1] - prediction_sirvd[1][0].cpu()) / torch.linalg.norm(label_sirvd[1])
    err_r = torch.linalg.norm(label_sirvd[2] - prediction_sirvd[2][0].cpu()) / torch.linalg.norm(label_sirvd[2])
    err_v = torch.linalg.norm(label_sirvd[3] - prediction_sirvd[3][0].cpu()) / torch.linalg.norm(label_sirvd[3])
    err_d = torch.linalg.norm(label_sirvd[4] - prediction_sirvd[4][0].cpu()) / torch.linalg.norm(label_sirvd[4])
    return err_s, err_i, err_r, err_v, err_d

with open(args.train_data_path, 'r') as file:
    lines = file.readlines()
    lines = list(map(lambda s: s.strip('\n'), lines))
Korea_data = mysplit(lines) # [t, S, I, R, V, D] form
Korea_data = torch.tensor(Korea_data).to(device)
mypinn = PINN(Korea_data[0], Korea_data[1], Korea_data[2], Korea_data[3], \
            Korea_data[4], Korea_data[5]).to(device) # [t, S, I, R, V, D] form

learning_rate = args.lr
optimizer = optim.Adam(mypinn.params, lr = learning_rate)
mypinn.optimizer = optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(mypinn.optimizer, T_0=500, T_mult=2, eta_min=1e-6)
mypinn.scheduler = scheduler

S_pred_list, I_pred_list, R_pred_list, V_pred_list, D_pred_list = mypinn.train(30000) #train
# Draw the plot of loss per epochs
myplot_loss(mypinn.losses[0:])

f = open(args.save_path+"coefficient_l2loss.txt", "w")
# Test data
with open(args.test_data_path, 'r') as file:
    lines = file.readlines()
    lines = list(map(lambda s: s.strip('\n'), lines))
test_data = mysplit(lines) # [t, S, I, R, V, D] form
test_data = torch.tensor(test_data)
test_data_time = test_data[0].to(device)
test_data_time = torch.reshape(test_data_time, (-1, 1))
prediction_normalized = mypinn.net_sirvd(test_data_time)
prediction_normalized = torch.transpose(prediction_normalized, 0, 1)
prediction_test = []
prediction_test.append(mypinn.S_min + (mypinn.S_max - mypinn.S_min) * prediction_normalized[0])
prediction_test.append(mypinn.I_min + (mypinn.I_max - mypinn.I_min) * prediction_normalized[1])
prediction_test.append(mypinn.R_min + (mypinn.R_max - mypinn.R_min) * prediction_normalized[2])
prediction_test.append(mypinn.V_min + (mypinn.V_max - mypinn.V_min) * prediction_normalized[3])
prediction_test.append(mypinn.D_min + (mypinn.D_max - mypinn.D_min) * prediction_normalized[4])

test_data[0] = test_data[0].cpu()
test_err_s, test_err_i, test_err_r, test_err_v, test_err_d = relative_l2_error(test_data, prediction_test)
myplot_prediction_test(test_data, prediction_test)
myplot_prediction_test2(test_data, prediction_test)
f.write("Test: relative L2 error (S): %f\n" % test_err_s)
f.write("Test: relative L2 error (I): %f\n" % test_err_i)
f.write("Test: relative L2 error (R): %f\n" % test_err_r)
f.write("Test: relative L2 error (V): %f\n" % test_err_v)
f.write("Test: relative L2 error (D): %f\n" % test_err_d)
test_avg = (test_err_s+test_err_i+test_err_r+test_err_v+test_err_d)/5
f.write("Test: relative L2 error (average): %f\n" % test_avg)

Korea_data = Korea_data.cpu()
# Draw the plot of predictions with ground_truth
prediction = [S_pred_list, I_pred_list, R_pred_list, V_pred_list, D_pred_list]
myplot_prediction(Korea_data, prediction)
myplot_prediction2(Korea_data, prediction)
Korea_data = list(Korea_data)
# Record the relative L2 loss and coefficients of models
err_s, err_i, err_r, err_v, err_d = relative_l2_error(Korea_data, prediction)
f.write("Training: relative L2 error (S): %f\n" % err_s)
f.write("Training: relative L2 error (I): %f\n" % err_i)
f.write("Training: relative L2 error (R): %f\n" % err_r)
f.write("Training: relative L2 error (V): %f\n" % err_v)
f.write("Training: relative L2 error (D): %f\n" % err_d)
avg = (err_s+err_i+err_r+err_v+err_d)/5
f.write("Training: relative L2 error (average): %f\n" % avg)
f.write("alpha: %f\n" % mypinn.alpha())
f.write("beta: %f\n" % mypinn.beta())
f.write("gamma: %f\n" % mypinn.gamma())
f.write("delta: %f\n" % mypinn.delta())
f.write("sigma: %f\n" % mypinn.sigma())
f.close()