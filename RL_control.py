import numpy as np
import theano
import theano.tensor as T
from sklearn.model_selection import train_test_split
import lasagne
from lasagne.utils import floatX

import time
import random
from collections import deque
from theano.tensor.shared_randomstreams import RandomStreams
from numpy import genfromtxt

import pickle
import csv
from theano.tensor import _shared
import time

import matplotlib.pyplot as plt

import os

from pynput import keyboard 
import time
import numpy as np

from base_method import * 
import matplotlib.pyplot as plt

from threading import Thread
from scipy.interpolate import interp1d
import subprocess 


dir_main = "/home/atharva/Morphing Wing_Python Code/Python Data/Apr2022/"
sub_dir = "050422/"
sub_sub_dir = "Human_exp/"
ver1 = "w1_w2" 
ver2 = "w2_w1"
ver = "_v4"
aoa = "175"
velo = "75"
dir = dir_main + sub_dir + sub_sub_dir
dir1 = dir + "Human_n" + aoa + "_v_" + velo + ver 
dir2 = dir + "Human_n" + aoa + "_v_" + velo + ver 
save_dat = 1
cond = 1


with open(dir1 + "_Data_human_auto_n" + aoa + "_v_" + velo + ".pkl",'rb') as f:  # Python 3: open(..., 'rb')
        state1_array,time_array,speed_array,Drag_array,Lift_array,data_ch6_ar,data_ch11_ar,firing_rate_2_array,firing_rate_1_array,Vact_array,rst_array,strexp_array,clk100_array,current_out2_array,current_out1_array,state_input1_array = pickle.load(f)

inds = 20000
inde = -1
state_max = 0.42
state_min = 0.31
scale_smax = 80

Vact = np.array(Vact_array[inds:inde])
DLRatio = np.array(state1_array[inds:inde])
  
V0 =[(Vact>=-0.05) & (Vact<0.05)]
V01 =[(Vact>=0.05) & (Vact<0.15)]
V02=[(Vact>=0.15) & (Vact<0.25)]
V03 =[(Vact>=0.25) & (Vact<0.35)]
V04 =[(Vact>=0.35) & (Vact<0.45)]
V05 = [(Vact>=0.45)  & (Vact<0.55)]
V06 = [(Vact>=0.55)  & (Vact<0.65)]
V07 = [(Vact>=0.65)  & (Vact<0.75)]
V08 = [(Vact>=0.75)  & (Vact<0.85)]
V09 = [(Vact>=0.85)  & (Vact<0.95)]
V1 = [(Vact>=0.95)  & (Vact<1.05)]
V11 = [(Vact>=1.05)  & (Vact<1.15)]
V12 = [(Vact>=1.15)  & (Vact<1.25)]
V13 = [(Vact>=1.25)  & (Vact<1.35)]
V14 = [(Vact>=1.35)  & (Vact<1.45)]
V15 = [(Vact>=1.45)  & (Vact<1.55)]
V16 = [(Vact>=1.55)  & (Vact<1.65)]
V17 = [(Vact>=1.65)  & (Vact<1.75)]
V18 = [(Vact>=1.75)  & (Vact<1.85)]
V19 = [(Vact>=1.85)  & (Vact<1.95)]
V2 = [(Vact>=1.95)  & (Vact<2.05)]
V21 = [(Vact>=2.05)  & (Vact<2.15)]
V22 = [(Vact>=2.15)  & (Vact<2.25)]
V23 = [(Vact>=2.25)  & (Vact<2.35)]
V24 = [(Vact>=2.35)  & (Vact<2.45)]
V25 = [(Vact>=2.45)  & (Vact<2.55)]
V26 = [(Vact>=2.55)  & (Vact<2.65)]
V27 = [(Vact>=2.65)  & (Vact<2.75)]
V28 = [(Vact>=2.75)  & (Vact<2.85)]
V29 = [(Vact>=2.85)  & (Vact<2.95)]
V3 = [(Vact>=2.95)  & (Vact<3.05)]
V31 = [(Vact>=3.05)  & (Vact<3.15)]
V32 = [(Vact>=3.15)  & (Vact<3.25)]
V33 = [(Vact>=3.25)  & (Vact<3.35)]
V34 = [(Vact>=3.35)  & (Vact<3.45)]
V35 = [(Vact>=3.45)  & (Vact<3.55)]
V36 = [(Vact>=3.55)  & (Vact<3.65)]
V37 = [(Vact>=3.65)  & (Vact<3.75)]
V38 = [(Vact>=3.75)  & (Vact<3.85)]
V39 = [(Vact>=3.85)  & (Vact<3.95)]
V4 = [(Vact>=3.95)  & (Vact<4.05)]
V41 = [(Vact>=4.05)  & (Vact<4.15)]
V42 = [(Vact>=4.15)  & (Vact<4.25)]
V43 = [(Vact>=4.25)  & (Vact<4.35)]
V44 = [(Vact>=4.35)  & (Vact<4.45)]
V45 = [(Vact>=4.45)  & (Vact<4.55)]
V46 = [(Vact>=4.55)  & (Vact<4.65)]
V47 = [(Vact>=4.65)  & (Vact<4.75)]
V48 = [(Vact>=4.75)  & (Vact<4.85)]
V49 = [(Vact>=4.85)  & (Vact<4.95)]
V5 = [(Vact>=4.95)  & (Vact<5.05)]



VList = [V0,V01,V02,V03,V04,V05,V06,V07,V08,V09,V1,V11,V12,V13,V14,V15,V16,V17,V18,V19,V2,V21,V22,V23,V24,V25,V26,V27,V28,V29,V3,V31,V32,V33,V34,V35,V36,V37,V38,V39,V4,V41,V42,V43,V44,V45,V46,V47,V48,V49,V5]
LDRList = []
for i in range(len(VList)):
    LDRList.append(np.mean(DLRatio[VList[i]]))

DLR = np.array(LDRList)


VL = np.arange(0,5.1,0.1)#np.array([0,0.1,0.1,0.1,0.4,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
f = interp1d(VL, DLR,kind='cubic')

data_sim = []
for i in np.arange(0,5.1,0.1):
    #data_sim.append(((f(i)-0.41)/0.093)*80)
    #data_sim.append((((f(i)-0.4187)/0.0846)*75) + 5)
    data_sim.append(f(i))
#print(np.min(np.array(data_sim)))
plt.plot(np.arange(0,5.1,0.1),data_sim)
plt.grid(True)
plt.xlabel('Voltage (V)')
plt.ylabel('State')
plt.title("State vs Voltage")
#plt.show()

ds_0 = 5.0*(f(0.2) - f(0))
ds_02 = 5.0*(f(0.4) - f(0.2))
ds_04 = 5.0*(f(0.6) - f(0.4))
ds_06 = 5.0*(f(0.8) - f(0.6))
ds_08 = 5.0*(f(1) - f(0.8))
ds_1 = 5.0*(f(1.2) - f(1))
ds_12 = 5.0*(f(1.4) - f(1.2))
ds_14 = 5.0*(f(1.6) - f(1.4))
ds_15 = f(2.5) - f(1.5)
ds_2 = f(3) - f(2)
ds_25 = f(3.5) - f(2.5)
ds_3 = f(4) - f(3)
ds_35 = f(4.5) - f(3.5)
ds_4 = f(5) - f(4)
ds_45 = 2.0*(f(5) - f(4.5))
ds_48 = 5.0*(f(5) - f(4.8))
ds_5 = 0

ds_list = np.array([ds_0,ds_02,ds_04,ds_06,ds_08,ds_1,ds_12,ds_14,ds_15,ds_2,ds_25,ds_3,ds_35,ds_4,ds_45,ds_48,ds_5])
V_list = np.array([0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.5,2,2.5,3,3.5,4,4.5,4.8,5])
f2 = interp1d(V_list, ds_list,kind='cubic')
f2 = interp1d(V_list, ds_list,kind='cubic')
data_sim = []
for i in np.arange(0,5.1,0.2):
    data_sim.append(f2(i))

plt.figure()
plt.plot(np.arange(0,5.1,0.2),data_sim)
plt.grid(True)
plt.xlabel('Voltage (V)')
plt.ylabel('ds')
plt.title("ds vs Voltage")

plt.figure()
plt.plot(V_list,ds_list)
plt.grid(True)
plt.xlabel('Voltage (V)')
plt.ylabel('ds_raw')
plt.title("ds_raw vs Voltage")

Vinit = 1.4
Vact_1 = Vinit
s1_old = f(Vact_1)
dVact = 0.2
HIDDEN_UNITS = 4
#STATE_MAX = 150
#STATE_LOW = 30
STATE_HIGH = 60
EXP_TIME = 1001
TOTAL_ITER = 100
PARENT_DIR = "/home/atharva/Morphing Wing_Python Code/Python Data/RL_Data/220422/RL_sim_24/"
DT_SIM = 0.05

REWARD_NEG = -1000.0


alphaf = 3
ALPHA_LOW_1 = -0.1 * alphaf
ALPHA_HIGH_1 = 0.1 * alphaf
betaf = 20
gammaf = 0.5
slope_Vact = 5e-2
tarr_a1 = 0.0
tarr_a2 = 0.0
state_reg = 0.0

alpha_arr_a1 = alphaf

def save_or_load_model_params(save, network, num):
    if save:
        params = lasagne.layers.get_all_param_values(network)
        with open(PARENT_DIR + "params" + str(num) + ".pickle", 'wb') as f:
            #Pour model parameters into the file
            pickle.dump(params, f,protocol=2)
    else:
        with open(PARENT_DIR + "params" + str(num) + ".pickle", 'rb') as f:
        #with open('params10.pickle', 'rb') as f:
        #with open('params2_300_1e_3.pickle', 'rb') as f:
            params = pickle.load(f)
        return params

def save_or_load_model_params_iter(save, network ,filename):
    if save:
        params = lasagne.layers.get_all_param_values(network)
        with open(filename,'wb') as f:
            #Pour model parameters into the file
            pickle.dump(params, f,protocol=2)
    else:
        with open(filename, 'rb') as f:
        #with open('params10.pickle', 'rb') as f:
        #with open('params2_300_1e_3.pickle', 'rb') as f:
            params = pickle.load(f)
        return params

def reset_weights(network):
    params = lasagne.layers.get_all_params(network, trainable=True)
    for v in params:
        val = v.get_value()
        if(len(val.shape) < 2):
            v.set_value(lasagne.init.Constant(0.0)(val.shape))
        else:
            v.set_value(lasagne.init.Uniform()(val.shape))

def reward_scheme(state_vector):
    global scale_smax,state_max,state_min
    ss = ((((state_vector)-state_min)/(state_max - state_min)) * scale_smax)
    #ss = (((state_vector/scale_smax)*(state_max - state_min)) + state_min)
    return -1*np.sum(ss,axis=1)

def policy_network(input_var=None):

    l_in = lasagne.layers.InputLayer(shape=(None, 1),
                                     input_var=input_var)
    
    l_hid_1 = lasagne.layers.DenseLayer(
        l_in, num_units=HIDDEN_UNITS,
        nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.Uniform())
    
    l_out = lasagne.layers.DenseLayer(
        l_hid_1, num_units=3,
        nonlinearity=lasagne.nonlinearities.softmax,W=lasagne.init.Uniform())
    
    return l_out

input_var = T.col('inputs_t',dtype = theano.config.floatX)
G_var = T.vector('G_t',dtype = theano.config.floatX)
a_var = T.lvector('action_index_t')
agent = policy_network(input_var)
probs_softmax = lasagne.layers.get_output(agent)

prediction_fn = theano.function([input_var],probs_softmax)

#print(prediction_fn(np.array([[4.66],[10.55],[1.3]])))

loss = -G_var * T.log(T.clip(probs_softmax[:,a_var],1e-7,(1-1e-7)))
loss = loss.mean()

def inference(x):
    
    probs = np.squeeze(prediction_fn(x))# lasagne.layers.get_output(agent,x).eval())
    if (np.sum(np.isnan(probs))>0):
        probs = np.array([1.0/3.0,1.0/3.0,1.0/3.0])
    action = np.random.choice(3,1,p=probs)

    return action

#print(inference(np.array([[4.66],[10.66],[5.33]])))

def dsigmoid(ts,alphas,betas,gammas):
    texp = -1*np.multiply(betas,(ts-gammas))
    dsig_num = np.multiply(np.multiply(alphas,betas),np.exp(texp))
    dsig_den = np.square((1 + np.exp(texp)))
    dsig = np.divide(dsig_num,dsig_den)
    return dsig

def sigmoid(ts,alphas,betas,gammas):
    texp = -1*np.multiply(betas,(ts-gammas))
    sig_num = alphas
    sig_den = (1 + np.exp(texp))
    sig = np.divide(sig_num,sig_den)
    return sig

def sim_env_wing(act_1,flag_act,act_diff,state,init):
    global state_reg,slope_Vact,Vact_1,f,dVact,s1_old,f2,tarr_a1,alpha_arr_a1,tarr_a2,alpha_arr_a2,ALPHA_LOW_1,ALPHA_HIGH_1,DT_SIM, alphaf, betaf, gammaf

    #s1_old = Vact_1

    '''if(act_1 == 0):
        Vact_1 = Vact_1 + dVact
    elif(act_1 == 1):
        Vact_1 = Vact_1 - dVact
    elif(act_1 == 2):
        Vact_1 = Vact_1'''
    
    '''if(Vact_1 > 5):
        Vact_1 = 5
    elif(Vact_1 < 1):
        Vact_1 = 1'''
    ds1_a1_net = np.NaN
    ds1_a2_net = np.NaN
    
    #print(init)
    if(init == 1):
        alphaf = 0
        state_reg = f(Vact_1)
    else:
        alphaf = (f(Vact_1)-state_reg) 
    
    #print(act_diff)
    ALPHA_LOW_1 = -0.1 * abs(alphaf)
    ALPHA_HIGH_1 = 0.1 * abs(alphaf)

    if (act_diff < 0):
        tarr_a1 = tarr_a1 + DT_SIM
        alpha_arr_a1 = (alphaf) + np.random.uniform(ALPHA_LOW_1,ALPHA_HIGH_1)
        ds1_a1 = sigmoid(tarr_a1,alpha_arr_a1,betaf,gammaf)
        ds1_a1_net = np.nansum(ds1_a1)

    if (act_diff > 0):
        tarr_a2 = tarr_a2 + DT_SIM
        alpha_arr_a2 = (alphaf) + np.random.uniform(ALPHA_LOW_1,ALPHA_HIGH_1)
        ds1_a2 = sigmoid(tarr_a2,alpha_arr_a2,betaf,gammaf)
        ds1_a2_net = np.nansum(ds1_a2)
    
    if(flag_act == 1):
        tarr_a1 = 0.0
        tarr_a2 = 0.0
        state_reg = state
        Vact_1 = Vact_1 + (slope_Vact * act_diff)

    s1_net = np.nansum(np.array([ds1_a1_net, ds1_a2_net]))
    #print(ds1_net)
    state = state_reg + s1_net


    return np.reshape(state,(1,1))


def training_data_sim(gamma=0.99):
    global Vact_1,f,s1_old,Vinit
    Vact_1 = Vinit
    s1_old = f(Vact_1)
    Vact_max = 5
    Vact_min = 1.8
    state_h = 0.40
    state_l = 0.315
    flag_act = 0
    states_init = np.zeros((1,1))
    states_init[0,0] = np.reshape(np.array(f(Vact_1)),(1,1))
    #print(states_init)
    state_list = states_init
    state_new = states_init
    #print(state_list.shape)
    done = 0
    act1 = 0
    act2 = 0
    act_diff = 0
    crash_i = 0
    init = 1
    #state_list.append(states_init)
    for i in range(EXP_TIME):

        if(i%20 == 0):
            flag_act = 1
        else:
            flag_act = 0

        if(Vact_1 <= 1 or Vact_1 >= 5):
            crash_i = crash_i + 1
        if(crash_i !=0 and (Vact_1 > 1 and Vact_1 < 5)):
            crash_i =0
        if(crash_i == 1):
            done = 2
            reward_list = np.concatenate((reward_list,np.array(REWARD_NEG).reshape(1)),axis=0)
            break
        if(crash_i !=2):
            #print(i)
            #print(state_new)
            action = inference(state_new)
            if(action == 0):
                act1 = act1 + 1
            elif(action == 1):
                act2 = act2 + 1
            if(i%20 == 0):
                act_diff = act1 - act2
            state_new = sim_env_wing(action,flag_act,act_diff,state_new,init)
            init = 0
            if(i%20 == 0):
                act1 = 0
                act2 = 0
            if i==0:
                action_list = action
            elif (i==(EXP_TIME-1)):
                done = 0
            else:
                action_list = np.concatenate((action_list,action),axis=0)
                state_list = np.concatenate((state_list,state_new),axis=0)
                #print(state_list.shape)
        #print(i)
        if(Vact_1 > 5):
            Vact_1 = 5
        elif(Vact_1 < 1):
            Vact_1 = 1
        if i!=0:
            if i==1:
                reward_list = reward_scheme(state_list[i,:].reshape((1,1)))
            else:
                reward_list = np.concatenate((reward_list,reward_scheme(state_new)),axis=0)
    
        
    G = 0
    acc_rwd = []
    #print(reward_list.shape)
    for r in reward_list[::-1]:
        G = gamma * G + r
        acc_rwd.append(G)
    acc_rwd.reverse()
    acc_rwd = np.array(acc_rwd)
    return state_list,action_list,acc_rwd,done

def policy_update_sim(resume_training=False):
    global Vact_1
    lr_list = np.array([5e-6,1e-5,5e-5,1e-4,5e-4])
    #lr_list = np.array([5e-4,1e-3,5e-3,1e-2,5e-2])
    #lr_list = np.array([1e-5,5e-5,1e-4,5e-4,1e-3])#,5e-3,1e-2,5e-2])
    #lr_list = np.array([5e-4,7e-4,1e-3])#,5e-3,1e-2,5e-2])
    #lr_list = np.array([5e-2,1e-1,5e-1,1e-0,5e-0])
    #lr_list = np.array([5e-8,1e-7,5e-7,1e-6,5e-6])
    gamma_list = np.array([0.8,0.9,0.95,0.96,0.97,0.98,0.99])
    #gamma_list = np.array([0.6,0.65,0.7,0.75,0.8])
    #gamma_list = np.array([0.4,0.45,0.5,0.55,0.6])
    #gamma_list = np.array([0.2,0.25,0.3,0.35,0.4])
    lr_num = lr_list.shape[0]
    gamma_num = gamma_list.shape[0]
    learn_time_all = np.zeros((TOTAL_ITER,lr_num,gamma_num))
    loss_all = np.zeros((TOTAL_ITER,lr_num,gamma_num))
    performance_all = np.zeros((TOTAL_ITER,lr_num,gamma_num))
    episode_length_all = np.zeros((TOTAL_ITER,lr_num,gamma_num))
    done_flag_all = np.zeros((TOTAL_ITER,lr_num,gamma_num))
    Vact_all = np.zeros((TOTAL_ITER,lr_num,gamma_num))
    state_fin_all = np.zeros((TOTAL_ITER,lr_num,gamma_num))
    for update_iter in range(TOTAL_ITER):
        for lr_its in range(lr_num):
            for gamma_its in range(gamma_num):
                #Vact_1 = 4
                start_time = time.time()
                lr = lr_list[lr_its]
                gamma = gamma_list[gamma_its]
                #states,actions,acc_rwd = load_data_tr(lr,num_epochs,batch_size,ITER_NUM)
                sub_dir = "params_lr_" + str(lr) + "_gamma_" + str(gamma) + "_bs_1"
                filename_old = PARENT_DIR + sub_dir + "/params_latest.pickle"

                if (update_iter == 0):
                  reset_weights(agent)
                  model_path = os.path.join(PARENT_DIR, sub_dir)
                  isdirq = os.path.isdir(model_path)
                  if (not isdirq):
                      os.mkdir(model_path)
                  #params = save_or_load_model_params(1,agent,num)
                  file_check = os.path.exists(filename_old)
                  if ((file_check is False) or (resume_training is False)):
                      save_or_load_model_params_iter(1,agent,filename_old)
                    
                params = save_or_load_model_params_iter(0,agent,filename_old)
                #params = save_or_load_model_params_iter(0,agent,filename_old)
                #print(params)
                lasagne.layers.set_all_param_values(agent, params)
                #lasagne.layers.set_all_param_values(actor_model, weights)
                #print(num_epochs)

                weights = lasagne.layers.get_all_params(agent,trainable=True)
                updates = lasagne.updates.adam(loss, weights, learning_rate=lr)
                train_fn = theano.function([input_var, G_var,a_var], loss, updates=updates)
                
                if (np.sum(np.isnan(params[0]))==0):

                    #sub_dir = "params_lr_" + str(lr) + "_epochs_" + str(num_epochs) + "_bs_" + str(batch_size)
                    states,actions,acc_rwd,done_flag = training_data_sim(gamma)

                    acc_rwd_mean = np.mean(acc_rwd)
                    
                    if (done_flag==0):
                        drone_status = "completing episode"
                    elif (done_flag==1):
                        drone_status = "reaching target"
                    else:
                        drone_status = "crashing"

                    performance_all[update_iter,lr_its,gamma_its] = -np.mean(np.sum(np.abs(states),axis=1))
                    
                    episode_length = states.shape[0]*DT_SIM
                    episode_length_all[update_iter,lr_its,gamma_its] = episode_length
                    done_flag_all[update_iter,lr_its,gamma_its] = done_flag
                    Vact_all[update_iter,lr_its,gamma_its] = Vact_1
                    state_fin_all[update_iter,lr_its,gamma_its] = states[-1,:]

                    train_err = 0
                    train_batches = 0
                    for its in range(states.shape[0]):
                        state = states[its,:].reshape((1,1)) 
                        action = actions[its].reshape(1)
                        ac_rwd = (acc_rwd[its]-acc_rwd_mean).reshape(1)
                        train_err += train_fn(state, ac_rwd, action)
                        train_batches += 1
                    

                    save_or_load_model_params_iter(1,agent,filename_old)
                    loss_all[update_iter,lr_its,gamma_its] = train_err / train_batches
                    end_time = time.time()
                    learn_time_all[update_iter,lr_its,gamma_its] = end_time - start_time
                    #params = save_or_load_model_params_iter(0,actor_model,filename)
                    if ((update_iter%10 == 0) | (update_iter==(TOTAL_ITER-1))):
                        print("Learning Rate = {}, gamma = {} : Episode {} took {:.3f}s".format(lr,gamma,
                                update_iter, end_time - start_time))
                        print("  training loss:\t\t{:.6f}".format(loss_all[update_iter,lr_its,gamma_its]))
                        print("  Mean Reward:\t\t{:.6f} for episode length of {:.2f} s with drone {}".format(performance_all[update_iter,lr_its,gamma_its],episode_length_all[update_iter,lr_its,gamma_its],drone_status))
                        print("  Final Vact:\t\t{:.6f}".format(Vact_1))
                        print("  Final state:\t\t{:.6f}".format(states[-1,0]))
                        filename_loss = PARENT_DIR + sub_dir + "/loss_latest.pickle"
                        with open(filename_loss,'wb') as f:
                            #Pour model parameters into the file
                            pickle.dump(loss_all, f,protocol=2)

                        filename_performance = PARENT_DIR + sub_dir + "/performance_latest.pickle"
                        with open(filename_performance,'wb') as f:
                            #Pour model parameters into the file
                            pickle.dump([performance_all,episode_length_all,done_flag_all,learn_time_all,Vact_all,state_fin_all], f,protocol=2)
    #return loss_all

policy_update_sim()
plt.show()