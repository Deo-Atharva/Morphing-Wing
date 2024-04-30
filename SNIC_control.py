
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
COPYRIGHT AND LICENSE
Copyright © 2019-2021 Rednova Innovations Inc (with updates at http://openOPU.org)
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are 
permitted provided that such redistribution and use is not for commercial and the 
following conditions are met.

Redistributions of source code must retain the above copyright notice, this list of 
conditions and the following disclaimer. 

Redistributions in binary form must reproduce the above copyright notice, this list 
of conditions and the following disclaimer in the documentation and/or other 
materials provided with the distribution.

Neither the name of Rednova Innovations Inc nor the names of its contributors may be 
used to endorse or promote products derived from this software without specific prior 
written permission.

THIS SOFTWARE IS PROVIDED BY REDNOVA INNOVATIONS INC ``AS IS'' AND ANY EXPRESS OR 
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL 
REDNOVA INNOVATIONS INC BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import usb
import numpy as np

from base_method import * 
import matplotlib.pyplot as plt

from threading import Thread
from nptdms import TdmsFile
from scipy.interpolate import interp1d

import time
from PyDAQmx.DAQmxTypes import *
from PyDAQmx import *
from PyDAQmx.DAQmxFunctions import *
from PyDAQmx.DAQmxConstants import *
import subprocess 



# for keyboard
from pynput.keyboard import Listener, Key, KeyCode
from collections import defaultdict
from enum import Enum
import pickle


start_exp = 0
state1 = 0
quit_flag = 0

speed = 0
Lift = 0
Drag = 0
flag_s = 0
flag_a = 0
run_exp = True

# D/L Range: 0.31 to 0.42 

dir_main = "/home/atharva/Morphing Wing_Python Code/Python Data/Mar2022/"
sub_dir = "310322/"
sub_sub_dir = "SNIC_exp/"
ver1 = "w1_w2"
ver2 = "w2_w1"
ver = "_v23"
aoa = "175"
velo = "75"
dir = dir_main + sub_dir + sub_sub_dir
dir1 = dir + "SNIC_real_n" + aoa + "_v_" + velo + ver1 + ver 
dir2 = dir + "SNIC_real_n" + aoa + "_v_" + velo + ver2 + ver
save_dat = 1
cond = 2




def find_device():
    """
    Find FX3 device and the corresponding endpoints (bulk in/out).
    If find device and not find endpoints, this may because no images are programed, we will program image;
    If image is programmed and still not find endpoints, raise error;
    If not find device, raise error.

    :return: usb device, usb endpoint bulk in, usb endpoint bulk out
    """

    # find device
    dev = usb.core.find(idVendor=0x04b4)
    intf = dev.get_active_configuration()[(0, 0)]

    # find endpoint bulk in
    ep_in = usb.util.find_descriptor(intf,
                                     custom_match=lambda e:
                                     usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)

    # find endpoint bulk out
    ep_out = usb.util.find_descriptor(intf,
                                      custom_match=lambda e:
                                      usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT)

    if ep_in is None and ep_out is None:
        print('Error: Cannot find endpoints after programming image.')
        return -1
    else:
        return dev, ep_in, ep_out

def echo_mode():
    # read/write mode
    # write data to FPGA     
    num = 1024 * 40
    np_data = np.random.randint(0, high=255, size = num, dtype=np.uint8)
    wr_data = list(np_data)
    length = len(wr_data)
    print('Write {} bytes, showing the first 100 bytes:'.format(len(wr_data)))
    for i in range(0, 99):
        print("0x{:02X}".format(wr_data[i]) + ' ', end='')
    print('\n')  
    s_inf = time.time()
    opu_dma(wr_data, num, 0, 0, usb_dev, usb_ep_out, usb_ep_in)
    e_inf = time.time()
    print('Write Operation Processing Time: {}ms.\n'.format((e_inf - s_inf)*1000))

    # read data from FPGA
    rd_data = []
    s_inf = time.time()
    opu_dma(rd_data, num, 0, 2, usb_dev, usb_ep_out, usb_ep_in)
    e_inf = time.time()
    print('Read Operation Processing Time: {}ms.'.format((e_inf - s_inf)* 1000))
    print('Read {} bytes, showing the first 100 bytes:'.format(len(rd_data)))
    for i in range(0, 99):
        print("0x{:02X}".format(rd_data[i]) + ' ', end='')
    print('\n') 

    no_mismatch = True
    for i in range(0, length - 1):
        if (wr_data[i] != rd_data[i]):
            print ("Read and write data mismatch found on byte {}, check your USB driver \n".format(i))
            no_mismatch = False
            break
    if no_mismatch:
        print("All {} bytes write and read data match, USB loop test was successful\n".format(length))

def sniffer_mode():
    # realtime reading mode
    opu_run([], 0, 0, 4, usb_dev, usb_ep_out, usb_ep_in)

# definining keyboard controls
class Ctrl(Enum):
    (
        QUIT,
        CALIBRATE_FORCES,
        CALIBRATE_WINDSPEED,
        EXPERIMENT_START,
        INITIAL_ACTUATION
    ) = range(5)


QWERTY_CTRL_KEYS = {
    Ctrl.QUIT: Key.esc,
    Ctrl.CALIBRATE_FORCES: "f",
    Ctrl.CALIBRATE_WINDSPEED: "w",
    Ctrl.EXPERIMENT_START: "s",
    Ctrl.INITIAL_ACTUATION: "a"
}

AZERTY_CTRL_KEYS = QWERTY_CTRL_KEYS.copy()


## KEYBOARD CLASS
class KeyboardCtrl(Listener):
    def __init__(self, ctrl_keys=None):
        self._ctrl_keys = self._get_ctrl_keys(ctrl_keys)
        self._key_pressed = defaultdict(lambda: False)
        self._last_action_ts = defaultdict(lambda: 0.0)
        super(KeyboardCtrl,self).__init__(on_press=self._on_press, on_release=self._on_release)
        self.start()

    def _on_press(self, key):
        if isinstance(key, KeyCode):
            self._key_pressed[key.char] = True
        elif isinstance(key, Key):
            self._key_pressed[key] = True
        if self._key_pressed[self._ctrl_keys[Ctrl.QUIT]]:
            return False
        else:
            return True

    def _on_release(self, key):
        if isinstance(key, KeyCode):
            self._key_pressed[key.char] = False
        elif isinstance(key, Key):
            self._key_pressed[key] = False
        return True

    def quit(self):
        if self._key_pressed[self._ctrl_keys[Ctrl.QUIT]]:
            print("EXPERIMENT END")
            return not self.running or self._key_pressed[self._ctrl_keys[Ctrl.QUIT]]

    def cal_forces(self,forces):
        if self._key_pressed[self._ctrl_keys[Ctrl.CALIBRATE_FORCES]]:
            print("CALIBRATE FORCES")
            return forces
        else:
            return [0]

    def cal_speed(self,speed):
        if self._key_pressed[self._ctrl_keys[Ctrl.CALIBRATE_WINDSPEED]]:
            print("CALIBRATE SPEED")
            return speed
        else:
            return [0]

    def start_exp(self):
        if self._key_pressed[self._ctrl_keys[Ctrl.EXPERIMENT_START]]:
            #print("EXPERIMENT START")
            return 1
        else:
            return 0

    def init_act(self):
        if self._key_pressed[self._ctrl_keys[Ctrl.INITIAL_ACTUATION]]:
            #print("EXPERIMENT START")
            return 1
        else:
            return 0

    def _get_ctrl_keys(self, ctrl_keys):
        # Get the default ctrl keys based on the current keyboard layout:
        if ctrl_keys is None:
            ctrl_keys = QWERTY_CTRL_KEYS
            try:
                # Olympe currently only support Linux
                # and the following only works on *nix/X11...
                keyboard_variant = (
                    subprocess.check_output(
                        "setxkbmap -query | grep 'variant:'|"
                        "cut -d ':' -f2 | tr -d ' '",
                        shell=True,
                    )
                    .decode()
                    .strip()
                )
            except subprocess.CalledProcessError:
                pass
            else:
                if keyboard_variant == "azerty":
                    ctrl_keys = AZERTY_CTRL_KEYS
        return ctrl_keys




class MultiChannelAnalogInput(threading.Thread):
    """Class to create a multi-channel analog input
    
    Usage: AI = MultiChannelInput(physicalChannel)
        physicalChannel: a string or a list of strings
    optional parameter: limit: tuple or list of tuples, the AI limit values
                        reset: Boolean
    Methods:
        read(name), return the value of the input name
        readAll(), return a dictionary name:value
    """
    def __init__(self,physicalChannel, limit = None, reset = False):
        self.control = KeyboardCtrl()
        if type(physicalChannel) == type(""):
            self.physicalChannel = [physicalChannel]
        else:
            self.physicalChannel  =physicalChannel
        self.numberOfChannel = physicalChannel.__len__()
        if limit is None:
            self.limit = dict([(name, (-10.0,10.0)) for name in self.physicalChannel])
        elif type(limit) == tuple:
            self.limit = dict([(name, limit) for name in self.physicalChannel])
        else:
            self.limit = dict([(name, limit[i]) for  i,name in enumerate(self.physicalChannel)])           
        if reset:
            DAQmxResetDevice(physicalChannel[0].split('/')[0] )
        super(MultiChannelAnalogInput,self).__init__()
        super(MultiChannelAnalogInput,self).start()

    def configure(self):
        # Create one task handle per Channel
        taskHandles = dict([(name,TaskHandle(0)) for name in self.physicalChannel])
        for name in self.physicalChannel:
            DAQmxCreateTask("",byref(taskHandles[name]))
            DAQmxCreateAIVoltageChan(taskHandles[name],name,"",DAQmx_Val_Diff,
                                     self.limit[name][0],self.limit[name][1],
                                     DAQmx_Val_Volts,None)
        self.taskHandles = taskHandles
    def readAll(self):
        #return dict([(name,self.read(name)) for name in self.physicalChannel])
        return np.array([self.read(name) for name in self.physicalChannel]).reshape(8,1)

    def read(self,name = None):
        if name is None:
            name = self.physicalChannel[0]
        taskHandle = self.taskHandles[name]                    
        DAQmxStartTask(taskHandle)
        data = numpy.zeros((1,), dtype=numpy.float64)
#        data = AI_data_type()
        read = int32()
        DAQmxReadAnalogF64(taskHandle,1,10.0,DAQmx_Val_GroupByChannel,data,1,byref(read),None)
        DAQmxStopTask(taskHandle)
        return data[0]




def system_dynamics():
    multipleAI = MultiChannelAnalogInput(physicalChannel=["Dev1/ai0","Dev1/ai1","Dev1/ai2","Dev1/ai3","Dev1/ai4","Dev1/ai5","Dev1/ai6","Dev1/ai7"])
    multipleAI.configure()

    #jr3 = np.array([[5.001 ,0.026 ,-0.082 ,-0.511 ,-0.116 ,-0.067],[-0.037 ,5.007 ,0.010 ,0.054 ,-0.518 ,-0.101],[0.100 ,0 ,10.595 ,0.143,0.356 ,0.093],[-0.001 ,0, 0.002, 0.414 ,0.001 ,-0.01],[0.002, -0.001, 0.006 ,-0.001 ,0.412 ,0.002],[0, -0.003, 0.008 ,0.003 ,0, 0.411]])
    #amti = np.array([[0.38388 ,-0.00145 ,0.00357 ,0.00231 ,-0.00148 ,0.00046],[-0.0031 ,0.38123 ,-9.2e-4 ,0.00149 ,-0.00141 ,-0.00086],[0.00167 ,0.00623 ,1.58332 ,-0.0025,0.00343 ,0.00018],[-0.0001 ,2e-5, -1.1e-4, 0.00747 ,-3e-5 ,-2e-5],[1e-5, -1e-5, 1.4e-4 ,-7e-5 ,0.00753 ,3e-5],[4e-5, -8e-5, -5e-5 ,-3e-5 ,-3e-5, 0.01043]])
    fts1 = np.array([[0.0571 ,0.01423 ,-0.09391 ,2.77404 ,-0.01591,-2.78630],[0.04095,-3.22119,-0.01149,1.60517,-0.04573,1.58506],[4.56553,-0.18154,4.70567,-0.06025,4.55201,-0.12412],[0.07372,-0.69171,2.62774,0.30786,-2.57630,0.40346],[-3.02574,0.11408,1.50842,-0.60906,1.48526,0.55378],[0.00389,-1.55228,0.04948,-1.49459,0.02750,-1.51443]])
    fts2 = np.eye(6)
    fts2[0,0] = 4.44822
    fts2[1,1] = 4.44822
    fts2[2,2] = 4.44822
    fts2[3,3] = 0.11298
    fts2[4,4] = 0.11298
    fts2[5,5] = 0.11298
    data=np.empty((6,1))
    dcon_calf = 0
    density = 1.220
    dcon_calw = 0
    #ts = time.time()
    sampling_time = 40e-3
    #flag_s=0
    tii = 0
    data_all = np.zeros((7,1))
    d1 = np.zeros((7,1))

    moving_tw = 50 #Time window = moving_tw * sampling_time
    state_arr = np.zeros(moving_tw)

    state_min = 0.31
    state_max = 0.42
    scale_smax = 80 
    s_up = 0

    ii = 0


    global start_exp
    global state1
    global flag_s
    global flag_a
    global run_exp
    global quit_flag
    global speed
    global Lift
    global Drag

    #count = 0

    while run_exp:
        time.sleep(0.001)
        #count = count + 1
        #print(count)
        #print(start_exp)
        if (start_exp == 1):
            ts = time.time()
            #multipleAI.readAll()
            #time.sleep(1e-3)
            raw_d = multipleAI.readAll()
            #print(raw_d[6]-raw_d[8])
            dcon = np.matmul(fts1,raw_d[0:6,:].reshape(6,1))
            dconw = raw_d[6:8,:].reshape(2,1)
            #dcon = np.matmul(amti,raw_d[0:6,:].reshape(6,1))
            dcon2 = np.matmul(fts2,dcon)
            data = np.append(data,dcon2,axis=1)
            #print(dcon2)s_up
            if ctrl.quit():
                print('Quit!')
                quit_flag = 1
                break
            if ctrl.cal_forces(dcon2)[0]:
                dcon_calf = ctrl.cal_forces(dcon2)
            if ctrl.cal_speed(dconw)[0]:
                dcon_calw = ctrl.cal_speed(dconw)
            if  ctrl.start_exp() and flag_s==0:
                print("START EXPERIMENT")
                flag_s=1
            if  ctrl.init_act() and flag_a==0:
                print("INITIAL ACTUATION")
                flag_a=1
            sp_v = dconw - dcon_calw
            forces = dcon2 - dcon_calf
            #print(forces)
            v1 = sp_v[0,:] * 1.000 * 249.1
            v2 = sp_v[1,:] * 1.000 * 249.1 * 1.2
            if v1>=0:
                wv1 = np.sqrt(((v1/density)*2)/(1 - 24.0 *24/62/62))
            else:
                wv1 = 0
            #print(wv1.shape)
            #print(forces[:,0].shape)
            d1[0:6,:] = forces
            d1[6,:] = wv1
            if d1[0,:] != 0:
                state_r = ((((-d1[0,:]/d1[1,:])-state_min)/(state_max - state_min)) * scale_smax) + s_up
            else:
                state_r  = 0
            Lift = d1[1,:]
            Drag = -d1[0,:]
            speed = wv1
            if(state_r >= scale_smax + s_up):
                state_r = scale_smax + s_up
            elif(state_r <= 0):
                state_r = 0
            

            state_arr = np.append(state_arr, state_r)
            state1 = np.mean(state_arr)

            #print('current average =', state1)
            #print(state_arr)
            #print('readings used for average:', state_arr)

            if(len(state_arr) >= moving_tw):
                state_arr = np.delete(state_arr, 0)
            
            if(ii!=0 and (sampling_time - (time.time()-ts)) >= 0):
                time.sleep(sampling_time - (time.time()-ts))
            #print(time.time()-ts)
            '''if(time.time() - tii >= 1):
                tii = time.time()
                #print(Drag)
                print(d1[1,:]/d1[0,:])'''
            #print(forces)
            #if flag_s==1:
            #data_all = np.append(data_all,d1,axis=1)
            ii = 1


# Demo modes
ECHO    = 0
SNIFFER = 1

RST_ADDR = 0x00000005
if __name__ == '__main__':
    print('Start!')
    # find device
    usb_dev, usb_ep_in, usb_ep_out = find_device()
    usb_dev.set_configuration()
    
    # initial reset usb and fpga
    usb_dev.reset()
    ctrl = KeyboardCtrl()

    # Demo modes:   ECHO = echo(write to fpge and read the same bytes back), 
    #               SNIFFER = sniffer(read the fpga output continously)
    # when switching from SNIFFER to ECHO mode, power cycle of the fpga is needed
    #mode = SNIFFER
    #if mode == ECHO:
    #    echo_mode()
    #elif mode == SNIFFER:
    #    sniffer_mode()
    #echo_mode()

    dynamics = Thread(target=system_dynamics)
    dynamics.start()
    str_time = time.time()

    exp_length = 150

    save_data = []
    tpr_arr = []
    time_array = []
    Lift_array = []
    Drag_array = []

    num = 64 * 1
    '''
    np_data = np.random.randint(0, high=255, size = num, dtype=np.uint8)
    wr_data = list(np_data)
    length = len(wr_data)

    print(wr_data[0])
    print(wr_data[1])
    print(wr_data[2])

    '''
    #Vact = 50
    tdiff=1
    sfac=10
    tt=0
    ccd=0
    ttt=[]
    Vact_array = []
    state1_array = []
    speed_array = []
    ntr=0
    flagg=0
    tss = time.time()
    while True:
        #print(state1)
        #ts=time.time()
        if(quit_flag == 1):
            break
        if(flag_s == 0):
            np_data1 = np.array([0,flag_s,flag_a],dtype=np.uint8)
        else:
            np_data1 = np.array([state1,flag_s,flag_a],dtype=np.uint8)
        np_data2 = np.random.randint(0, high=255, size = num-3, dtype=np.uint8)
        np_data = np.concatenate((np_data1,np_data2))
        wr_data = list(np_data)
        length = len(wr_data)
        '''
        num = 64 * 1
        np_data = np.random.randint(0, high=255, size = num, dtype=np.uint8)
        wr_data = list(np_data)
        length = len(wr_data)
        '''
    
        # write data to ddr
        #s_inf = time.time()
        opu_dma(wr_data, num, 10, 0, usb_dev, usb_ep_out, usb_ep_in)
        #e_inf = time.time()
        #print('Write Operation Processing Time: {}ms.\n'.format((e_inf - s_inf)*1000))
    
        # start calculation
        #s_inf = time.time()
        opu_run([], 0, 0, 3, usb_dev, usb_ep_out, usb_ep_in)
        #e_inf = time.time()
        #print('Calculation Processing Time: {}ms.\n'.format((e_inf - s_inf)*1000))

        # read data from FPGA
        rd_data = []
        #s_inf = time.time()
        opu_dma(rd_data, num, 11, 2, usb_dev, usb_ep_out, usb_ep_in)
        #e_inf = time.time()
        #print('Read Operation Processing Time: {}ms.'.format((e_inf - s_inf)* 1000))
        #print(rd_data[1])
        '''action3 = rd_data[0]
        action2 = rd_data[1]
        action1 = rd_data[2]'''
        start_exp = rd_data[6]
        #Vact=50
        '''if(start_exp==1 and flagg==0):
            flagg=1
            Vact = 50
        elif(start_exp==1 and flagg==1 and ccd==5000):
            Vact = np.array(rd_data[0]) + 256 * np.array(rd_data[1]) + 256 ** 2 * np.array(rd_data[2]) + 256 ** 3 * np.array(rd_data[3])
            ccd=0'''
        if(np.array(rd_data[0])!=0 or np.array(rd_data[1])!=0 or np.array(rd_data[2])!=0 or np.array(rd_data[3])!=0):
            Vact = np.array(rd_data[0]) + 256 * np.array(rd_data[1]) + 256 ** 2 * np.array(rd_data[2]) + 256 ** 3 * np.array(rd_data[3])
            Vact = (Vact*20/65536)-10
            #print(Vact)
        #else:
         #   Vact=50

        #print(rd_data[0])
        save_data.append(rd_data)
        ttt.append(tt)
        if(flagg==1):
            ccd=ccd+1

        end_time = time.time()

        time_array.append((end_time-str_time))
        Vact_array.append(Vact)
        state1_array.append(state1)
        speed_array.append(speed)
        Lift_array.append(np.array(Lift))
        Drag_array.append(np.array(Drag))

        if ((end_time-str_time)>exp_length):
            break

        #print(time.time()-ts)
        #print('Total Processing Time: {}ms.'.format((e_inf - s_inf)* 1000))

        # check the correctness
        '''
        mismatch = False
        for i in range(64):
          if rd_data[i] != wr_data[i] + 1:
             print('ERROR: DATA NOT MATCH!!!')
             print('ERROR Index: {}!!!'.format(i))
             mismatch = True
             break
        if mismatch is False:
            print("PASSED!")
        else:
            print("FAILED!")
            break

        '''
        # wait for 1ms and do it again
        #time.sleep(0.001)

    #tpr_np = np.array(tpr_arr)
    #high_lpt = (tpr_np>1.2).sum()

    run_exp = False
    dynamics.join()

    save_data_np = np.array(save_data)

    lenth_arr = save_data_np.shape[0]
    state_input1_array = []
    current_out1_array=[]
    current_out2_array=[]
    firing_rate_1_array=[]
    firing_rate_2_array=[]
    state_input2_array = []
    state_input3_array = []
    F_array = []

    

    clk100_array = []
    strexp_array = []
    rst_array = []
    Vmax = 1.5
    Vmin = -1.5
    act1_array = []
    act2_array = []
    act3_array = []
    curr_temp1=[]
    curr_temp2=[]
    fr1_temp=[]
    fr2_temp=[]
    op_1_ar=[]
    op_2_ar=[]
    ch_num_ar = []
    fb_sign_ar = []
    data_ch10_ar = []
    data_ch5_ar = []
    inp1_ar = []
    fb1_ar = []
    fb2_ar = []
    cc=0
    i1=0
    i2=0
    ff1=0
    ff2=0
    for i in range(lenth_arr): 
        if (save_data_np[i,-1] < 128):
            state_input1 = float(save_data_np[i,-1])
        else:
            state_input1 = float(save_data_np[i,-1])
            state_input1 = state_input1 - 256

        '''if (save_data_np[i,-2] < 128):
            state_input2 = float(save_data_np[i,-2])
        else:
            state_input2 = float(save_data_np[i,-2])
            state_input2 = state_input2 - 256

        if (save_data_np[i,-3] < 128):
            state_input3 = float(save_data_np[i,-3])
        else:
            state_input3 = float(save_data_np[i,-3])
            state_input3 = state_input3 - 256'''
        current_out1 = float(save_data_np[i,-3]) + 256 * float(save_data_np[i,-2])
        current_out2 = float(save_data_np[i,11]) + 256 * float(save_data_np[i,12])
        fr1 = float(save_data_np[i,7]) + 256 * float(save_data_np[i,8])
        fr2 = float(save_data_np[i,9]) + 256 * float(save_data_np[i,10])
        dch10 = float(save_data_np[i,20]) + 256 * float(save_data_np[i,21])
        dch5 = float(save_data_np[i,22]) + 256 * float(save_data_np[i,23])
        ch_num = float(save_data_np[i,14])
        fb_sign = float(save_data_np[i,13])
        ch_out_1_2 = float(save_data_np[i,-7]) 
        ch_in_1_2 = float(save_data_np[i,-4])
        ch_out_1_2_bin = "{0:08b}".format(int(ch_out_1_2)) 
        ch_in_1_2_bin = "{0:08b}".format(int(ch_in_1_2)) 
        ch_out_1 = int(ch_out_1_2_bin[0:3],2)
        ch_out_2 = int(ch_out_1_2_bin[3:6],2)
        ch_in_1 = int(ch_in_1_2_bin[0:2],2)
        #print(ch_out_1_2_bin)
        #print(ch_in_1)      

        Ft = 0.5*((state_input1**2)) #+ (state_input2**2) + (state_input3**2))


        state_input1_array.append(state_input1)
        current_out1_array.append(current_out1)
        current_out2_array.append(current_out2)
        firing_rate_1_array.append(fr1)
        firing_rate_2_array.append(fr2)
        data_ch10_ar.append(dch10)
        data_ch5_ar.append(dch5)
        ch_num_ar.append(ch_num)
        fb_sign_ar.append(fb_sign)
        
        if(ch_out_1 == 2):
            fb1_ar.append(0)
        elif(ch_out_1 == 5):
            fb1_ar.append(Vmax)
        elif(ch_out_1 == 6):
            fb1_ar.append(Vmin)
        else:
            fb1_ar.append(0)

        if(ch_out_2 == 2):
            fb2_ar.append(0)
        elif(ch_out_2 == 5):
            fb2_ar.append(Vmax)
        elif(ch_out_2 == 6):
            fb2_ar.append(Vmin)
        else:
            fb2_ar.append(0)

        if(ch_in_1 == 0):
            inp1_ar.append(0)
        elif(ch_in_1 == 2):
            inp1_ar.append(Vmax)
        elif(ch_in_1 == 3):
            inp1_ar.append(Vmin)
        else:
            inp1_ar.append(0)

        F_array.append(Ft)
        strexp_array.append(float(save_data_np[i,6]))
        rst_array.append(float(save_data_np[i,5]))
        clk100_array.append(float(save_data_np[i,4]))

        '''act1_array.append(act1)
        act2_array.append(act2)
        act3_array.append(act3)'''
        if (i<lenth_arr-1 and ttt[i+1]-ttt[i]==0):
            i1=i1+current_out1
            i2=i2+current_out2
            ff1=ff1+fr1
            ff2=ff2+fr2
            cc=cc+1
        elif (i==lenth_arr-1 or ttt[i+1]-ttt[i]!=0):
            #if(cc==0):
                #cc=1
            curr_temp1.append(i1/cc)
            curr_temp2.append(i2/cc)
            fr1_temp.append(ff1/cc)
            fr2_temp.append(ff2/cc)
            cc=0
            ff1=0
            ff2=0
            i1=0
            i2=0
    
    
    if(cond == 1 and save_dat == 1):
        with open(dir1 + "_Data_SNIC.pkl", 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([Lift_array,Drag_array,state1_array,time_array,speed_array,data_ch5_ar,data_ch10_ar,firing_rate_2_array,firing_rate_1_array,Vact_array,rst_array,strexp_array,clk100_array,current_out2_array,current_out1_array,state_input1_array,fb1_ar,fb2_ar], f)
    elif(cond == 2 and save_dat == 1):
        with open(dir2 + "_Data_SNIC.pkl", 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([Lift_array,Drag_array,state1_array,time_array,speed_array,data_ch5_ar,data_ch10_ar,firing_rate_2_array,firing_rate_1_array,Vact_array,rst_array,strexp_array,clk100_array,current_out2_array,current_out1_array,state_input1_array,fb1_ar,fb2_ar], f)
    
    if(cond == 1 and save_dat == 1):
        with open(dir1 + "_Data_SNIC_all.pkl", 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(save_data_np, f)
    elif(cond == 2 and save_dat == 1):
        with open(dir2 + "_Data_SNIC_all.pkl", 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(save_data_np, f)

    
    '''plt.rcParams.update({'font.size': 12})
    #t = np.arange(0,plot_time*0.00016,0.00016)
    fig, axt = plt.subplots(2,1,sharex=True)
    fig.subplots_adjust(hspace=0.2)

    axs = plt.subplot(211)
    axs.plot(curr_temp1,fr1_temp, 'tab:blue')
#        axs[0].set_yticks(np.arange(0, 5000, 500))
    #axs.set_xlim(0,110)
    #axs.set_ylim(-100, 100)
    axs.grid(True)
    axs.set(ylabel='Firing Rate_1')
    axs.set(xlabel='Current')

    axs = plt.subplot(212)
    axs.plot(curr_temp1,fr2_temp, 'tab:blue')
#        axs[0].set_yticks(np.arange(0, 5000, 500))
    #axs.set_xlim(0,110)
    #axs.set_ylim(-100, 100)
    axs.grid(True)
    axs.set(ylabel='Firing Rate_2')
    axs.set(xlabel='Current')
    

    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_I_vs_FR.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_I_vs_FR.png",dpi=600)'''

    plt.rcParams.update({'font.size': 12})
    #t = np.arange(0,plot_time*0.00016,0.00016)
    fig, axt = plt.subplots(3,1,sharex=True)
    fig.subplots_adjust(hspace=0.2)
    # axs[5].plot(t, current_array, 'tab:orange')
    # axs[5].set_yticks(np.arange(0, 1800, 200))
    # axs[5].set_ylim(-100, 2000)
    # axs[5].set(ylabel='Output current (nA)')
    
    
    axs = plt.subplot(311)
    axs.plot(time_array,state_input1_array, 'tab:blue')
#        axs[0].set_yticks(np.arange(0, 5000, 500))
    axs.set_xlim(0,exp_length+10)
    axs.set_ylim(0, 80)
    axs.grid(True)
    axs.set(ylabel='s1')
    
    '''axs = plt.subplot(412)
    axs.plot(time_array,state_input2_array, 'tab:blue')
#        axs[0].set_yticks(np.arange(0, 5000, 500))
    axs.set_xlim(0,110)
    axs.set_ylim(-100, 100)
    axs.grid(True)
    axs.set(ylabel='s2')

    axs = plt.subplot(413)
    axs.plot(time_array,state_input3_array, 'tab:blue')
#        axs[0].set_yticks(np.arange(0, 5000, 500))
    axs.set_xlim(0,110)
    axs.set_ylim(-100, 100)
    axs.grid(True)
    axs.set(ylabel='s3')'''

    ''' axs = plt.subplot(312)
    axs.plot(time_array,F_array, 'tab:red')
#        axs[0].set_yticks(np.arange(0, 5000, 500))
    axs.set_xlim(0,110)
    #axs.set_ylim(-100, 100)
    axs.grid(True)
    axs.set(xlabel='time (s)', ylabel='F')'''

    axs = plt.subplot(312)
    axs.plot(time_array,current_out1_array, 'tab:red')
    axs.set_xlim(0,exp_length+10)
    axs.grid(True)
    axs.set(xlabel='time (s)', ylabel='Current_1')
    
    axs = plt.subplot(313)
    axs.plot(time_array,current_out2_array, 'tab:red')
    axs.set_xlim(0,exp_length+10)
    axs.grid(True)
    axs.set(xlabel='time (s)', ylabel='Current_2')
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    #plt.show()
    
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_S_I_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_S_I_vs_t.png",dpi=600)

    '''plt.rcParams.update({'font.size': 12})
    #t = np.arange(0,plot_time*0.00016,0.00016)
    fig, axt = plt.subplots(3,1,sharex=True)
    fig.subplots_adjust(hspace=0.2)
    # axs[5].plot(t, current_array, 'tab:orange')
    # axs[5].set_yticks(np.arange(0, 1800, 200))
    # axs[5].set_ylim(-100, 2000)
    # axs[5].set(ylabel='Output current (nA)')
    
    
    axs = plt.subplot(311)
    axs.plot(time_array,clk100_array, 'tab:blue')
#        axs[0].set_yticks(np.arange(0, 5000, 500))
    axs.set_xlim(0,exp_length+10)
    axs.set_ylim(-0.2, 1.2)
    axs.grid(True)
    axs.set(ylabel='clk')
    
    axs = plt.subplot(312)
    axs.plot(time_array,strexp_array, 'tab:blue')
#        axs[0].set_yticks(np.arange(0, 5000, 500))
    axs.set_xlim(0,exp_length+10)
    axs.set_ylim(-0.2, 1.2)
    axs.grid(True)
    axs.set(ylabel='str')


    axs = plt.subplot(313)
    axs.plot(time_array,rst_array, 'tab:red')
#        axs[0].set_yticks(np.arange(0, 5000, 500))
    axs.set_xlim(0,exp_length+10)
    axs.set_ylim(-0.2, 1.2)
    axs.grid(True)
    axs.set(xlabel='time (s)', ylabel='rst')
    
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_clkstrrst.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_clkstrrst.png",dpi=600)'''

    plt.rcParams.update({'font.size': 12})
    #t = np.arange(0,plot_time*0.00016,0.00016)
    fig, axt = plt.subplots(1,1,sharex=True)
    fig.subplots_adjust(hspace=0.2)
    # axs[5].plot(t, current_array, 'tab:orange')
    # axs[5].set_yticks(np.arange(0, 1800, 200))
    # axs[5].set_ylim(-100, 2000)
    # axs[5].set(ylabel='Output current (nA)')
    
    
    axs = plt.subplot(111)
    axs.plot(time_array,Vact_array, 'tab:blue')
#        axs[0].set_yticks(np.arange(0, 5000, 500))
    axs.set_xlim(0,exp_length+10)
    axs.set_ylim(0, 5)
    axs.grid(True)
    axs.set(ylabel='Actuation Voltage (V)')

    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_Vact_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_Vact_vs_t.png",dpi=600)

    plt.figure()
    fig1 = plt.plot(time_array,firing_rate_1_array, 'tab:blue')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Firing rate 1')
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_FR1_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_FR1_vs_t.png",dpi=600)

    plt.figure()
    fig2 = plt.plot(time_array,firing_rate_2_array, 'tab:blue')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Firing rate 2')
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_FR2_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_FR2_vs_t.png",dpi=600)

    '''plt.figure()
    fig1 = plt.plot(time_array,data_ch10_ar, 'tab:blue')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Data_ch_10')
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_datach10_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_datach10_vs_t.png",dpi=600)

    plt.figure()
    fig2 = plt.plot(time_array,data_ch5_ar, 'tab:blue')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Data_ch_5')
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_datach5_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_datach5_vs_t.png",dpi=600)'''
    
    plt.figure()
    fig1 = plt.plot(time_array,speed_array, 'tab:blue')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_Wind_Speed_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_Wind_Speed_vs_t.png",dpi=600)

    plt.figure()
    fig1 = plt.plot(time_array,fb1_ar, 'tab:blue')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('FB1')
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_FB1_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_FB1_vs_t.png",dpi=600)

    plt.figure()
    fig1 = plt.plot(time_array,fb2_ar, 'tab:blue')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('FB2')
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_FB2_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_FB2_vs_t.png",dpi=600)

    plt.figure()
    fig1 = plt.plot(time_array,inp1_ar, 'tab:blue')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('X1')
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_x1_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_x1_vs_t.png",dpi=600)

    '''plt.figure()
    fig1 = plt.plot(time_array,Lift_array, 'tab:blue')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Lift (N)')
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_Lift_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_Lift_vs_t.png",dpi=600)

    plt.figure()
    fig1 = plt.plot(time_array,Drag_array, 'tab:blue')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Drag (N)')
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_Drag_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_Drag_vs_t.png",dpi=600)
    
    plt.figure()
    fig1 = plt.plot(np.array(time_array),np.array(Drag_array)/np.array(Lift_array), 'tab:blue')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('D/L')
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_DL_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_DL_vs_t.png",dpi=600)'''
    
    '''axs = plt.subplot(211)
    axs.plot(time_array,fb1_ar, 'tab:blue')
#        axs[0].set_yticks(np.arange(0, 5000, 500))
    axs.set_ylim(-2, 2)
    axs.grid(True)
    axs.set(ylabel='clk')
    axs.set(xlabel='time (s)', ylabel='FB1')
    
    axs = plt.subplot(212)
    axs.plot(time_array,fb2_ar, 'tab:blue')
#        axs[0].set_yticks(np.arange(0, 5000, 500))
    axs.set_ylim(-2, 2)
    axs.grid(True)
    axs.set(ylabel='str')
    axs.set(xlabel='time (s)', ylabel='FB2')
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_fb_data.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_fb_data.png",dpi=600)'''

    plt.show()