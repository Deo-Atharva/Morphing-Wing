from pynput import keyboard 
import time
import usb
import numpy as np

from base_method import * 
import matplotlib.pyplot as plt

from threading import Thread
from nptdms import TdmsFile
from scipy.interpolate import interp1d

# for keyboard
from pynput.keyboard import Listener, Key, KeyCode
from collections import defaultdict
from enum import Enum

action1 = 0
action2 = 0
action3 = 0
start_exp = 0
#state1 = 100.0
state2 = 30.0
state3 = 30.0
Vact = 0

gain_sys = 0.0

run_exp = True


dir_main = "/home/atharva/Morphing Wing_Python Code/Python Data/"
sub_dir = "271021/"
ver1 = "w1_w2"
ver2 = "w2_w1"
ver = "_v1"
dir = dir_main + sub_dir
dir1 = dir + "Human_" + ver1 + ver 
dir2 = dir + "Human_" + ver2 + ver
save_dat = 0
cond = 2

'''tdms_file = TdmsFile.read("/home/atharva/Morphing Wing_Python Code/Python Data/090621_LAR/test_v_15_aoa_n20_1.tdms")
group = tdms_file["Untitled"]
for i in range(3,6):
    chname = "Untitled " + str(i)
    channel = group[chname]
    if(i==3):
        Vact = np.array(channel[:])
    elif(i==4):
        Lift = np.array(channel[:])
    elif(i==5):
        Drag = np.array(channel[:])
        DLRatio = np.true_divide(Drag,Lift)
    

V0 =[(Vact>=-0.05) & (Vact<0.05)]
V05 = [(Vact>=0.45)  & (Vact<0.55)]
V1 = [(Vact>=0.95)  & (Vact<1.05)]
V15 = [(Vact>=1.45)  & (Vact<1.55)]
V2 = [(Vact>=1.95)  & (Vact<2.05)]
V25 = [(Vact>=2.45)  & (Vact<2.55)]
V3 = [(Vact>=2.95)  & (Vact<3.05)]
V35 = [(Vact>=3.45)  & (Vact<3.55)]
V4 = [(Vact>=3.95)  & (Vact<4.05)]
V45 = [(Vact>=4.45)  & (Vact<4.55)]
V5 = [(Vact>=4.95)  & (Vact<5.05)]

VList = [V0,V05,V1,V15,V2,V25,V3,V35,V4,V45,V5]
LDRList = []
for i in range(len(VList)):
    LDRList.append(np.mean(DLRatio[VList[i]]))

DLR = np.array(LDRList)


VL = np.array([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
f = interp1d(VL, DLR,kind='cubic')
state1 = f(0)
data_sim = []
for i in np.arange(0,5.1,0.1):
    data_sim.append(((f(i)-0.41)/0.093)*80)
#print(np.min(np.array(data_sim)))
plt.plot(np.arange(0,5.1,0.1),data_sim)
plt.grid(True)
plt.xlabel('Voltage (V)')
plt.ylabel('State')
plt.title("State vs Voltage")
plt.savefig(dir + "S_vs_V.png")'''
#ts = time.time()
#print(f(0.3))pip install scipy
#print(time.time()-ts)

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

class Ctrl(Enum):
    (
        QUIT,
        CALIBRATE_FORCES,
        CALIBRATE_WINDSPEED,
        EXPERIMENT_START
    ) = range(4)


QWERTY_CTRL_KEYS = {
    Ctrl.QUIT: Key.esc,
    Ctrl.CALIBRATE_FORCES: "f",
    Ctrl.CALIBRATE_WINDSPEED: "w",
    Ctrl.EXPERIMENT_START: "s",
    Ctrl.TURN_LEFT: Key.left,
    Ctrl.TURN_RIGHT: Key.right
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

    def move_left(self):
        if self._key_pressed[self._ctrl_keys[Ctrl.TURN_LEFT]]:
            #print("CALIBRATE SPEED")
            return 1
        else:
            return 0
    
    def move_right(self):
        if self._key_pressed[self._ctrl_keys[Ctrl.TURN_RIGHT]]:
            #print("CALIBRATE SPEED")
            return 1
        else:
            return 0

    def start_exp(self):
        if self._key_pressed[self._ctrl_keys[Ctrl.EXPERIMENT_START]]:
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
    sampling_time = 20e-3
    flag_s=0
    data_all = np.zeros((7,1))
    d1 = np.zeros((7,1))

    global start_exp
    global state1
    global state2
    global state3
    global action1
    global action2
    global action3
    global run_exp

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
            #print(dcon2)
            if ctrl.quit():
                break
            if ctrl.cal_forces(dcon2)[0]:
                dcon_calf = ctrl.cal_forces(dcon2)
            if ctrl.cal_speed(dconw)[0]:
                dcon_calw = ctrl.cal_speed(dconw)
            if  ctrl.start_exp() and flag_s==0:
                print("START EXPERIMENT")
                flag_s=1
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
            state1 = d1[0,:]
            state2 = d1[1,:]
            state3 = d1[2,:]
            if state1 < 0:
                state1 = 0
            #print(forces)
            #if flag_s==1:
            #data_all = np.append(data_all,d1,axis=1)



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
    

    exp_length = 45

    save_data = []
    tpr_arr = []
    time_array = []
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
    tdiff=10
    sfac=10
    tt=0
    ccd=0
    ttt=[]
    Vact_array = []
    ntr=0
    flagg=0
    
    while True:
        #print(state1)
        '''if (tt==0 or time.time() - ntr>=tdiff):
            tt=tt+20
            ntr=time.time()

        state1=sfac+tt'''
        with keyboard.Listener(on_press = on_key_press) as press_listener: #setting code for listening key-press
            press_listener.join()

        t = time.time() #reading time in sec

        with keyboard.Listener(on_release = on_key_release) as release_listener: #setting code for listening key-release
            release_listener.join()
        
        np_data1 = np.array([state1,0,0],dtype=np.uint8)
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
        #print(Vact)
        save_data.append(rd_data)
        ttt.append(tt)
        if(flagg==1):
            ccd=ccd+1

        end_time = time.time()

        time_array.append((end_time-str_time))
        Vact_array.append(Vact)

        if ((end_time-str_time)>exp_length):
            break

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

    act1_array = []
    act2_array = []
    act3_array = []
    curr_temp1=[]
    curr_temp2=[]
    fr1_temp=[]
    fr2_temp=[]
    data_ch11_ar = []
    data_ch6_ar = []
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
        dch11 = float(save_data_np[i,20]) + 256 * float(save_data_np[i,21])
        dch6 = float(save_data_np[i,22]) + 256 * float(save_data_np[i,23])
        #Vact = float(save_data_np[i,0]) + 256 * float(save_data_np[i,1]) + 256 ** 2 * float(save_data_np[i,2])+ 256 ** 3 * float(save_data_np[i,3]) 
        #print(Vact)
        #Vact = Vact + dVact

        Ft = 0.5*((state_input1**2)) #+ (state_input2**2) + (state_input3**2))

        '''if (save_data_np[i,2] == 1):
            act1 = 1
        elif (save_data_np[i,2] == 3):
            act1 = -1
        else:
            act1 = 0

        if (save_data_np[i,1] == 1):
            act2 = 1
        elif (save_data_np[i,1] == 3):
            act2 = -1
        else:
            act2 = 0

        if (save_data_np[i,0] == 1):
            act3 = 1
        elif (save_data_np[i,0] == 3):
            act3 = -1
        else:
            act3 = 0'''

        state_input1_array.append(state_input1)
        current_out1_array.append(current_out1)
        current_out2_array.append(current_out2)
        firing_rate_1_array.append(fr1)
        firing_rate_2_array.append(fr2)
        data_ch11_ar.append(dch11)
        data_ch6_ar.append(dch6)
        #Vact_array.append(Vact)

        
        #state_input2_array.append(state_input2)
        #state_input3_array.append(state_input3)
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
    
    plt.rcParams.update({'font.size': 12})
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
        plt.savefig(dir2 + "_I_vs_FR.png",dpi=600)

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
    axs.set_xlim(0,50)
    axs.set_ylim(0, 100)
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
    axs.set_xlim(0,50)
    axs.grid(True)
    axs.set(xlabel='time (s)', ylabel='Current_1')
    
    axs = plt.subplot(313)
    axs.plot(time_array,current_out2_array, 'tab:red')
    axs.set_xlim(0,50)
    axs.grid(True)
    axs.set(xlabel='time (s)', ylabel='Current_2')
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    #plt.show()
    
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_S_I_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_S_I_vs_t.png",dpi=600)

    plt.rcParams.update({'font.size': 12})
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
    axs.set_xlim(0,110)
    axs.set_ylim(-0.2, 1.2)
    axs.grid(True)
    axs.set(ylabel='clk')
    
    axs = plt.subplot(312)
    axs.plot(time_array,strexp_array, 'tab:blue')
#        axs[0].set_yticks(np.arange(0, 5000, 500))
    axs.set_xlim(0,110)
    axs.set_ylim(-0.2, 1.2)
    axs.grid(True)
    axs.set(ylabel='str')


    axs = plt.subplot(313)
    axs.plot(time_array,rst_array, 'tab:red')
#        axs[0].set_yticks(np.arange(0, 5000, 500))
    axs.set_xlim(0,110)
    axs.set_ylim(-0.2, 1.2)
    axs.grid(True)
    axs.set(xlabel='time (s)', ylabel='rst')
    
    
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_clkstrrst.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_clkstrrst.png",dpi=600)

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
    axs.set_xlim(0,50)
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

    plt.figure()
    fig1 = plt.plot(time_array,data_ch11_ar, 'tab:blue')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Data_ch_11')
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_datach11_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_datach11_vs_t.png",dpi=600)

    plt.figure()
    fig2 = plt.plot(time_array,data_ch6_ar, 'tab:blue')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Data_ch_6')
    if(cond == 1 and save_dat == 1):
        plt.savefig(dir1 + "_datach6_vs_t.png",dpi=600)
    elif(cond == 2 and save_dat == 1):
        plt.savefig(dir2 + "_datach6_vs_t.png",dpi=600)

    
    
    plt.show()