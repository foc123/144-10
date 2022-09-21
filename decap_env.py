"""
A new ckt environment based on a new structure of MDP
"""
import gym
from gym import spaces

import numpy as np
import random
import statistics
import os
import IPython
import itertools
import pickle
import os


NCOL, NROW, NCAP = 12, 12, 10
CAP_VAL = 1e-9
VDI_LOW = -1000.00
VDI_HIGH = 1.0
VDI_TARGET = 0.9e-10
NODE_LOC = [416, 1250, 2083, 2916, 3750, 4583, 5416, 6250, 7083, 7916, 8750, 9583]
PATH_PREFIX = '/simulation/leilai/Github/decap_rl/'

def run_os():
    # first recreate pid+chiplet1_tr_new.subckt
    filename = 'chiplet1_tr_new.subckt'
    with open(PATH_PREFIX+filename,'r') as f:
        chiplet = f.read()
        chiplet = chiplet.replace(".include 'vdd_decap.1'",".include '"+str(os.getpid())+"_vdd_decap.1'")
        f.close()
    with open(PATH_PREFIX+str(os.getpid())+'_'+filename,'w') as f:
        f.write(chiplet)
        f.close()
    # recreate 
    filename = 'interposer1_tr_new.sp'
    with open(PATH_PREFIX+filename,'r') as f:
        chiplet = f.read()
        chiplet = chiplet.replace(".include 'chiplet1_tr_new.subckt'",".include '"+str(os.getpid())+"_chiplet1_tr_new.subckt'")
        f.close()
    with open(PATH_PREFIX+str(os.getpid())+'_'+filename,'w') as f:
        f.write(chiplet)
        f.close()
    # add pid to the linux command
    os.system('ngspice -b '+str(os.getpid())+'_interposer1_tr_new.sp -r '+str(os.getpid())+'_interposer1_tr.raw')
    os.system('bin/inttrvmap int1.conf '+str(os.getpid())+'_interposer1_tr.raw 1.0 0.05 '+str(os.getpid()))

def readvdi(file):  # 读取csv中的vdi的数据，得到的是array数据，这里作为input
    zvdi = readresult(file)
    z = zvdi[:, 2]
    return z

def readresult(filename):
    a1 = np.genfromtxt(filename)
    return a1

class DecapPlaceParallel(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config):
        self.env_steps = 0
	#stay, left, up, right, down = 0, 1, 2, 3, 4
        self.action_meaning = [0, -1, -NCOL, 1, NCOL] 
        self.action_space = spaces.Tuple([spaces.Discrete(len(self.action_meaning))]*NCAP)
        self.observation_space = spaces.Box(
            low=np.array([VDI_LOW]*(NCOL*NROW+1)+NCAP*[0]),
            high=np.array([VDI_HIGH]*(NCOL*NROW+1)+NCAP*[NCOL*NROW-1]),dtype=np.float64)

        #initialize current param/spec observations
        self.cur_params_idx = np.zeros(NCAP, dtype=np.int32)

        #Get the g* (overall design spec) you want to reach
        self.global_g = VDI_TARGET


    def reset(self):

        #initialize current parameters
        self.cur_params_idx = np.array([NCOL*NROW/2-1]*NCAP)
        self.cur_specs = self.update(self.cur_params_idx)
        cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
        reward = self.reward(self.cur_specs, self.global_g)

        #observation is a combination of current specs distance from ideal, ideal spec, and current param vals
        self.ob = np.concatenate([cur_spec_norm, [self.global_g], self.cur_params_idx])
        return self.ob
 
    def step(self, action):
        """
        :param action: is vector with elements between 0 and 1 mapped to the index of the corresponding parameter
        :return:
        """

        #Take action that RL agent returns to change current params
        action = list(np.reshape(np.array(action),(np.array(action).shape[0],)))
        self.cur_params_idx = self.cur_params_idx + np.array([self.action_meaning[a] for a in action])

        self.cur_params_idx = np.clip(self.cur_params_idx, [0]*NCAP, [NCOL*NROW-1]*NCAP)
        #Get current specs and normalize
        self.cur_specs = self.update(self.cur_params_idx)
        cur_spec_norm  = self.lookup(self.cur_specs, self.global_g)
        reward = self.reward(self.cur_specs, self.global_g)
        done = False

        #incentivize reaching goal state
        if (reward >= 10):
            done = True
            print('-'*10)
            print('params = ', self.cur_params_idx)
            print('specs:', self.cur_specs)
            print('ideal specs:', self.specs_ideal)
            print('re:', reward)
            print('-'*10)

        #self.ob = np.concatenate([cur_spec_norm, [self.global_g], self.cur_params_idx])
        self.ob = np.concatenate([cur_spec_norm, [1.0], self.cur_params_idx])
        self.env_steps = self.env_steps + 1

        #print('cur ob:' + str(self.cur_specs))
        #print('ideal spec:' + str(self.specs_ideal))
        #print(reward)
        return self.ob, reward, done, {}

    def lookup(self, spec, goal_spec):
	# normalized by global_vdi/grid_size
        grid_size = len(spec)
        norm_factor = float(goal_spec)/grid_size
        norm_spec = [(norm_factor-s)/norm_factor for s in spec]
        return norm_spec
    
    def reward(self, spec, goal_spec):
        '''
        Reward: doesn't penalize for overshooting spec, is negative
        '''
        real_specs = self.lookup(spec, goal_spec)
        pos_val = [] 
        reward = 0.0
        for i,rel_spec in enumerate(real_specs):
            if rel_spec < 0:
                reward += rel_spec
                pos_val.append(0)
            else:
                pos_val.append(1)

        return reward if reward < -0.02 else 10

    def update(self, params_idx):
        """

        :param action: an int between 0 ... n-1
        :return:
        """
        str_dc = ''
        for i in range(NCAP):
            str_dc += 'c_decap_%d_%d nd_1_0_%d_%d 0 %e\n' % (i, i, NODE_LOC[int(params_idx[i]//NCOL)],
                                                              NODE_LOC[int(params_idx[i]%NCOL)], CAP_VAL)
        print(str_dc)
        f = open(PATH_PREFIX+str(os.getpid())+'_vdd_decap.1', 'w')
        f.write(str_dc)
        f.close()
        run_os()
        # add pid info
        state_ = readvdi(PATH_PREFIX+str(os.getpid())+'_chiplet1_vdd_1_vdi.csv')

        return state_



def main():
  env_config = {}
  env = DecapPlaceParallel(env_config)
  env.reset()
  env.step([2,2,2,2,2,2,2,2,2,2])

  IPython.embed()

if __name__ == "__main__":
  print(os.getpid())
  main()
