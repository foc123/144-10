import random, os, numpy as np, torch.nn as nn, torch, torch.nn.functional as F
import matplotlib.pyplot as plt, gym
from torch.distributions import Categorical
from copy import deepcopy


stay, left, up, right, down = 0, 1, 2, 3, 4
nrow = 12
ncol = 12
nstate = nrow * ncol
lr = 0.001
node_loc = []
loc_num = [416, 1250, 2083, 2916, 3750, 4583, 5416, 6250, 7083, 7916, 8750, 9583]
action_space = []
gamma = 0.9
fig_num = 0
loss_val = []

# debug
memorysize = 2000
itertimes = 20
batch_size = 32


#training
#memorysize = 3000
#itertimes = 25
#batch_size = 24
#


# 节点坐标数据集 [0,143]对应位置
for i in range(nrow):
    for j in range(ncol):
        node_loc.append([loc_num[i], loc_num[j]])


def readresult(filename):
    a1 = np.genfromtxt(filename)
    return a1


def readvdi(file):  # 读取csv中的vdi的数据，得到的是array数据，这里作为input
    zvdi = readresult(file)
    z = zvdi[:, 2]
    return z


def savefig( capidx,fn,vdi):
    fn += 1
    res_array = readresult('chiplet1_vdd_1_vdi.csv')
    x = res_array[:, 0]
    y = res_array[:, 1]
    z = res_array[:, 2]
    fig, (ax2) = plt.subplots(nrows=1)
    ax2.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
    cntr2 = ax2.tricontourf(x, y, z, levels=14, cmap="RdBu_r")

    fig.colorbar(cntr2, ax=ax2)
    ax2.plot(x, y, 'ko', ms=3)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    plt.subplots_adjust(hspace=0.5)
    plt.title('VDI={},cap={}'.format(vdi,capidx))
    plt.savefig('./result_fig/VDI_distri_%d' % fig_num)
    plt.close()
    return fn


def location(file):  # 节点位置
    a = readresult(file)
    x = a[:, 0]
    y = a[:, 1]
    loc = []
    for r in range(len(a)):
        loc.append([x[r], y[r]])
    return loc

# 将vdi和decap位置归一化
def norm(vdi,cap_idx):
    maxv = max(vdi)
    minv = min(vdi)
    norm_v = np.zeros(len(vdi))
    norm_c = np.array(cap_idx)

    for i in range(len(vdi)):
        norm_v[i] = (vdi[i] - minv) / (maxv-minv)
    #for j in range(len(cap_idx)):
        #norm_c[j] = (cap_idx[j] ** 0.5) / 12
    n = np.concatenate((norm_v,norm_c))

    return n

# 运行程序获得VDI分布图,即得到csv数据
def run_os():
    os.system('ngspice -b interposer1_tr.sp -r interposer1_tr.raw')
    os.system('bin/inttrvmap int1.conf interposer1_tr.raw 1.0 0.05')


def move(action, cap_idx):
    while 1:
        if action == 0:
            break
        if action == 1:
            if cap_idx % 12 == 0:
                if cap_idx == 0:
                    action = random.choice([3, 2, 0])
                    continue
                if cap_idx == 132:
                    action = random.choice([3, 4, 0])
                    continue
                action = random.choice([3, 2, 0, 4])
                continue
            cap_idx -= 1
            break
        if action == 3:
            if cap_idx % 12 == 11:
                if cap_idx == 11:
                    action = random.choice([1, 2, 0])
                    continue
                if cap_idx == 143:
                    action = random.choice([1, 4, 0])
                    continue
                action = random.choice([1, 2, 0, 4])
                continue
            cap_idx += 1
            break

        if action == 2:
            if 132 < cap_idx < 143:
                action = random.choice([3, 4, 1, 0])
                continue
            if cap_idx == 143:
                action = random.choice([1, 4, 0])
                continue
            if cap_idx == 132:
                action = random.choice([3, 4, 0])
                continue
            cap_idx += 12
            break

        if action == 4:
            if 0 < cap_idx < 11:
                action = random.choice([3, 2, 1, 0])
                continue
            if cap_idx == 0:
                action = random.choice([1, 2, 0])
                continue
            if cap_idx == 11:
                action = random.choice([3, 2, 0])
                continue
            cap_idx -= 12
            break

    return action, cap_idx


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(154, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 50),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.model(x)
        return out


class Agent:

    def __init__(self, dcnum, action_space):
        self.target_net, self.eval_net = Net(), Net()
        self.memorysize = memorysize
        self.memory = np.zeros([memorysize, (nstate + dcnum) * 2 + 11])
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.learn_step_counter = 0
        self.dcnum = dcnum
        self.action_space = action_space
        self.mem_cnt = 0
        self.a_p = []

    def reset(self):
        str = ''
        f = open('vdd_decap.1', 'w')
        f.write(str)
        f.close()
        run_os()
       # state_ = readvdi('chiplet1_vdd_1_vdi.csv')


    def initial_loc(self, dc_val):
        cap_idx = random.sample([a for a in range(nstate)], 10)
        cap_idx.sort()
        str = ''
        for i in range(len(cap_idx)):
            str += 'c_decap_%d_%d nd_1_0_%d_%d 0 %e\n' % (i, cap_idx[i], node_loc[cap_idx[i]][0],
                                                           node_loc[cap_idx[i]][1], dc_val)
        f1 = open('vdd_decap.1', 'w')
        f1.write(str)
        f1.close()
        return cap_idx

    def csact(self, input_state) -> list:
        action_move = []
        input1 = torch.unsqueeze(torch.FloatTensor(input_state), 0)
        #input1 = torch.FloatTensor(input_state)
        action_prob = torch.FloatTensor(self.target_net(input1))
        self.a_p.append(action_prob)
        for j in range(50):
            if action_prob[0,j] == 0.:
                action_prob[0,j] = 1e-4
        for i in range(10):
            dist = Categorical(action_prob[0,i * 5:i * 5 + 5])
            idx = dist.sample()  # 返回0~4的tensor
           # idx_list = idx.tolist()
            #action_move.append((idx + i * 5).tolist())
            action_move.append(idx.item()+i*5)

        return action_move  # 返回的是序列号[0,49] sampling 10

    def step(self, a, cap_idx, dc_val):
        cap = deepcopy(cap_idx)
        #print(cap)
        action = deepcopy(a)
        #print(action)
        for i in range(len(a)):
            action[i], cap[i] = move((a[i]%5), cap_idx[i])
        #print('action',action)
        #print('cap',cap)
        str_dc = ''
        for j in range(len(cap)):
            str_dc += 'c_decap_%d_%d nd_1_0_%d_%d 0 %e\n' % (j, cap[j], node_loc[cap[j]][0],
                                                              node_loc[cap_idx[j]][1], dc_val)
        print(str_dc)
        f = open('vdd_decap.1', 'w')
        f.write(str_dc)
        f.close()
        run_os()
        state_ = readvdi('chiplet1_vdd_1_vdi.csv')
        return state_, action, cap

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))  # replace the old memory with new memory
        index = self.mem_cnt % memorysize
        self.memory[index, :] = transition
        self.mem_cnt += 1

    def update(self):
        if self.learn_step_counter % itertimes == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        sample_index = np.random.choice(min(memorysize, self.mem_cnt), batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_s = torch.FloatTensor(batch_memory[:, :nstate + self.dcnum])
        batch_a = torch.LongTensor(batch_memory[:, nstate+10:nstate + 20].astype(int))
        batch_r = torch.FloatTensor(batch_memory[:, nstate + 20:nstate + 21])
        batch_s_ = torch.FloatTensor(batch_memory[:, -nstate-self.dcnum:])

        # MSELoss
        np.savetxt('mem.txt',self.memory[:,  nstate+10:nstate + 20],fmt='%d')
        np.savetxt('a_array.txt',a_array,fmt='%d')
        np.savetxt('a_0_49.txt',np.array(acs),fmt='%d')
        q_eval = self.eval_net(batch_s).gather(1, batch_a)  # shape = (batch, 10)
        q_next = self.target_net(batch_s_).detach()
        q_next_max = torch.zeros([batch_size,10])
        for i in range(batch_size):
            for h in range(10):
                q_next_max[i][h] = max(q_next[i][h * 5:h * 5 + 5])
        q_target = batch_r + gamma * q_next_max
        loss = self.loss_func(q_eval, q_target)
        loss_val.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


episodes = 1
dnn = Agent(dcnum=10, action_space=50)
target_vdi = 0.9e-10        # 1.5 -> 1.2
reward = []
total_vdis=[]
cap_list = []
cap_idx = np.zeros(10)
obs = np.zeros(144)
record = None
a_change = []
acs = []
q_target = 0
for eps in range(episodes):
    if eps == 0:
        dnn.reset()
        fig_num = savefig(0,fig_num,'initial')
        cap_idx = dnn.initial_loc(1e-9)
        record = cap_idx
        run_os()
        obs = readvdi('chiplet1_vdd_1_vdi.csv')
        
    if (eps + 1) % 500 == 0:
        str_dc = ''
        for j in range(len(record)):
            str_dc += 'c_decap_%d_%d nd_1_0_%d_%d 0 %e\n' % (j, record[j], node_loc[record[j]][0],
                                                              node_loc[record[j]][1], 1e-9)
        print(str_dc)
        f = open('vdd_decap.1', 'w')
        f.write(str_dc)
        f.close()

        #cap_idx = dnn.initial_loc(1e-9)
    s = norm(obs, cap_idx)
    a = dnn.csact(s)
    acs.append(a)                       # initial action
    obs_, a, cap = dnn.step(a, cap_idx, 1e-9)
    a_change.append(a)
    a_array = np.array(a_change)        # modified action
    cap_list.append(cap)                # cap move data
    r = (target_vdi - np.sum(obs_)) / target_vdi
    s_ = norm(obs_,cap_idx)
    dnn.store_transition(s,a,r,s_)
    reward.append(r)
    total_vdis.append(np.sum(obs_))
    obs = obs_
    cap_idx = cap
    if dnn.mem_cnt > 0.9 * memorysize:
        dnn.update()
    if r > 0:
        fig_num = savefig(cap,fig_num,np.sum(obs_))

loss_list = torch.tensor(loss_val)
torch.save(dnn,'dnn_net.pth')
cap1 = np.array(cap_list)
np.savetxt('cap_idx.txt',cap1,fmt='%d',delimiter='\t')








plt.plot([a1 for a1 in range(len(cap1))],cap1)
plt.title('cap move')
plt.figure()




plt.plot([a for a in range(len(loss_list))],loss_list)
plt.title('loss funtion')
plt.xlabel('episodes')
plt.ylabel('value')
plt.figure()
plt.plot([b for b in range(len(reward))],reward)
plt.title('reward')
plt.figure()
plt.plot([c for c in range(len(total_vdis))],total_vdis)
plt.title('VDI plot')
plt.show()
