#!/usr/bin/env python3
from genericpath import exists
from imghdr import tests
import warnings
import gym
from lib.common import EpsilonTracker
from ptan import ptan
import argparse
import numpy as np
try:
    testVar = np.zeros((3,3),dtype=np.bool)
except :
    np.bool = bool
    testVar = np.zeros((3,3),dtype=np.bool)
    

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from lib.SummaryWriter import SummaryWriter

from lib import dqn_model, common
import os
import sys
import collections
from lib.env import ForexEnv
from datetime import datetime
import time

MY_DATA_PATH = 'data'
# n-step
REWARD_STEPS = 2

# priority replay
PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

# C51
Vmax = 0.03
Vmin = -0.03
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_val = nn.Sequential(
            dqn_model.NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            dqn_model.NoisyLinear(512, N_ATOMS)
        )

        self.fc_adv = nn.Sequential(
            dqn_model.NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            dqn_model.NoisyLinear(512, n_actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z-(DELTA_Z/1000.0) , DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        val_out = self.fc_val(conv_out).view(batch_size, 1, N_ATOMS)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + (adv_out - adv_mean)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())


class LSTM_Forex (nn.Module):
    def __init__(self,selDevice,input_shape,actions):

        super(LSTM_Forex,self).__init__()
        self.input_shape = input_shape
        self.actions = actions
        self.selected_device = selDevice
        self.inSize = self.input_shape[1]
        self.hiddenSize = 100
        self.numLayers = 2
        self.outSize = 512
        self.lstm = nn.LSTM(self.inSize,self.hiddenSize,self.numLayers,batch_first=True)
        """ self.size = np.prod(self.input_shape)
        
        self.network = nn.Sequential(
            nn.Linear(self.input_shape[0], self.hiddenSize),
            nn.ReLU(),
            nn.Linear(self.hiddenSize, self.hiddenSize),
            nn.ReLU()
        )
        """ 
        
        #self.lin = nn.Sequential(nn.Linear(self.input_shape[0],self.input_shape[0],True)
        #                         ,nn.ReLU())
        
        '''
        self.fc_val = nn.Sequential(
            dqn_model.NoisyLinear(self.input_shape[0], 512),
            nn.ReLU(),
            dqn_model.NoisyLinear(512, N_ATOMS)
        )

        self.fc_adv = nn.Sequential(
            dqn_model.NoisyLinear(self.input_shape[0], 512),
            nn.ReLU(),
            dqn_model.NoisyLinear(512, self.actions * N_ATOMS)
        )

        '''
        #self.lin = nn.Sequential(nn.Linear(self.hiddenSize,self.hiddenSize)
        #                         ,nn.ReLU())
        self.fc_val = nn.Sequential(
            dqn_model.NoisyLinear(self.hiddenSize, self.outSize),
            nn.ReLU(),
            dqn_model.NoisyLinear(self.outSize, N_ATOMS)
        )

        self.fc_adv = nn.Sequential(
            dqn_model.NoisyLinear(self.hiddenSize, self.outSize),
            nn.ReLU(),
            dqn_model.NoisyLinear(self.outSize, self.actions * N_ATOMS)
        )
        

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z-(DELTA_Z/1000.0), DELTA_Z))
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        batch_size = x.size()[0]
        h0 = torch.zeros(self.numLayers,x.size(0),self.hiddenSize,device=self.selected_device)
        c0 = torch.zeros(self.numLayers,x.size(0),self.hiddenSize,device=self.selected_device)
        out,(hn,cn) = self.lstm(x,(h0,c0))
        #out = self.network(x)
        #out = self.lin(out[:,-1,:])
        #out = x.view(batch_size,-1)
        #out = self.network(out)
        #firstLayerOut = self.lin(x)
        #val_out = self.fc_val(firstLayerOut).view(batch_size, 1, N_ATOMS)
        #adv_out = self.fc_adv(firstLayerOut).view(batch_size, -1, N_ATOMS)
        #val_out = self.fc_val(out).view(batch_size, 1, N_ATOMS)
        #adv_out = self.fc_adv(out).view(batch_size, -1, N_ATOMS)
        val_out = self.fc_val(out[:,-1,:]).view(batch_size, 1, N_ATOMS)
        adv_out = self.fc_adv(out[:,-1,:]).view(batch_size, -1, N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + (adv_out - adv_mean)
    

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())




def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    # next state distribution
    # dueling arch -- actions from main net, distr from tgt_net

    # calc at once both next and cur states
    distr_v, qvals_v = net.both(torch.cat((states_v, next_states_v)))
    next_qvals_v = qvals_v[batch_size:]
    distr_v = distr_v[:batch_size]

    next_actions_v = next_qvals_v.max(1)[1]
    next_distr_v = tgt_net(next_states_v)
    next_best_distr_v = next_distr_v[range(batch_size), next_actions_v.data]
    next_best_distr_v = tgt_net.apply_softmax(next_best_distr_v)
    next_best_distr = next_best_distr_v.data.cpu().numpy()

    dones = dones.astype(np.bool)

    # project our distribution using Bellman update
    proj_distr = common.distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

    # calculate net output
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)

    loss_v = -state_log_sm_v * proj_distr_v
    loss_v = batch_weights_v * loss_v.sum(dim=1)
    #loss_v =  loss_v.sum(dim=1)
    return loss_v.mean(), loss_v + 1e-5


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    startTime = time.time()
    testRewards = []#collections.deque(maxlen=100)
    
    testRewardsMean = 0
    newTestRewardsMean = 0
    valRewards = []#collections.deque(maxlen=100)
    valRewardsMean = 0
    params = common.HYPERPARAMS['Forex']
    params['epsilon_frames'] *= 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", default=False, action="store_true", help="Disable cuda")
    parser.add_argument("-f","--frame", default=0, help="Current Frame Idx")
    args = parser.parse_args()
    isCuda = torch.cuda.is_available()
    if args.cpu :
        isCuda = False
    device = torch.device("cuda" if isCuda else "cpu")
    print('device : ')
    print(device)
    if (not os.path.exists(MY_DATA_PATH)):
        os.makedirs(MY_DATA_PATH)
    modelRoot = os.path.join(MY_DATA_PATH,params['env_name'] + "-")
    modelCurrentPath = modelRoot + "current.dat"

    #env = gym.make(params['env_name'])
    #env = ptan.common.wrappers.wrap_dqn(env)
    #envTest = ptan.common.wrappers.wrap_dqn(gym.make(params['env_name']))
    #envVal = ptan.common.wrappers.wrap_dqn(gym.make(params['env_name']))
    env = ForexEnv('minutes15_100/data/train_data.csv',True,True,True ) 
    envTest = ForexEnv('minutes15_100/data/test_data.csv',False,True,True )
    envVal = ForexEnv('minutes15_100/data/train_data.csv',False,True,True )
    writer = SummaryWriter(comment="-" + params['run_name'] + "-rainbow")
    #net = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    #tgt_net = ptan.agent.TargetNet(net)
    net = LSTM_Forex(device,env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    
    if exists(modelCurrentPath):
        print('loading model ' , modelCurrentPath)
        net.load_state_dict(torch.load(modelCurrentPath,map_location=device))
        tgt_net.sync()
    
    epsilonTracker = EpsilonTracker(ptan.actions.EpsilonGreedyActionSelector(params['epsilon_start'],ptan.actions.ArgmaxActionSelector()),params)
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), epsilonTracker.epsilon_greedy_selector , device=device)
    
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=REWARD_STEPS)
    buffer = ptan.experience.PrioritizedReplayBuffer (exp_source, params['replay_size'], PRIO_REPLAY_ALPHA)
    #buffer = ptan.experience.ExperienceReplayBuffer(exp_source, params['replay_size'])#, PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = int(args.frame)
    beta = BETA_START
    test_idx = 0
    val_idx = 0
    mean_reward = -100
    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            try:

                frame_idx += 1
                epsilonTracker.frame(frame_idx)
                buffer.populate(1)
                beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if reward_tracker.reward(new_rewards[0], frame_idx,epsilonTracker.epsilon_greedy_selector.epsilon):
                        torch.save(net.state_dict(), modelCurrentPath)
                        break
                    if len(reward_tracker.total_rewards) >= 213 and reward_tracker.last_mean > mean_reward:
                        print('better mean reward  old %.5f new %.5f'%(mean_reward,reward_tracker.last_mean))
                        mean_reward = reward_tracker.last_mean
                        currentFilePath = '%s_%d_%.5f.dat'%(modelCurrentPath,frame_idx,mean_reward)
                        print('saving %s'%(currentFilePath))
                        torch.save(net.state_dict(), currentFilePath)

                if isCuda:
                    torch.cuda.empty_cache()

                
                currentTime = time.time()
                if (currentTime-startTime) > (5*60):
                    
                    print('sleeping 5 minutes on ' + str(datetime.now()))
                    sys.stdout.flush()
                    #if isCuda:
                    time.sleep(5*60)
                    print('resuming on ' + str(datetime.now()))
                    sys.stdout.flush()
                    startTime = time.time()
                
                if frame_idx > params['replay_size'] and len(buffer) < params['replay_size']:    
                #if frame_idx > 100000 and len(buffer) < 100000 :
                    #if isCuda:
                    #    time.sleep((1/250))
                    continue
                if len(buffer) < params['replay_initial']:
                    #if isCuda:
                    #    time.sleep((1/250))
                    continue

                #if isCuda:
                #    time.sleep((1/20))
                optimizer.zero_grad()
                
                
                batch,batch_indices, batch_weights = buffer.sample(params['batch_size'], beta)
                #batch_weights=[]
                #batch = buffer.sample(params['batch_size'])#, beta)
                loss_v, sample_prios_v = calc_loss(batch, batch_weights, net, tgt_net.target_model,
                                                params['gamma'] ** REWARD_STEPS, device=device)
                loss_v.backward()
                optimizer.step()
                buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

                
                
                
                if frame_idx % params['target_net_sync'] == 0:
                    tgt_net.sync()

                if frame_idx % 10000 == 0:
                    
                    torch.save(net.state_dict(), modelCurrentPath)
                
                

                if frame_idx % 200000 == 0:
                    
                    testIdx = 0
                    testRewards=[]
                    while testIdx < 100:
                        currentTime = time.time()
                        if (currentTime-startTime) > (5*60):
                            print('sleeping 5 minutes on ' + str(datetime.now()))
                            sys.stdout.flush()
                            #if isCuda:
                            time.sleep(5*60)
                            print('resuming on ' + str(datetime.now()))
                            sys.stdout.flush()
                            startTime = time.time()
                        testState = envTest.reset()
                        testState = np.array(testState,dtype=np.float32)
                        testIdx+=1
                        test_idx +=1
                        #start testing
                        rewardTest = 0
                        testSteps = 0
                        isDone = False
                        while not isDone:
                            #if isCuda:
                            #    time.sleep((1/250))
                            test_idx +=1
                            testSteps += 1
                            #play step
                            states_v = torch.tensor([testState]).to(device)
                            q_v = net.qvals(states_v)
                            q = q_v.detach().data.cpu().numpy()
                            actions = np.argmax(q, axis=1)
                            
                            testState,r_w,isDone,_ = envTest.step(actions[0])
                            rewardTest += r_w
                            testState = np.array(testState,dtype=np.float32)
                        testRewards.append(rewardTest)
                        testRewardsnp = np.array(testRewards,dtype=np.float32,copy=False)
                        newTestRewardsMean = np.mean(testRewardsnp)
                        writer.add_scalar("test mean reward",newTestRewardsMean,test_idx)
                        writer.add_scalar("test reward",rewardTest,test_idx)
                        writer.add_scalar("test steps",testSteps,test_idx)
                        print("test steps " + str(testSteps) + " test reward " + str(rewardTest) + ' mean test reward ' + str(newTestRewardsMean))
                        sys.stdout.flush()
                        if isCuda:
                            torch.cuda.empty_cache()
                    #if newTestRewardsMean > testRewardsMean and len(testRewards) >= 100 :
                    #    print('found new test mean %.6f old %.6f'%(newTestRewardsMean,testRewardsMean))
                    testRewardsMean = newTestRewardsMean
                    testPeriodPath = os.path.join(MY_DATA_PATH,params['env_name'] + ("-frameidx_%d-test_%.5f.dat"%(frame_idx, testRewardsMean)))
                    torch.save(net.state_dict(), testPeriodPath)
                

                    
                    valIndx = 0
                    valRewards=[]
                    while valIndx < 100:
                        currentTime = time.time()
                        if (currentTime-startTime) > (5*60):
                            print('sleeping 5 minutes on ' + str(datetime.now()))
                            sys.stdout.flush()
                            #if isCuda:
                            time.sleep(5*60)
                            print('resuming on ' + str(datetime.now()))
                            sys.stdout.flush()
                            startTime = time.time()
                        valState = envVal.reset()
                    
                        valState = np.array(valState,dtype=np.float32)
                        valIndx+=1
                        val_idx+=1
                        #start testing
                        rewardVal = 0
                        valSteps = 0
                        isDone = False
                        while not isDone:
                            if isCuda:
                                time.sleep((1/250))
                            val_idx+=1
                            valSteps += 1
                            #play step
                            states_v = torch.tensor([valState]).to(device)
                            q_v = net.qvals(states_v)
                            q = q_v.detach().data.cpu().numpy()
                            actions = np.argmax(q, axis=1)
                            
                            valState,r_w,isDone,_ = envVal.step(actions[0])
                            rewardVal += r_w
                            valState = np.array(valState,dtype=np.float32)
                        valRewards.append(rewardVal)
                        valRewardsnp = np.array(valRewards,dtype=np.float32,copy=False)
                        valRewardsMean = np.mean(valRewardsnp)
                        writer.add_scalar("val mean reward",valRewardsMean,val_idx)
                        writer.add_scalar("val reward",rewardVal,val_idx)
                        writer.add_scalar("val steps",valSteps,val_idx)
                        print("val steps " + str(valSteps) + " val reward " + str(rewardVal) + ' mean val reward ' + str(valRewardsMean))
                        sys.stdout.flush()
                        if isCuda:
                            torch.cuda.empty_cache()
                        
                    valPeriodPath = os.path.join(MY_DATA_PATH,params['env_name'] + ("-frameidx_%d-val_%.5f.dat"%(frame_idx,valRewardsMean)))
                    torch.save(net.state_dict(), valPeriodPath)
                    
            
        

            except Exception as err:
                print("exception happen : ")
                print(f"Unexpected {err=}, {type(err)=}")
                if isCuda:
                    time.sleep(1 * 60)
