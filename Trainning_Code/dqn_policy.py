#!/usr/bin/env python3

from lib import dqn_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import glob
import pickle
import warnings
import math
from lib import wrappers




from lib.SummaryWriter import SummaryWriter

from lib.env import ForexEnv
from lib.dqn_model import Vmax
from lib.dqn_model import Vmin
from lib.dqn_model import N_ATOMS
from lib.dqn_model import DELTA_Z



#DEFAULT_ENV_NAME = "Forex-100-15m-200max-100hidden-lstm"
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 20

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 1000000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = BATCH_SIZE * 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5 #* BATCH_SIZE
EPSILON_START = 1
EPSILON_FINAL = 0.02
WIN_STEP_START = 0
WIN_STEP_FINAL = 0
WIN_STEP_DECAY_LAST_FRAME = 10**5 * BATCH_SIZE
MY_DATA_PATH = 'data'


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class FileDataset(torch.utils.data.Dataset):
    def __init__(self,root,itemLen):
        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.maxLen = itemLen
        print('start counting files!')
        self.len  = len(glob.glob1(root,"*.pickle"))
        print('counted files is {}'.format(self.len))
        
        if(self.len > self.maxLen):
            self.len =self.maxLen
        
        
        self.pos = (self.len % self.maxLen)


        print('start check file sizes!')
        if self.len > 0:
            idx = self.len-1
            b = os.path.getsize(os.path.join(self.root,str(idx) + '.pickle'))
            if b == 0:
                print('found corrupted! {}'.format(idx))
                
                self.__setitem__(idx,self.__getitem__((idx + 1)%self.len))
            if idx % 10000 == 0:
                print('finished check number {}'.format(idx))
                

                
                    


    
    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        obj = pickle.load(open(os.path.join(self.root,str(idx) + '.pickle'),'rb'))
        return obj
    

    def __setitem__(self,idx,obj):
        pickle.dump(obj,open(os.path.join(self.root,str(idx) + '.pickle'),'wb'))
    
    def append(self,obj):
        
        self.__setitem__(self.pos,obj)
        self.pos += 1
        self.pos = (self.pos % self.maxLen)
        if(self.len < self.maxLen):
            self.len +=1

    def __str__(self):
        ret = '[\n'
        found = False
        for i in range(self.len):
            ret = ret + str(self.__getitem__(i))
            ret = ret + ',\n'
            found = True
        if found:
            ret = ret[:-2]
            ret = ret + '\n'
        ret = ret + ']\n'
        return ret



    def __iter__(self):
        
        for i in range(self.len):
            yield self.__getitem__(i)

        


class ExperienceBuffer:
    def __init__(self,capacity):
        self.capacity = capacity
        #self.buffer = FileDataset(buffer_path,capacity)
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)
    
    def clear(self):
        self.buffer = collections.deque(maxlen=self.capacity)



class Agent:
    def __init__(self, env, exp_buffer,envTest):
        self.env = env
        self.envTest = envTest
        self.exp_buffer = exp_buffer
        self.win = False
        self.winStep = None
        self.tradeDir = 0
        self.actionTraded = 0
        self.game_count = 0
        self._reset()
        self._resetTest()
        


    def _reset(self):
        self.state = self.env.reset()
        
        self.total_reward = 0.0
        
        self.win = False
        self.winStep = None
        self.tradeDir = 0
        self.actionTraded = 0
        self.game_count+=1
    
    def _resetTest(self):
        self.stateTest = self.envTest.reset()
        self.total_rewardTest = 0.0


    def play_stepWin(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        action = None
        if not self.win :
            self.win,self.winStep = self.env.analysisUpTrade()
            if self.win:
                self.tradeDir = 1
            if not self.win:
                self.win,self.winStep = self.env.analysisDownTrade()
                if self.win:
                    self.tradeDir = 2

        if self.actionTraded == 0 and self.win :
            #take action
            action = self.tradeDir
            self.actionTraded = action
        elif self.actionTraded != 0 and self.env.stepIndex >= self.winStep:
            if self.actionTraded == 1:
                action = 2
            else :
                action = 1
        else :
            action = 0

                

        

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net.qvals(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    def play_step_test(self, net, device="cpu"):
        done_reward = None

        
        
        state_a = np.array([self.stateTest], copy=False)
        state_v = torch.tensor(state_a).to(device)
        q_vals_v = net.qvals(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.envTest.step(action)
        self.total_rewardTest += reward

        
        self.stateTest = new_state
        if is_done:
            done_reward = self.total_rewardTest
            self._resetTest()
        return done_reward



class AgentPolicy:
    def __init__(self, envs, exp_buffer,envTest,currentFrame,gameCount):
        self.envs = envs
        self.envTest = envTest
        self.envVal = envs[-1]
        self.exp_buffer = exp_buffer
        self.currentFrame = currentFrame
        self.win = [ False for y in self.envs]
        self.winStep = [None for y in self.envs]
        self.tradeDir = [0 for y in self.envs]
        self.actionTraded = [0 for y in self.envs]
        self.game_count = gameCount

        self.currentWinStepValue = WIN_STEP_START
        self.total_reward = [0.0 for env in self.envs]
        self.state= [None for env in self.envs]
        #self.gameSteps = [env.stepIndex for env in self.envs]
        self.gameSteps = [0 for env in self.envs]
        _=[self._reset(i) for i in range(len(self.envs)) ]
        self._resetTest()
        self._resetVal()
        
    def calcWinStep(self):
        self.currentWinStepValue = WIN_STEP_START - int(round(((WIN_STEP_START-WIN_STEP_FINAL)/WIN_STEP_DECAY_LAST_FRAME ) * self.currentFrame))
        if self.currentWinStepValue < WIN_STEP_FINAL:
            self.currentWinStepValue = WIN_STEP_FINAL

    def _reset(self,envIndex):
        self.currentFrame +=1# self.envs[envIndex].stepIndex
        self.state[envIndex] = self.envs[envIndex].reset()
        
        self.total_reward[envIndex] = 0.0
        
        self.win[envIndex] = False
        self.winStep[envIndex] = None
        self.tradeDir[envIndex] = 0
        self.actionTraded[envIndex] = 0
        self.game_count+=1
        self.calcWinStep()
    
    def _resetTest(self):
        self.stateTest = self.envTest.reset()
        self.total_rewardTest = 0.0
    
    def _resetVal(self):
        self.stateVal = self.envVal.reset()
        self.total_rewardVal = 0.0


    def play_stepWin(self,envIndex):
        done_reward = None
        action = None
        
        if not self.win[envIndex] :
            self.win[envIndex],self.winStep[envIndex] = self.envs[envIndex].analysisUpTrade()
            if self.win[envIndex]:
                self.tradeDir[envIndex] = 1
            if not self.win[envIndex]:
                self.win[envIndex],self.winStep[envIndex] = self.envs[envIndex].analysisDownTrade()
                if self.win[envIndex]:
                    self.tradeDir[envIndex] = 2

        if self.actionTraded[envIndex] == 0 and self.win[envIndex] :
            #take action
            action = self.tradeDir[envIndex]
            self.actionTraded[envIndex] = action
        elif self.actionTraded[envIndex] != 0 and self.envs[envIndex].stepIndex >= self.winStep[envIndex]:
            if self.actionTraded[envIndex] == 1:
                action = 2
            else :
                action = 1
        else :
            action = 0

                

        

        return action
    
    def _step_action(self,envIndex,action):
        # do step in the environment
        done_reward = None
        new_state, reward, is_done, _ = self.envs[envIndex].step(action)
        self.total_reward[envIndex] += reward

        exp = Experience(self.state[envIndex], action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state[envIndex] = new_state
        if is_done:
            done_reward = self.total_reward[envIndex]
            self.gameSteps[envIndex] = 1#self.envs[envIndex].stepIndex
            self._reset(envIndex)
        return done_reward

    def getNetActions(self,state,net,device="cpu"):
        state_a = np.array(state, copy=False)
        state_v = torch.tensor(state_a).to(device)
        q_vals_v = net.qvals(state_v)
        q_vals_v = q_vals_v.detach().data.cpu().numpy()
        actions = np.argmax(q_vals_v, axis=1)
        
        return actions


    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        if self.game_count % ((WIN_STEP_START -  self.currentWinStepValue)+1) == 0 and self.currentWinStepValue > 0:
            action = [self.play_stepWin(envIndex) for envIndex in range(len(self.envs)) ]
        else: 
            if np.random.random() < epsilon:
                action = [env.action_space.sample() for env in self.envs]
            else:
                
                action = self.getNetActions(self.state,net,device)

        # do step in the environment
        action[0] =0
        action[1:4] = self.getNetActions(self.state[1:4],net,device)
        done_reward = [self._step_action(envIndex,action[envIndex]) for envIndex in range(len(self.envs))]
        return done_reward

    def play_step_test(self, net, device="cpu"):
        done_reward = None

        
        action = self.getNetActions([self.stateTest],net,device)[0]
        

        # do step in the environment
        new_state, reward, is_done, _ = self.envTest.step(action)
        self.total_rewardTest += reward

        
        self.stateTest = new_state
        if is_done:
            done_reward = self.total_rewardTest
            self._resetTest()
        return done_reward


    def play_step_val(self, net, device="cpu"):
        done_reward = None

        
        
        action = self.getNetActions([self.stateVal],net,device)[0]

        # do step in the environment
        new_state, reward, is_done, _ = self.envVal.step(action)
        self.total_rewardVal += reward

        
        self.stateVal = new_state
        if is_done:
            done_reward = self.total_rewardVal
            self._resetVal()
        return done_reward


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr

def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    batch_size = len(states)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    

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
    proj_distr = distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

    # calculate net output
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)

    loss_v = -state_log_sm_v * proj_distr_v
    loss_v =  loss_v.sum(dim=1)
    return loss_v.mean()



def createAgents(buffer):
    retColl = collections.deque(maxlen=BATCH_SIZE)
    i = 0
    for i in range(BATCH_SIZE):
        
        env = ForexEnv('minutes15_100/data/train_data.csv',True,True)
        envTest = ForexEnv('minutes15_100/data/test_data.csv',False,True)
        agent = Agent(env, buffer,envTest)
        retColl.append((env,envTest,agent))
    
    return retColl

def createOnePolicyAgents(buffer,currentFrame,gameCount):
    
    envs = [wrappers.make_env(args.env) for i in range(BATCH_SIZE)]
    #envs = [ForexEnv('minutes15_100/data/train_data.csv',True,True) for i in range(BATCH_SIZE)]  
    envTest = wrappers.make_env(args.env)
    #envTest = ForexEnv('minutes15_100/data/test_data.csv',False,True)
    agent = AgentPolicy (envs, buffer,envTest,currentFrame,gameCount)
    
    return agent

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    cudaDefault = False
    if (torch.cuda.is_available()):
        cudaDefault = True
    if (not os.path.exists(MY_DATA_PATH)):
        os.makedirs(MY_DATA_PATH)
    
    parser.add_argument("-f","--frame", default=0, help="Current Frame Idx")
    parser.add_argument("-g","--gameCount", default=0, help="Current game count")     
    parser.add_argument("-c","--cuda", default=cudaDefault, help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    print("device : ",device)

    #env = wrappers.make_env(args.env)
    buffer_path = os.path.join(MY_DATA_PATH,'buffer')
    buffer_path = os.path.join(buffer_path,'data')
    buffer = ExperienceBuffer(10000)
    frame_idx = int(args.frame) #0#len(buffer)
    gameCount = int(args.gameCount)
    agent = createOnePolicyAgents(buffer,frame_idx,gameCount)
    
    env = agent.envs[0]
    net = dqn_model.DQN(env.observation_space.shape,env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape,env.action_space.n).to(device)
    #net = dqn_model.LSTM_Forex(device, env.observation_space.shape, env.action_space.n).to(device)
    #tgt_net = dqn_model.LSTM_Forex(device,env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(net)
    



    
    
    

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = collections.deque(maxlen=213)
    
    epsilon =  max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME) 
    ts_frame = frame_idx
    ts = time.time()
    best_mean_reward = None
    myFilePath = os.path.join(MY_DATA_PATH,args.env + "-best.dat")
    myFilePathTest = os.path.join(MY_DATA_PATH,args.env + "test-best.dat")
    myFilePath1000 = os.path.join(MY_DATA_PATH,args.env + "-10000.dat")
    if os.path.exists(myFilePath1000):
        print('loading model ' , myFilePath1000)
        net.load_state_dict(torch.load(myFilePath1000,map_location=device))
        tgt_net.load_state_dict(net.state_dict())
    
    testRewards = collections.deque(maxlen=213)
    testRewardsLastMean = -10000
    testRewardsMean = 0
    valRewards = collections.deque(maxlen=213)
    valRewardsLastMean = -10000
    valRewardsMean = 0
    while True:
        
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
         
        envTest = agent.envTest
        
        

        batch_rewards = agent.play_step(net,epsilon,device)
        for rewardIdx in range(len(batch_rewards)):
            frame_idx +=1
            reward = batch_rewards[rewardIdx]
            gameSteps = agent.gameSteps[rewardIdx]
            if reward is not None:
                total_rewards.append(reward)
                gameCount+=1
                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts_frame = frame_idx
                ts = time.time()
                mean_reward = np.mean(np.array(total_rewards,copy=False)[-100:])
                print("%d: done %d games game reward %.7f , game steps : %d , mean reward %.7f , epsilon %.2f, speed %.2f f/s" % (
                    frame_idx, gameCount , reward , gameSteps , mean_reward,epsilon,
                    speed
                ))
                writer.add_scalar("epsilon", epsilon, frame_idx)
                writer.add_scalar("speed", speed, frame_idx)
                writer.add_scalar("reward_100", mean_reward, frame_idx)
                writer.add_scalar("reward", reward, frame_idx)
                writer.add_scalar("steps", gameSteps, frame_idx)
                writer.add_scalar("win step value", agent.currentWinStepValue , frame_idx)
                
                if best_mean_reward is None or best_mean_reward < mean_reward:
                    
                    torch.save(net.state_dict(), myFilePath)
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward
                
                



                if mean_reward > args.reward:
                    print("Solved in %d frames!" % frame_idx)
                    break
            
            
            
            if frame_idx % 10000 == 0 and frame_idx > 0:
                torch.save(net.state_dict(), myFilePath1000)
        
            if frame_idx % (10000 * 213) == 0 and frame_idx > 10000:
                testIdx = 0
                while testIdx < 213:
                    testIdx+=1
                    #start testing
                    rewardTest = None
                    testSteps = 0
                    while rewardTest is None:
                        testSteps += 1
                        rewardTest = agent.play_step_test(net,device)
                    testRewards.append(rewardTest)
                    testRewardsnp = np.array(testRewards,dtype=np.float32,copy=False)
                    testRewardsMean = np.mean(testRewardsnp)
                    writer.add_scalar("test mean reward",testRewardsMean,frame_idx)
                    writer.add_scalar("test reward",rewardTest,frame_idx)
                    writer.add_scalar("test steps",testSteps,frame_idx)
                    print("test steps " + str(testSteps) + " test reward " + str(rewardTest) + ' mean test reward ' + str(testRewardsMean))
                testPeriodPath = os.path.join(MY_DATA_PATH,args.env + ("-%.5f.dat"%(testRewardsMean)))
                torch.save(net.state_dict(), testPeriodPath)
                print("test last mean reward before checking ",testRewardsLastMean)
                if (testRewardsLastMean < testRewardsMean and len(testRewards) == 213 ) or not os.path.exists(myFilePathTest)  :
                    if len(testRewards) == 213:
                        testRewardsLastMean = testRewardsMean
                    print("found better test model , saving ... ")
                    torch.save(net.state_dict(), myFilePathTest)
            
                valIndx = 0
                while valIndx < 213:
                    valIndx+=1
                    #start testing
                    rewardVal = None
                    valSteps = 0
                    while rewardVal is None:
                        valSteps += 1
                        rewardVal = agent.play_step_val(net,device)
                    valRewards.append(rewardVal)
                    valRewardsnp = np.array(valRewards,dtype=np.float32,copy=False)
                    valRewardsMean = np.mean(valRewardsnp)
                    writer.add_scalar("val mean reward",valRewardsMean,frame_idx)
                    writer.add_scalar("val reward",rewardVal,frame_idx)
                    writer.add_scalar("val steps",valSteps,frame_idx)
                    print("val steps " + str(valSteps) + " val reward " + str(rewardVal) + ' mean val reward ' + str(valRewardsMean))
                valPeriodPath = os.path.join(MY_DATA_PATH,args.env + ("-val-%.5f.dat"%(valRewardsMean)))
                torch.save(net.state_dict(), valPeriodPath)
               
                

            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())


        if len(buffer) < 10000:
            continue


        num =0
        while num < BATCH_SIZE:
            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net,GAMMA, device=device)
            loss_t.backward()
            optimizer.step()
            #buffer.clear()
            num+=1



    writer.close()
