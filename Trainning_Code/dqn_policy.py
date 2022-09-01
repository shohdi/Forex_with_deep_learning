#!/usr/bin/env python3

from lib import dqn_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import pickle
import warnings
import math





from lib.SummaryWriter import SummaryWriter

from lib.env import ForexEnv


DEFAULT_ENV_NAME = "Forex-100-15m-200max-100hidden-lstm"
MEAN_REWARD_BOUND = 0.01

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 1000000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1
EPSILON_FINAL = 0.0002
WIN_STEP_START = 32
WIN_STEP_FINAL = 0
WIN_STEP_DECAY_LAST_FRAME = 10**5
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
            q_vals_v = net(state_v)
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
        q_vals_v = net(state_v)
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
    def __init__(self, envs, exp_buffer,envTest,currentFrame,gameSteps):
        self.envs = envs
        self.envTest = envTest
        self.exp_buffer = exp_buffer
        self.currentFrame = currentFrame
        self.win = [ False for y in self.envs]
        self.winStep = [None for y in self.envs]
        self.tradeDir = [0 for y in self.envs]
        self.actionTraded = [0 for y in self.envs]
        self.game_count = gameSteps

        self.currentWinStepValue = WIN_STEP_START
        self.total_reward = [0.0 for env in self.envs]
        self.state= [None for env in self.envs]
        
        _=[self._reset(i) for i in len(self.envs) ]
        self._resetTest()
        
    def calcWinStep(self):
        self.currentWinStepValue = WIN_STEP_START - int(math.Round(((WIN_STEP_START-WIN_STEP_FINAL)/WIN_STEP_DECAY_LAST_FRAME ) * self.currentFrame))
        if self.currentWinStepValue < WIN_STEP_FINAL:
            self.currentWinStepValue = WIN_STEP_FINAL

    def _reset(self,envIndex):
        self.currentFrame += self.envs[envIndex].stepIndex
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
            self._reset(envIndex)
        return done_reward


    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        if self.game_count % ((WIN_STEP_START -  self.currentWinStepValue)+1) == 0:
            action = [self.play_stepWin(envIndex) for envIndex in len(self.envs) ]
        else: 
            if np.random.random() < epsilon:
                action = [env.action_space.sample() for env in self.envs]
            else:
                state_a = np.array(self.state, copy=False)
                state_v = torch.tensor(state_a).to(device)
                q_vals_v = net(state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
                action = act_v.detach().numpy()

        # do step in the environment
        done_reward = [self._step_action(envIndex,action[envIndex]) for envIndex in len(self.envs)]
        return done_reward

    def play_step_test(self, net, device="cpu"):
        done_reward = None

        
        
        state_a = np.array([self.stateTest], copy=False)
        state_v = torch.tensor(state_a).to(device)
        q_vals_v = net(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.detach().item())

        # do step in the environment
        new_state, reward, is_done, _ = self.envTest.step(action)
        self.total_rewardTest += reward

        
        self.stateTest = new_state
        if is_done:
            done_reward = self.total_rewardTest
            self._resetTest()
        return done_reward



def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

def createAgents(buffer):
    retColl = collections.deque(maxlen=BATCH_SIZE)
    i = 0
    for i in range(BATCH_SIZE):
        
        env = ForexEnv('minutes15_100/data/train_data.csv')
        envTest = ForexEnv('minutes15_100/data/test_data.csv')
        agent = Agent(env, buffer,envTest)
        retColl.append((env,envTest,agent))
    
    return retColl

def createOnePolicyAgents(buffer,currentFrame,gameSteps):
    
 
    envs = [ForexEnv('minutes15_100/data/train_data.csv') for i in range(BATCH_SIZE)]  
    envTest = ForexEnv('minutes15_100/data/test_data.csv')
    agent = AgentPolicy (envs, buffer,envTest,currentFrame,gameSteps)
    
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
    parser.add_argument("-g","--gameSteps", default=0, help="Current game count")     
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
    buffer = ExperienceBuffer(BATCH_SIZE)
    frame_idx = int(args.frame) #0#len(buffer)
    gameSteps = int(args.gameSteps)
    agent = createOnePolicyAgents(buffer,frame_idx,gameSteps)
    
    env = agent.envs[0]
    net = dqn_model.LSTM_Forex(device, env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.LSTM_Forex(device,env.observation_space.shape, env.action_space.n).to(device)
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
    while True:
        
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
         
        envTest = agent.envTest
        
        gameSteps = agent.game_count

        batch_rewards = agent.play_step(net,epsilon,device)
        for rewardIdx in range(len(batch_rewards)):
            frame_idx +=1
            reward = batch_rewards[rewardIdx]
            
            if reward is not None:
                total_rewards.append(reward)
                
                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts_frame = frame_idx
                ts = time.time()
                mean_reward = np.mean(np.array(total_rewards,copy=False)[-100:])
                print("%d: done %d games game reward %.7f , game steps : %d , mean reward %.7f , epsilon %.2f, speed %.2f f/s" % (
                    frame_idx, len(total_rewards) , reward , gameSteps , mean_reward,epsilon,
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
        
            if frame_idx % 10000 == 0 and frame_idx > 10000:
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
                if testRewardsLastMean < testRewardsMean:
                    testRewardsLastMean = testRewardsMean
                    print("found better test model , saving ... ")
                    torch.save(net.state_dict(), myFilePathTest)
                

            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())


        if len(buffer) < BATCH_SIZE:
            continue



        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
        buffer.clear()



    writer.close()
