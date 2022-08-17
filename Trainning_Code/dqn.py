#!/usr/bin/env python3
from lib import wrappers
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





from lib.SummaryWriter import SummaryWriter

from lib.env import ForexEnv


DEFAULT_ENV_NAME = "Forex-v4"
MEAN_REWARD_BOUND = 0.01

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 1000000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 0.02
EPSILON_FINAL = 0.02
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
    def __init__(self, buffer_path,capacity):
        self.buffer = FileDataset(buffer_path,capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer,envTest):
        self.env = env
        self.envTest = envTest
        self.exp_buffer = exp_buffer
        self.win = False
        self.winStep = None
        self.tradeDir = 0
        self.actionTraded = 0
        self._reset()
        self._resetTest()


    def _reset(self):
        self.state = self.env.reset()
        
        self.total_reward = 0.0
        
        self.win = False
        self.winStep = None
        self.tradeDir = 0
        self.actionTraded = 0
    
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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    cudaDefault = False
    if (torch.cuda.is_available()):
        cudaDefault = True
    if (not os.path.exists(MY_DATA_PATH)):
        os.makedirs(MY_DATA_PATH)
    
        
    parser.add_argument("--cuda", default=cudaDefault, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    print("device : ",device)

    #env = wrappers.make_env(args.env)
    env = ForexEnv('minutes15_100/data/train_data.csv')
    envTest = ForexEnv('minutes15_100/data/test_data.csv')
    net = dqn_model.LSTM_Forex(device, env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.LSTM_Forex(device,env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(net)
    
    buffer_path = os.path.join(MY_DATA_PATH,'buffer')
    buffer_path = os.path.join(buffer_path,'data')
    buffer = ExperienceBuffer(buffer_path,REPLAY_SIZE)
    agent = Agent(env, buffer,envTest)
    

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = len(buffer)
    epsilon =  max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME) 
    ts_frame = frame_idx
    ts = time.time()
    best_mean_reward = None
    myFilePath = os.path.join(MY_DATA_PATH,args.env + "-best.dat")
    myFilePath1000 = os.path.join(MY_DATA_PATH,args.env + "-10000.dat")
    if os.path.exists(myFilePath1000):
        net.load_state_dict(torch.load(myFilePath,map_location=device))
        tgt_net.load_state_dict(net.state_dict())
    gameSteps = 0
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = 0
        if frame_idx < 20000 or len(total_rewards) % 2 == 0:
            reward = agent.play_stepWin(net,epsilon,device=device)
        else :
            reward = agent.play_step(net, epsilon, device=device)
        gameSteps+=1
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games game reward %.7f , game steps : %d , mean reward %.7f, speed %.2f f/s" % (
                frame_idx, len(total_rewards) , reward , gameSteps , mean_reward,
                speed
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            writer.add_scalar("steps", gameSteps, frame_idx)
            gameSteps = 0
            if best_mean_reward is None or best_mean_reward < mean_reward:
                
                torch.save(net.state_dict(), myFilePath)
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            
            if frame_idx % 10000 == 0 and frame_idx > 0:
                torch.save(net.state_dict(), myFilePath1000)



            if mean_reward > args.reward:
                print("Solved in %d frames!" % frame_idx)
                break
        
        if frame_idx % 10000 == 0 and frame_idx > 0:
            #start testing
            rewardTest = None
            testSteps = 0
            while rewardTest is None:
                testSteps += 1
                rewardTest = agent.play_step_test(net,device)
            writer.add_scalar("test reward",rewardTest,frame_idx)
            writer.add_scalar("test steps",testSteps,frame_idx)
            print("test steps " + testSteps + " test reward " + rewardTest)
        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()



    writer.close()
