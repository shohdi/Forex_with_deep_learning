#!/usr/bin/env python3
import torch
import gym
import time
import argparse
import numpy as np
try:
    testVar = np.zeros((3,3),dtype=np.bool)
except :
    np.bool = bool
    testVar = np.zeros((3,3),dtype=np.bool)




from lib.SummaryWriter import SummaryWriter

from lib.env import ForexEnv
from dqn_rainbow import LSTM_Forex

import collections
import os
import warnings

DEFAULT_ENV_NAME = "Forex-100-15m-200max-100hidden-lstm-test"
MY_DATA_PATH = 'data'
FPS = 25


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    cudaDefault = False
    if (torch.cuda.is_available()):
        cudaDefault = True
    parser.add_argument("-c","--cuda", default=cudaDefault, help="Enable cuda")
    myFilePath = os.path.join(MY_DATA_PATH,DEFAULT_ENV_NAME + "-10000.dat")
    parser.add_argument("-m", "--model", default=myFilePath, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    #parser.add_argument("-r", "--record", help="Directory to store video recording")
    parser.add_argument("--no-visualize", default=True, action='store_false', dest='visualize',
                        help="Disable visualization of the game play")
    args = parser.parse_args()

    env = ForexEnv('minutes15_100/data/test_data.csv',True,False,True)
    device = torch.device("cuda" if args.cuda else "cpu")

    print("device : ",device)
    #if args.record:
    #    env = gym.wrappers.Monitor(env, args.record)
    net = LSTM_Forex(device, env.observation_space.shape, env.action_space.n).to(device)
    if os.path.exists(args.model):
        print('loading model')
        net.load_state_dict(torch.load(args.model, map_location=device))

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()
    printed_reward = 0.0
    gameNumber = 0
    writer = SummaryWriter(comment="-" + args.env)
    frameIdx = 0
    while gameNumber < 213:
        start_ts = time.time()
        if args.visualize:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            printed_reward += total_reward
            gameNumber += 1
            print ("finish game number " , gameNumber)
            print ("reward " , total_reward)
            print ("all reward " ,  printed_reward)
            writer.add_scalar("reward" , total_reward,frameIdx)
            writer.add_scalar("sum reward" , printed_reward,gameNumber)
            state = env.reset()
            total_reward = 0.0
#        if args.visualize:
#            delta = 1/FPS - (time.time() - start_ts)
#            if delta > 0:
#                time.sleep(delta)
        frameIdx +=1

#    if args.record:
#        env.env.close()