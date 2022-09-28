#!/usr/bin/env python3
import gym
import time
import argparse
import numpy as np

import torch




import collections
import warnings
from lib.common import HYPERPARAMS
from ptan import ptan
import os
from dqn_rainbow import RainbowDQN
from genericpath import exists

FPS = 25
MY_DATA_PATH = 'data'

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    params = HYPERPARAMS['Pong']
    modelRoot = os.path.join(MY_DATA_PATH,params['env_name'] + "-")
    modelCurrentPath = os.path.join(MY_DATA_PATH,'model.dat')
    parser = argparse.ArgumentParser()

    parser.add_argument("--cpu", default=False, action="store_true", help="Disable cuda")
    args = parser.parse_args()

    env =ptan.common.wrappers.wrap_dqn( gym.make(params['env_name']))
    isCuda = torch.cuda.is_available()
    if args.cpu :
        isCuda = False
    device = torch.device("cuda" if isCuda else "cpu")
    net = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    if (not os.path.exists(MY_DATA_PATH)):
        os.makedirs(MY_DATA_PATH)
    if exists(modelCurrentPath):
        print('loading model ' , modelCurrentPath)
        net.load_state_dict(torch.load(modelCurrentPath,map_location=device))

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        
        env.render()
        state_v = torch.tensor(np.array([state], copy=False)).to(device)
        q_vals = net.qvals(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        
        delta = 1/FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)


