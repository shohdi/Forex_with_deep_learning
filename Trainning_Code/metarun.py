
from typing import Collection
import gym
import gym.spaces
from gym.utils import seeding
import collections
import numpy as np
import csv
import torch
import os
import warnings
from threading import Thread
from time import sleep
from flask import Flask
from flask_restful import Resource, Api
from lib.metaenv import ForexMetaEnv
from lib.SummaryWriter import SummaryWriter

from lib import dqn_model
import time

class Options:
    def __init__(self):
        self.ActionAvailable = False
        self.StateAvailable = False
        self.takenAction = 0

options = Options()


stateObj = collections.deque(maxlen=100)
headers = ("open","close","high","low","ask","bid")

class MetaTrade(Resource):
    def get(self,open,close,high,low,ask,bid):
        while options.StateAvailable:
            None
        
        stateObj.append(np.array([open,close,high,low,ask,bid],dtype=np.float32))
        options.StateAvailable = True
        while not options.ActionAvailable:
            None
        
        ret = str(options.takenAction)
        print ('state : ',stateObj[-1])
        print ('taken action : ',ret)
        options.ActionAvailable = False
        return ret


   



DEFAULT_ENV_NAME = "Forex-100-15m-200max-100hidden-lstm-run"
MY_DATA_PATH = 'data'

def startApp():
    warnings.filterwarnings("ignore")
    cudaDefault = False
    if (torch.cuda.is_available()):
        cudaDefault = True
    myFilePath = os.path.join(MY_DATA_PATH,DEFAULT_ENV_NAME + "-10000.dat")
    env = ForexMetaEnv(stateObj,options,False)
    device = torch.device("cuda" if cudaDefault else "cpu")
    print("device : ",device)
    net = dqn_model.LSTM_Forex(device, env.observation_space.shape, env.action_space.n).to(device)
    if os.path.exists(myFilePath):
        net.load_state_dict(torch.load(myFilePath, map_location=device))
        state = env.reset()
    total_reward = 0.0
    c = collections.Counter()
    printed_reward = 0.0
    gameNumber = 0
    writer = SummaryWriter(comment="-" + DEFAULT_ENV_NAME)
    frameIdx = 0
    while True:
        start_ts = time.time()

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




if __name__ == "__main__":
    
    thread = Thread(target=startApp)
    thread.start()
    
    #start server
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(MetaTrade, '/')
    app.run(port=80)




    









    

    
        



        
