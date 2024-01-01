
from typing import Collection
import gym
import gym.spaces
from gym.utils import seeding
import collections
import numpy as np
try:
    testVar = np.zeros((3,3),dtype=np.bool)
except :
    np.bool = bool
    testVar = np.zeros((3,3),dtype=np.bool)
import csv
import torch
import os
import warnings
from threading import Thread
from time import sleep
from flask import Flask
from flask_restful import Resource, Api,reqparse
from lib.metaenv import ForexMetaEnv
from lib.SummaryWriter import SummaryWriter
import argparse



from dqn_rainbow import LSTM_Forex
import time

class Options:
    def __init__(self):
        self.ActionAvailable = False
        self.StateAvailable = False
        self.takenAction = 0
        self.tradeDir = 0

options = Options()


stateObj = collections.deque(maxlen=16)
headers = ("open","close","high","low","ask","bid")

class MetaTrade(Resource):
    def get(self):
        while options.StateAvailable:
            None
        parser = reqparse.RequestParser()
        parser.add_argument('open', type=float,location='args')
        parser.add_argument('close', type=float,location='args')
        parser.add_argument('high', type=float,location='args')
        parser.add_argument('low', type=float,location='args')
        parser.add_argument('ask', type=float,location='args')
        parser.add_argument('bid', type=float,location='args')
        parser.add_argument('tradeDir' , type=int,location='args')
        parser.add_argument('day' , type=float,location='args')
        parser.add_argument('week' , type=float,location='args')
        parser.add_argument('month' , type=float,location='args')
        args = parser.parse_args()
        open = args.open
        close = args.close
        high = args.high
        low = args.low
        ask = args.ask
        bid = args.bid
        tradeDir = args.tradeDir
        day = args.day
        week = args.week
        month = args.month
        assert open > 0
        assert close > 0
        assert high > 0
        assert low > 0
        assert ask > 0
        assert bid > 0
        assert tradeDir == 0 or tradeDir == 1 or tradeDir == 2
        assert day > 0
        assert week > 0
        assert month > 0
        print("new state ",open,close,high,low,ask,bid,day,week,month)
        stateObj.append(np.array([open,close,high,low,ask,bid,day,week,month],dtype=np.float32))
        options.tradeDir = tradeDir
        options.StateAvailable = True
        while not options.ActionAvailable:
            None
        
        ret = str(options.takenAction)
        #print ('state : ',stateObj[-1])
        #print ('taken action : ',ret)
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
    env = ForexMetaEnv(stateObj,options,False,True)
    device = torch.device("cuda" if cudaDefault else "cpu")
    print("device : ",device)
    net = LSTM_Forex(device, env.observation_space.shape, env.action_space.n).to(device)
    if os.path.exists(myFilePath):
        print('loading model')
        net.load_state_dict(torch.load(myFilePath, map_location=device))
        state = env.reset()
    net = net.qvals
    total_reward = 0.0
    c = collections.Counter()
    printed_reward = 0.0
    gameNumber = 0
    writer = SummaryWriter(comment="-" + DEFAULT_ENV_NAME)
    frameIdx = 0
    while True:
        start_ts = time.time()

        state_v = torch.tensor(np.array([state], copy=False)).to(device)
        q_vals = net(state_v).cpu().data.numpy()[0]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--port", default=5000, help="port number")
    args = parser.parse_args()
    thread = Thread(target=startApp)
    thread.start()
    
    #start server
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(MetaTrade, '/')
    app.run(host="0.0.0.0",port=args.port)




    









    

    
        



        
