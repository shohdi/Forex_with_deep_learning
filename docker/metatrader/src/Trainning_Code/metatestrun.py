
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
from metarun import headers,options,stateObj,MetaTrade




import time

   



DEFAULT_ENV_NAME = "Forex-100-15m-200max-100hidden-lstm-run"
MY_DATA_PATH = 'data'

def startApp():
    warnings.filterwarnings("ignore")
    
    
    env = ForexMetaEnv(stateObj,options,False,True)
    state = env.reset()
    print("start " , state[0])
    print("start close ",env.startClose)
    
    done = False
    while not done :
        print("last ",state[-1])
        action = 0
        state,reward,done,_ = env.step(action)
        print('reward ',reward)




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
    app.run(port=args.port)




    









    

    
        



        
