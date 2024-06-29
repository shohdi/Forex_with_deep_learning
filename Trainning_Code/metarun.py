
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
from flask import Flask, jsonify
from flask_restful import Resource, Api,reqparse
from lib.metaenv import ForexMetaEnv
from lib.SummaryWriter import SummaryWriter
import argparse



from dqn_rainbow import LSTM_Forex
import time
import json
import pickle

from lib.check_trade_by_news import CheckTradeByNews



class Options:
    def __init__(self):
        self.ActionAvailable = False
        self.StateAvailable = False
        self.takenAction = 0
        self.tradeDir = 0
        self.stateObjTimes = collections.deque(maxlen=99)

options = dict()


stateObj = dict()

headers = ("open","close","high","low","ask","bid","volume","tradeDir","env","time")
envs = dict()
states= dict()
actions=dict()
lastStepRet = dict()

def hasKey(dic, key):
     
    if key in dic:
        return True
    else:
        return False

class StateObjLength(Resource):
    def get(self, envName):
        if envName not in stateObj:
            return jsonify({"found": False,"length":0})
        else:
            return jsonify({"found": True, "length": len(stateObj[envName])})

class MetaTrade(Resource):
    def get(self):
        
        parser = reqparse.RequestParser()
        parser.add_argument('open', type=float,location='args')
        parser.add_argument('close', type=float,location='args')
        parser.add_argument('high', type=float,location='args')
        parser.add_argument('low', type=float,location='args')
        parser.add_argument('ask', type=float,location='args')
        parser.add_argument('bid', type=float,location='args')
        parser.add_argument('volume', type=int,location='args')
        parser.add_argument('tradeDir' , type=int,location='args')
        parser.add_argument('env' , type=str,location='args')
        parser.add_argument('time' , type=str,location='args')
       
        args = parser.parse_args()
        open = args.open
        close = args.close
        high = args.high
        low = args.low
        ask = args.ask
        bid = args.bid
        volume = float(args.volume)
        tradeDir = args.tradeDir
        env = args.env
        time = args.time
        
        assert open > 0
        assert close > 0
        assert high > 0
        assert low > 0
        
        assert tradeDir == 0 or tradeDir == 1 or tradeDir == 2
       
        ret = doAction(open,close,high,low,ask,bid,volume,tradeDir,env,time,True)
        
        return ret

def doAction(open,close,high,low,ask,bid,volume,tradeDir,env,time,allowModel=True,action=0):
    if not allowModel :
        actions[env] = action
    if (not hasKey(options,env)) or options[env] is None :
        options[env] = Options()
        stateObj[env] = collections.deque(maxlen=99)
        options[env].stateObjTimes = collections.deque(maxlen=99)
        
    
    currentEnv = None
    if (not hasKey(envs,env)) or envs[env] is None:
        envs[env] = ForexMetaEnv(stateObj[env],options[env],env,False,True)
        envs[env] = loadEnv(env)
    currentEnv = envs[env]
    

    
    if  hasKey(actions,env) and actions[env] is not None   :
        currentEnv.beforeActionState = np.array(currentEnv.states,dtype=np.float32,copy=True)
        if len(options[env].stateObjTimes) > 0: 
            currentEnv.beforeActionTime = options[env].stateObjTimes[-1]
    #if len(options[env].stateObjTimes) == 0 or options[env].stateObjTimes[-1] != time:
    stateObj[env].append( np.array([open,close,high,low,ask,bid,volume],dtype=np.float32))
    options[env].stateObjTimes.append(time)
    #else:
    #    stateObj[env][-1] = np.array([open,close,high,low,ask,bid,volume],dtype=np.float32)

    options[env].tradeDir = tradeDir
    options[env].StateAvailable = True
    if hasKey(actions,env) and actions[env] is not None and actions[env] != 12 and hasKey(states,env) and states[env] is not None :
        
        stepState, reward, done, dataItem = currentEnv.step(actions[env])
        states[env] = stepState
        lastStepRet[env] = (stepState, reward, done, dataItem)
        if done:
            #if allowModel:
                #
            actions[env] = 12
            saveEnv(env)
            return str(12)

    elif  hasKey(actions,env) and actions[env] is not None and actions[env] == 12:
        states[env] = currentEnv.reset()
        
    

    
    if (not hasKey(states,env)) or states[env] is None :
        states[env] = currentEnv.reset()
        if states[env] is None :
            currentEnv.beforeActionState = None
            actions[env] = 12
            saveEnv(env)
            return str(12)
    

    if allowModel:
        net,device = configureApp(currentEnv)
        state_v = torch.tensor(np.array([states[env]], copy=False)).to(device)
        q_vals = net(state_v).cpu().data.numpy()[0]
        actions[env] = np.argmax(q_vals)
        currentEnv.nextAction = actions[env]
        currentEnv.nextProp = q_vals

        if currentEnv.openTradeDir == 0 and  actions[env] != 0:
            news = CheckTradeByNews()
            is_bad = news.check_currency_pair_is_bad(env,actions[env])
            if is_bad == True:
                print(f'{env} action is {actions[env]} bad decision regards to chatgpt news!')
                
                actions[env]=0
                currentEnv.nextAction = actions[env]
        
    else:
        actions[env] = action

    
    
    
    ret = str(actions[env])
    saveEnv(env)
    return ret


'''
def createEnvsThread():
    for attr, value in options.__dict__.items():
        env = stateObj[attr][0]
        if envs[env] is None:
            envs[env] = ForexMetaEnv(stateObj[env],options[env],False,True)
'''

   



DEFAULT_ENV_NAME = "Forex-100-15m-200max-100hidden-lstm-run"
MY_DATA_PATH = 'data'



def saveEnv(envName):
    saveObj  = {}
    saveObj['env'] = {}
    env = None
    if hasKey(envs,envName) and envs[envName] is not None:
        env = envs[envName]
    if env is None:
        return
    saveObj['env']['states'] = None
    if env.states is not None:
        saveObj['env']['states'] = []
        for i in range(len(env.states)):
            for j in range(len(env.states[i])):
                saveObj['env']['states'].append(env.states[i][j])
    
    saveObj['env']['envName'] = env.envName
    saveObj['env']['options'] = {}
    saveObj['env']['options']['ActionAvailable'] = env.options.ActionAvailable
    saveObj['env']['options']['StateAvailable'] = env.options.StateAvailable
    saveObj['env']['options']['takenAction'] = env.options.takenAction
    saveObj['env']['options']['tradeDir'] = env.options.tradeDir
    saveObj['env']['options']['stateObjTimes'] = None
    if env.options.stateObjTimes is not None:
        saveObj['env']['options']['stateObjTimes'] = []
        for i in range(len(env.options.stateObjTimes)):
            saveObj['env']['options']['stateObjTimes'].append(env.options.stateObjTimes[i])

    saveObj['env']['punishAgent'] = env.punishAgent
    saveObj['env']['stopTrade'] = env.stopTrade
    saveObj['env']['startTradeStep'] = env.startTradeStep
    saveObj['env']['startClose'] = env.startClose
    saveObj['env']['openTradeDir'] = env.openTradeDir
    saveObj['env']['lastTenData'] = None
    if env.lastTenData is not None:
        saveObj['env']['lastTenData'] = []
        for i in range(len(env.lastTenData)):
            for j in range(len(env.lastTenData[i])):
                saveObj['env']['lastTenData'].append(env.lastTenData[i][j])
    
    saveObj['env']['reward_queue'] = None
    if env.reward_queue is not None:
        saveObj['env']['reward_queue'] = []
        for i in range(len(env.reward_queue)):
            saveObj['env']['reward_queue'].append(env.reward_queue[i])
    
    saveObj['env']['startAsk'] = env.startAsk
    saveObj['env']['startBid'] = env.startBid
    saveObj['env']['openTradeAsk'] = env.openTradeAsk
    saveObj['env']['openTradeBid'] = env.openTradeBid
    saveObj['env']['stepIndex'] = env.stepIndex
    saveObj['env']['stopLoss'] = env.stopLoss
    saveObj['env']['beforeActionState'] = None
    if env.beforeActionState is not None :
        saveObj['env']['beforeActionState'] = []
        for i in range(len(env.beforeActionState)):
            for j in range(len(env.beforeActionState[i])):
                saveObj['env']['beforeActionState'].append(env.beforeActionState[i][j])
    
    saveObj['env']['beforeActionTime'] = env.beforeActionTime
    saveObj['env']['nextAction'] = env.nextAction
    saveObj['env']['nextProp'] = None
    if env.nextProp is not None:
        saveObj['env']['nextProp'] = []
        for i in range(len(env.nextProp)):
            saveObj['env']['nextProp'].append(env.nextProp[i])
    



    #env data
    saveObj['envData'] = {}
    saveObj['envData']['actions'] = None
    if hasKey(actions,envName) :
        saveObj['envData']['actions'] = actions[envName]
    saveObj['envData']['stateObj'] = None

    if hasKey(stateObj,envName) and stateObj[envName] is not None :
        saveObj['envData']['stateObj'] = []
        for i in range(len(stateObj[envName])):
            for j in range(len(stateObj[envName][i])):
                saveObj['envData']['stateObj'].append(stateObj[envName][i][j])
    

    saveObj['envData']['lastStepRet'] = None
    if hasKey(lastStepRet,envName) and lastStepRet[envName] is not None:
        saveObj['envData']['lastStepRet'] = {}
        saveObj['envData']['lastStepRet']['stepState'] = None
        if lastStepRet[envName][0] is not None:
            saveObj['envData']['lastStepRet']['stepState'] =[]
            for i in range(len(lastStepRet[envName][0])):
                for j in range(len(lastStepRet[envName][0][i])):
                    saveObj['envData']['lastStepRet']['stepState'].append(lastStepRet[envName][0][i][j])
        
        saveObj['envData']['lastStepRet']['reward'] = lastStepRet[envName][1]
        saveObj['envData']['lastStepRet']['done'] = lastStepRet[envName][2]
        saveObj['envData']['lastStepRet']['dataItem'] = lastStepRet[envName][3]
    
    saveObj['envData']['states'] = None
    if hasKey(states,envName) and states[envName] is not None:
        saveObj['envData']['states'] = []
        for i in range(len(states[envName])):
            for j in range(len(states[envName][i])):
                saveObj['envData']['states'].append(states[envName][i][j])


    if not os.path.isdir('env_data/'):
        os.mkdir('env_data')
    with open('env_data/' + envName  + '.pickle','wb') as myEnvFile:
        pickle.dump(saveObj,myEnvFile)#,default=lambda o: float(o) if isinstance(o, np.float32) else o)



def loadEnv(envName):

    if os.path.isfile('env_data/' + envName  + '.pickle'):
        saveObj = None
        with open('env_data/' + envName  + '.pickle','rb') as myEnvFile:
            saveObj=pickle.load(myEnvFile)
        
        if saveObj is not None:
            env = envs[envName]
            if hasKey(saveObj,'env') and saveObj['env'] is not None:
                if hasKey(saveObj['env'],'states') and saveObj['env']['states'] is not None:
                    #load states
                    statesData = saveObj['env']['states']
                    statesData = np.array(statesData,dtype=np.float32)
                    statesData = np.reshape(statesData,(99,-1))
                    statesObj = collections.deque(maxlen=99)
                    for i in range(len(statesData)):
                        statesObj.append(statesData[i])
                    env.states = statesObj
                    stateObj[envName] = statesObj
                
                env.envName = envName
                if hasKey(saveObj['env'],'options') and saveObj['env']['options'] is not None:
                    objOptions = Options()
                    objOptions.ActionAvailable = saveObj['env']['options']['ActionAvailable']
                    objOptions.StateAvailable = saveObj['env']['options']['StateAvailable']
                    objOptions.takenAction = saveObj['env']['options']['takenAction']
                    objOptions.tradeDir = saveObj['env']['options']['tradeDir']
                    objOptions.stateObjTimes = collections.deque(maxlen=99)
                    if  hasKey(saveObj['env']['options'],'stateObjTimes') and saveObj['env']['options']['stateObjTimes'] is not None:
                        for i in range(len(saveObj['env']['options']['stateObjTimes'])):
                            objOptions.stateObjTimes.append(saveObj['env']['options']['stateObjTimes'][i])
                    env.options = objOptions
                    options[envName] = objOptions
                env.punishAgent = saveObj['env']['punishAgent']
                env.stopTrade = saveObj['env']['stopTrade']
                env.startTradeStep = saveObj['env']['startTradeStep']
                env.startClose = saveObj['env']['startClose']
                env.openTradeDir = saveObj['env']['openTradeDir']
                #env.lastTenData = collections.deque(maxlen=10)
                if hasKey(saveObj['env'],'lastTenData') and saveObj['env']['lastTenData'] is not None:
                    lastTenDataArr = saveObj['env']['lastTenData']
                    lastTenDataLength = len(lastTenDataArr)
                    #lastTenDataObj = collections.deque(maxlen=10)
                    # convert the list to a numpy array
                    lastTenDataArr = np.array(lastTenDataArr)

                    # reshape the array into 10 rows x 5 columns
                    lastTenDataArr = lastTenDataArr.reshape(10, -1)
                    # convert the array into a list of tuples
                    lastTenDataArr = list(map(tuple, lastTenDataArr))
                    for i in range(len(lastTenDataArr)):
                        env.lastTenData.append(lastTenDataArr[i])
                
                if hasKey(saveObj['env'],'reward_queue') and saveObj['env']['reward_queue'] is not None:
                    reward_queueArr = saveObj['env']['reward_queue']
                    for i in range(len(reward_queueArr)):
                        env.reward_queue.append(reward_queueArr[i])
                
                env.startAsk = saveObj['env']['startAsk']
                env.startBid = saveObj['env']['startBid']
                env.openTradeAsk = saveObj['env']['openTradeAsk']
                env.openTradeBid = saveObj['env']['openTradeBid']
                env.stepIndex = saveObj['env']['stepIndex']
                env.stopLoss = saveObj['env']['stopLoss']
                if hasKey(saveObj['env'],'beforeActionState') and saveObj['env']['beforeActionState'] is not None:
                    #load states
                    statesData = saveObj['env']['beforeActionState']
                    statesData = np.array(statesData,dtype=np.float32)
                    statesData = np.reshape(statesData,(99,-1))
                    statesObj = collections.deque(maxlen=99)
                    for i in range(len(statesData)):
                        statesObj.append(statesData[i])
                    env.beforeActionState = statesObj
                
                env.beforeActionTime = saveObj['env']['beforeActionTime']
                env.nextAction = saveObj['env']['nextAction']
                nextPropArr = saveObj['env']['nextProp']
                if nextPropArr is not None:
                    env.nextProp = np.array(nextPropArr,dtype=np.float32)
            
            if hasKey(saveObj,'envData') and  saveObj["envData"] is not None:
                if hasKey(saveObj['envData'],'actions') and saveObj['envData']['actions'] is not None:
                    actions[envName] = saveObj['envData']['actions']

                

                if hasKey(saveObj['envData'],'lastStepRet') and saveObj['envData']['lastStepRet'] is not None:
                    lastStepRet[envName] = {}
                    lastStepRet[envName]['stepState'] = None
                    if saveObj['envData']['lastStepRet']['stepState'] is not None:
                        lastStepRet[envName]['stepState'] =[]
                        statesData = saveObj['envData']['lastStepRet']['stepState']
                        statesData = np.array(statesData,dtype=np.float32)
                        statesData = np.reshape(statesData,(99,-1))
                        for i in range(len(statesData)):
                            lastStepRet[envName]['stepState'].append(statesData[i])

                    lastStepRet[envName]['reward'] = saveObj['envData']['lastStepRet']['reward']
                    lastStepRet[envName]['done'] = saveObj['envData']['lastStepRet']['done']
                    lastStepRet[envName]['dataItem'] = saveObj['envData']['lastStepRet']['dataItem']
                    myTuble = (lastStepRet[envName]['stepState'], lastStepRet[envName]['reward'], lastStepRet[envName]['done'], lastStepRet[envName]['dataItem'])        
                    lastStepRet[envName] = myTuble

                if hasKey(saveObj['envData'],'states') and saveObj['envData']['states'] is not None:
                    statesObj = collections.deque(maxlen=99)
                    statesData = saveObj['envData']['states']
                    statesData = np.array(statesData,dtype=np.float32)
                    statesData = np.reshape(statesData,(99,-1))
                    for i in range(len(statesData)):
                        statesObj.append(statesData[i])
                    states[envName] = statesObj


        
            

    if hasKey(envs,envName) :
        return envs[envName]
    else:
        return None
        





def configureApp(env):
    warnings.filterwarnings("ignore")
    cudaDefault = False
    if (torch.cuda.is_available()):
        cudaDefault = True
    myFilePath = os.path.join(MY_DATA_PATH,DEFAULT_ENV_NAME + "-10000.dat")
    
    
    device = torch.device("cuda" if cudaDefault else "cpu")
    #print("device : ",device)
    net = LSTM_Forex(device, env.observation_space.shape, env.action_space.n).to(device)
    if os.path.exists(myFilePath):
        #print('loading model')
        net.load_state_dict(torch.load(myFilePath, map_location=device))
        
    net = net.qvals
    return net,device
    




def startApp():
    warnings.filterwarnings("ignore")
    cudaDefault = False
    if (torch.cuda.is_available()):
        cudaDefault = True
    myFilePath = os.path.join(MY_DATA_PATH,DEFAULT_ENV_NAME + "-10000.dat")
    env = ForexMetaEnv(stateObj,options,False,True)
    device = torch.device("cuda" if cudaDefault else "cpu")
    #print("device : ",device)
    net = LSTM_Forex(device, env.observation_space.shape, env.action_space.n).to(device)
    if os.path.exists(myFilePath):
        #print('loading model')
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
            #print ("finish game number " , gameNumber)
            #print ("reward " , total_reward)
            #print ("all reward " ,  printed_reward)
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
    news = CheckTradeByNews()
    is_bad  = news.check_currency_pair_is_bad('USDJPY',1)
    print(f'testing openai is working : USDJPY action is 1 checking bad decision regards to chatgpt news is {is_bad}')

    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--port", default=5000, help="port number")
    args = parser.parse_args()
    
    
    #start server
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(MetaTrade, '/')
    api.add_resource(StateObjLength, '/len/<string:envName>')
    app.run(host="0.0.0.0",port=args.port)




    









    

    
        



        
