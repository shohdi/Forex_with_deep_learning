from ssl import ALERT_DESCRIPTION_INSUFFICIENT_SECURITY
import gym
import gym.spaces
from gym.utils import seeding
import collections
import numpy as np
import csv
import time

class ForexEnv(gym.Env):
    def __init__(self,filePath , haveOppsiteData:bool , punishAgent = True,stopTrade = True):
        self.haveOppsiteData = haveOppsiteData
        self.punishAgent = punishAgent
        self.stopTrade = stopTrade
        self.filePath = filePath
        self.action_space = gym.spaces.Discrete(n=3)
        
        
        self.startTradeStep = None
        self.startClose = None
        self.openTradeDir = 0
        self.lastTenData = collections.deque(maxlen=10)
        self.header = None
        self.data_arr = []
        self.data = None
        self.startAsk = None
        self.startBid = None
        self.openTradeAsk = None
        self.openTradeBid = None
        self.startIndex = None
        self.stepIndex = 0
        
        with open(self.filePath, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            self.header = next(reader)
            self.data_arr.append( np.array(list(reader)).astype(np.float32))
            if self.haveOppsiteData:
                self.data_arr.append(np.array(self.data_arr[0],copy=True))
                self.data_arr[1] = 1/self.data_arr[1]
                tempData = np.array(self.data_arr[1][:,4],copy=True)
                self.data_arr[1][:,4] = np.array(self.data_arr[1][:,5],copy=True)
                self.data_arr[1][:,5] = tempData
                tempData = np.array(self.data_arr[1][:,2],copy=True)
                self.data_arr[1][:,2] = np.array(self.data_arr[1][:,3],copy=True)
                self.data_arr[1][:,3] = tempData

            self.data = self.data_arr[np.random.randint(len(self.data_arr))]
        
        test_state = self.reset()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=test_state.shape, dtype=np.float32)



    def reset(self):
        self.lastTenData.append((self.startIndex,self.startTradeStep,self.startClose,self.startAsk,self.startBid,self.openTradeDir))
        #print(self.lastTenData[-1])
        self.data = self.data_arr[np.random.randint(len(self.data_arr))]
        self.startIndex = np.random.randint(len(self.data)-500)
        self.startTradeStep = None
        self.stepIndex = 0
        self.startClose = self.data[self.startIndex+ self.stepIndex][self.header.index("close")]

        self.openTradeDir = 0
        
        
        self.startAsk = self.data[self.startIndex+ self.stepIndex,self.header.index("ask")]
        self.startBid = self.data[self.startIndex+ self.stepIndex,self.header.index("bid")]
        self.openTradeAsk = None
        self.openTradeBid = None
        return self.getState()


    def step(self,action_idx):
        #check punish
        if self.openTradeDir == 1 and (self.stepIndex - self.startTradeStep) > 200 and self.stopTrade:
            action_idx = 2
        elif self.openTradeDir == 2 and (self.stepIndex - self.startTradeStep) > 200 and self.stopTrade:
            action_idx = 1

        #end of punish action

        #only one candle

        if self.openTradeDir == 1:
            action_idx = 2
        elif self.openTradeDir == 2:
            action_idx = 1



        #end of one candle


        reward = 0
        done = False
        if action_idx == 0:
            None
        elif action_idx == 1:
            #check open trade
            if self.openTradeDir == 0 :
                self.openUpTrade()
            elif self.openTradeDir == 1:
                None
            else :
                #close trade
                reward = self.closeDownTrade()
                done = True
        else :#2 :
            
            
            #check open trade
            if  self.openTradeDir == 0 :
                self.openDownTrade()
            elif self.openTradeDir == 2:
                None
            else : # 1
                #close trade
                reward = self.closeUpTrade()
                done = True
        if (self.stepIndex + self.startIndex) >= (len(self.data) - 400) and not done:
            if self.openTradeDir == 1 :
                reward = self.closeUpTrade()
            elif self.openTradeDir == 2 :
                reward = self.closeDownTrade()
            else:
                reward = 0
            done = True
        
        self.stepIndex+=1
        state = self.getState()
        
        
        if self.startTradeStep is None:
            if self.stepIndex > 200 and self.punishAgent:
                reward = -0.01
                done = True
        return state , reward , done ,None

        
    def getState(self):
        state = self.data[self.startIndex+self.stepIndex:(self.startIndex+self.stepIndex+100)]
       

        actions = np.zeros((100,4),dtype=np.float32)
        if self.openTradeDir == 1:
            actions[:,0] = self.openTradeAsk
        if self.openTradeDir == 2:
            actions[:,1] = self.openTradeBid
        
        


        
        
        
        state = np.concatenate((state,actions),axis=1)
        state = (state/self.startClose)/1.5
        state[:,-2] = self.stepIndex/(200.0 * 2.0)
        if self.startTradeStep is not None :
            
            state[:,-1] = (self.stepIndex - self.startTradeStep)/200.0
        state = state[-16:,:4]
        return state

    def openUpTrade(self):
        if self.openTradeDir == 1 or self.openTradeDir == 2:
            return
        self.openTradeDir = 1
        self.openTradeAsk = self.data[self.startIndex+self.stepIndex+99,self.header.index("close")]
        self.openTradeBid = self.data[self.startIndex+self.stepIndex+99,self.header.index("close")]
        self.startTradeStep = self.stepIndex

    def openDownTrade(self):
        if self.openTradeDir == 1 or self.openTradeDir == 2:
            return
        self.openTradeDir = 2
        self.openTradeAsk = self.data[self.startIndex+self.stepIndex+99,self.header.index("close")]
        self.openTradeBid = self.data[self.startIndex+self.stepIndex+99,self.header.index("close")]
        self.startTradeStep = self.stepIndex


    def closeUpTrade(self):
        if  self.openTradeDir == 0 or self.openTradeDir == 2:
            return
        currentBid = self.data[self.startIndex+self.stepIndex+99,self.header.index("close")]
        return ((currentBid - self.openTradeAsk)/self.startClose)/1.5

    def closeDownTrade(self):
        if  self.openTradeDir == 0 or self.openTradeDir == 1:
            return
        currentAsk = self.data[self.startIndex+self.stepIndex+99,self.header.index("close")]
        return ((self.openTradeBid - currentAsk)/self.startClose)/1.5

    def analysisUpTrade(self):
        startStep = self.startIndex + self.stepIndex
        currentStep = startStep
        startAsk = self.data[startStep+99,self.header.index("ask")]
        currentBid = self.data[currentStep+99,self.header.index("bid")]
        diff = (startAsk - currentBid)
        while (( startAsk - currentBid) < (2*diff) and (currentBid - startAsk) < (diff) and currentStep < (len(self.data)-500) and (currentStep - startStep) < 200 ):
            currentStep += 1
            currentBid = self.data[currentStep+99,self.header.index("bid")]
        if currentStep == (len(self.data)-500) or ((currentStep - startStep) >= 200) :
            #end of game
            return False,None
        
        if (currentBid - startAsk) >= (diff):
            #win
            return True,currentStep - self.startIndex
        else:
            #loss
            return False,None

    def analysisDownTrade(self):
        startStep = self.startIndex + self.stepIndex
        currentStep = startStep
        startBid = self.data[startStep+99,self.header.index("bid")]
        currentAsk = self.data[currentStep+99,self.header.index("ask")]
        diff = (currentAsk - startBid)
        while (( currentAsk - startBid) < (2*diff) and (startBid - currentAsk) < (diff) and currentStep < (len(self.data)-500) and (currentStep - startStep) < 200):
            currentStep += 1
            currentAsk = self.data[currentStep+99,self.header.index("ask")]
        if currentStep == (len(self.data)-500) or ((currentStep - startStep) >= 200):
            #end of game
            return False,None
        
        if (startBid - currentAsk) >= (diff):
            #win
            return True,currentStep - self.startIndex
        else:
            #loss
            return False,None
        

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random((int(time.time()*10000000)%2**31) if seed is None else seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
        #return [int(time.time()*1000000)%2**31,int(time.time()*1000000)%2**31]








'''
if __name__ == "__main__":
    env = ForexEnv("minutes15_100/data/test_data.csv")
    print(env.reset())
    i = 0
    sumReward = 0
    reward = 0
    done = False
    win = False
    while i < 500 :
        i+=1
        print(i)
        win = False
        while not win :
            win,winStep = env.analysisUpTrade()

            if(win):
                #up is sucess
                _,reward,done,_ = env.step(1)
                while env.stepIndex < winStep and not done:
                    _,reward,done,_ = env.step(0)
                
                _,reward,done,_ = env.step(2)
            
            if not win :
                win,winStep = env.analysisDownTrade()
                if(win):
                    #down is success
                    _,reward,done,_ = env.step(2)
                    while env.stepIndex < winStep and not done:
                        _,reward,done,_ = env.step(0)
                    
                    _,reward,done,_ = env.step(1)   

            
            
            if done :
                print ("reward : " , reward)
                sumReward += reward
                env.reset()
                reward = 0
               
                print ("ten last one ",env.lastTenData[-1] )
            else:
                _,reward,done,_ = env.step(0)
            
    
    print ("average reward ",sumReward/i)
'''


if __name__ == "__main__":
    env = ForexEnv("minutes15_100/data/train_data.csv",True)
    state = env.reset()
    print("start " , state[0])
    print("start close ",env.startClose)
    
    done = False
    while not done :
        print("last ",state[-1])
        action = int(input("action 0,1,2 : "))
        state,reward,done,_ = env.step(action)
        print('reward ',reward)
    





            
    


    




        



    

    
        



        
