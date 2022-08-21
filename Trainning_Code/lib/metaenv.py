from ssl import ALERT_DESCRIPTION_INSUFFICIENT_SECURITY
import gym
import gym.spaces
from gym.utils import seeding
import collections
import numpy as np
import csv

class ForexMetaEnv(gym.Env):
    def __init__(self,punishAgent = True):
        self.states = collections.deque(maxlen=100)
        self.punishAgent = punishAgent
        
        self.action_space = gym.spaces.Discrete(n=3)
        
        
        self.startTradeStep = None
        self.startClose = None
        self.openTradeDir = 0
        self.lastTenData = collections.deque(maxlen=10)
        self.header = ("open","close","high","low","ask","bid")
        self.data = None
        self.startAsk = None
        self.startBid = None
        self.openTradeAsk = None
        self.openTradeBid = None
   
        self.stepIndex = 0
        
 
        
        test_state = self.reset()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=test_state.shape, dtype=np.float32)



    def reset(self):

        
        
        
        self.startTradeStep = None
        self.stepIndex = 0
        self.startClose = self.states[-1,self.header.index("close")]

        self.openTradeDir = 0
        
        
        self.startAsk = self.states[-1,self.header.index("ask")]
        self.startBid = self.states[-1,self.header.index("bid")]
        self.openTradeAsk = None
        self.openTradeBid = None
        return self.getState()


    def step(self,action_idx):
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
            if action_idx == 0:
                None
            elif action_idx == 2:
                #check open trade
                if  self.openTradeDir == 0 :
                    self.openDownTrade()
                elif self.openTradeDir == 2:
                    None
                else : # 1
                    #close trade
                    reward = self.closeUpTrade()
                    done = True
        if (self.stepIndex + self.startIndex) == (len(self.data) - 100) and not done:
            if self.openTradeDir == 1 :
                reward = self.closeUpTrade()
            elif self.openTradeDir == 2 :
                reward = self.closeDownTrade()
            else:
                reward = 0
            done = True
        

        state = self.getState()
        self.stepIndex+=1
        if self.startTradeStep is not None:
            if (self.stepIndex - self.startTradeStep) > 200 and self.punishAgent:
                reward = -1
                done = True
        if self.startTradeStep is None:
            if self.stepIndex > 200 and self.punishAgent:
                reward = -1
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
        state = (state/self.startClose)/2
        state[:,-2] = self.stepIndex/(200.0 * 2.0)
        if self.startTradeStep is not None :
            
            state[:,-1] = (self.stepIndex - self.startTradeStep)/200.0

        return state

    def openUpTrade(self):
        if self.openTradeDir == 1 or self.openTradeDir == 2:
            return
        self.openTradeDir = 1
        self.openTradeAsk = self.data[self.startIndex+self.stepIndex,self.header.index("ask")]
        self.openTradeBid = self.data[self.startIndex+self.stepIndex,self.header.index("bid")]
        self.startTradeStep = self.stepIndex

    def openDownTrade(self):
        if self.openTradeDir == 1 or self.openTradeDir == 2:
            return
        self.openTradeDir = 2
        self.openTradeAsk = self.data[self.startIndex+self.stepIndex,self.header.index("ask")]
        self.openTradeBid = self.data[self.startIndex+self.stepIndex,self.header.index("bid")]
        self.startTradeStep = self.stepIndex


    def closeUpTrade(self):
        if  self.openTradeDir == 0 or self.openTradeDir == 2:
            return
        currentBid = self.data[self.startIndex+self.stepIndex,self.header.index("bid")]
        return ((currentBid - self.openTradeAsk)/self.startClose)/2

    def closeDownTrade(self):
        if  self.openTradeDir == 0 or self.openTradeDir == 1:
            return
        currentAsk = self.data[self.startIndex+self.stepIndex,self.header.index("ask")]
        return ((self.openTradeBid - currentAsk)/self.startClose)/2


        

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]









if __name__ == "__main__":


    

    
        



        
