from ssl import ALERT_DESCRIPTION_INSUFFICIENT_SECURITY
import gym
import collections
import numpy as np
import csv

class ForexEnv(gym.Env):
    def __init__(self,filePath):
        self.filePath = filePath
        
        self.startTradeStep = None
        self.startClose = None
        self.openTradeDir = None
        self.lastTenData = collections.deque(maxlen=10)
        self.header = None
        self.data = None
        self.startAsk = None
        self.startBid = None
        self.step = 0
        
        with open(self.filePath, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            self.header = next(reader)
            self.data = np.array(list(reader)).astype(np.float32)
        
        self.reset()



    def reset(self):
        self.lastTenData.append((self.startIndex,self.startTradeStep,self.startClose,self.startAsk,self.startBid,self.openTradeDir))
        self.startIndex = np.random.randint(len(self.data)-500)
        self.startTradeStep = None
        self.startClose = self.data[self.startIndex][self.header.index("close")]

        self.openTradeDir = None
        self.step = 0
        
        self.startAsk = self.data[self.startIndex,self.header.index("ask")]
        self.startBid = self.data[self.startIndex,self.header.index("bid")]

    def step(self,action_idx):
        reward = 0
        done = False
        if action_idx == 0:
            None
        elif action_idx == 1:
            #check open trade
            if self.openTradeDir is None or self.openTradeDir == 0 :
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
                if self.openTradeDir is None or self.openTradeDir == 0 :
                    self.openDownTrade()
                elif self.openTradeDir == 2:
                    None
                else : # 1
                    #close trade
                    reward = self.closeUpTrade()
                    done = True
        if self.step + self.startIndex == len(self.data) - 100:
            if self.openTradeDir == 1 :
                reward = self.closeUpTrade()
            elif self.openTradeDir == 2 :
                reward = self.closeDownTrade()
            else
                reward = 0
            done = True
        

        state = self.getState()

        return state , reward , done ,None

        



    

    
        



        
