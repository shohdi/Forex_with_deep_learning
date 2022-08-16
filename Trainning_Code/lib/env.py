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
        self.openTradeDir = 0
        self.lastTenData = collections.deque(maxlen=10)
        self.header = None
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
            self.data = np.array(list(reader)).astype(np.float32)
        
        self.reset()



    def reset(self):
        self.lastTenData.append((self.startIndex,self.startTradeStep,self.startClose,self.startAsk,self.startBid,self.openTradeDir))
        #print(self.lastTenData[-1])
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
        return state , reward , done ,None

        
    def getState(self):
        state = self.data[self.startIndex+self.stepIndex:(self.startIndex+self.stepIndex+100)]
        actions = np.zeros((100,2),dtype=np.float32)
        if self.openTradeDir == 1:
            actions[:,0] = self.openTradeAsk
        if self.openTradeDir == 2:
            actions[:,1] = self.openTradeBid
        
        
        
        state = np.concatenate((state,actions),axis=1)
        state = (state/self.startClose)/2
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


if __name__ == "__main__":
    env = ForexEnv("minutes15_100/data/test_data.csv")
    print(env.reset())
    i = 0
    sumReward = 0
    while i < 500 :
        i+=1
        action = np.random.randint(3)
        _,reward,done,_ = env.step(action)
        if done :
            print ("reward : " , reward)
            sumReward += reward
            env.reset()
            print ("ten last one ",env.lastTenData[-1] )
    
    print ("average reward ",sumReward/i)
            
    


    




        



    

    
        



        
