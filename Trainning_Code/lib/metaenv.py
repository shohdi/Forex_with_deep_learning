
import gym
import gym.spaces
from gym.utils import seeding
import collections
import numpy as np




class ForexMetaEnv(gym.Env):

    def __init__(self,statesCol,options,punishAgent = True):
        self.states = statesCol
        self.options = options
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

    def wait100(self) :
        while len(self.states) < 100 :
            while not self.options.StateAvailable:
                None
            
            self.options.StateAvailable = False
            while self.options.ActionAvailable:
                None
            self.options.takenAction = 0
            self.options.ActionAvailable = True

    def waitForTakeAction(self,action):
        while self.options.ActionAvailable:
            None
        self.options.takenAction = action
        self.options.ActionAvailable = True
    
    def waitForNewState(self):
        while not self.options.StateAvailable:
            None
        
        myState = np.array(self.states,dtype=np.float32,copy=True)
        self.options.StateAvailable = False

        return myState
        

    def reset(self):
        self.wait100()

        myState = self.waitForNewState()


        self.startTradeStep = None
        self.stepIndex = 0
        
        self.startClose = myState[0,self.header.index("close")]

        self.openTradeDir = 0
        

        
        self.startAsk = myState[0,self.header.index("ask")]
        self.startBid = myState[0,self.header.index("bid")]
        self.openTradeAsk = None
        self.openTradeBid = None

        return self.getState(myState)


    def step(self,action_idx):
        self.wait100()
        beforeActionState = np.array(self.states,dtype=np.float32,copy=True)
        self.waitForTakeAction(action_idx)
        
        myState = self.waitForNewState()
        reward = 0
        done = False
        if action_idx == 0:
            None
        elif action_idx == 1:
            #check open trade
            if self.openTradeDir == 0 :
                self.openUpTrade(beforeActionState)
            elif self.openTradeDir == 1:
                None
            else :
                #close trade
                reward = self.closeDownTrade(beforeActionState)
                done = True
        else :#2 :
            if action_idx == 0:
                None
            elif action_idx == 2:
                #check open trade
                if  self.openTradeDir == 0 :
                    self.openDownTrade(beforeActionState)
                elif self.openTradeDir == 2:
                    None
                else : # 1
                    #close trade
                    reward = self.closeUpTrade(beforeActionState)
                    done = True

        

        state = self.getState(myState)
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

        
    def getState(self,myState):
        state = myState
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

    def openUpTrade(self,myState):
        if self.openTradeDir == 1 or self.openTradeDir == 2:
            return
        self.openTradeDir = 1
        self.openTradeAsk = myState[-1,self.header.index("ask")]
        self.openTradeBid = myState[-1,self.header.index("bid")]
        self.startTradeStep = self.stepIndex

    def openDownTrade(self,myState):
        if self.openTradeDir == 1 or self.openTradeDir == 2:
            return
        self.openTradeDir = 2
        self.openTradeAsk = myState[-1,self.header.index("ask")]
        self.openTradeBid = myState[-1,self.header.index("bid")]
        self.startTradeStep = self.stepIndex


    def closeUpTrade(self,myState):
        if  self.openTradeDir == 0 or self.openTradeDir == 2:
            return
        currentBid = myState[-1,self.header.index("bid")]
        return ((currentBid - self.openTradeAsk)/self.startClose)/2

    def closeDownTrade(self,myState):
        if  self.openTradeDir == 0 or self.openTradeDir == 1:
            return
        currentAsk = myState[-1,self.header.index("ask")]
        return ((self.openTradeBid - currentAsk)/self.startClose)/2


        

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
