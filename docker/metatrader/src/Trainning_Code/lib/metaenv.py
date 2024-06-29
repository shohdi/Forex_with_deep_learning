
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
import time




class ForexMetaEnv(gym.Env):

    def __init__(self,statesCol,options,punishAgent = True,stopTrade = True):

        self.states = statesCol
        self.options = options
        self.punishAgent = punishAgent
        self.stopTrade = stopTrade
        self.action_space = gym.spaces.Discrete(n=3)
        
        
        self.startTradeStep = None
        self.startClose = None
        self.openTradeDir = 0
        self.lastTenData = collections.deque(maxlen=10)
        self.reward_queue = collections.deque(maxlen=16)
        while len(self.reward_queue) < 16:
            self.reward_queue.append(0.0)
        self.header = ("open","close","high","low","ask","bid")
        self.data = None
        self.startAsk = None
        self.startBid = None
        self.openTradeAsk = None
        self.openTradeBid = None
        self.stepIndex = 0
        self.stopLoss = None
        
        
        
        
        
        test_state = self.reset()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=test_state.shape, dtype=np.float32)

    def wait100(self,is_reset = False) :
        while len(self.states) < 16 :
            self.options.StateAvailable = False
            while not self.options.StateAvailable:
                None
            
            
            while self.options.ActionAvailable:
                None
            
            self.options.takenAction = 0 if not is_reset else "012"
            self.options.ActionAvailable = True

    def waitForTakeAction(self,action):
        while self.options.ActionAvailable:
            None
        if action == 1 and self.openTradeDir == 2:
            action = 12
        if action == 2 and self.openTradeDir == 1:
            action = 12
        self.options.takenAction = action
        self.options.ActionAvailable = True
    
    def waitForNewState(self):
        
        self.options.StateAvailable = False
        while not self.options.StateAvailable:
            None
        
        if self.options.tradeDir == 0 and self.openTradeDir != 0  :
            #close trade dir
            self.openTradeDir = 0
            self.startTradeStep = None
            self.openTradeAsk = None
            self.openTradeBid = None
        myState = np.array(self.states,dtype=np.float32,copy=True)
    
            

        return myState
        

    def reset(self):
        self.options.takenAction = 12
        self.options.ActionAvailable = True
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
        self.stopLoss = None
        self.reward_queue = collections.deque(maxlen=16)
        while len(self.reward_queue) < 16:
            self.reward_queue.append(0.0)
        return self.getState(myState)

    def calculateStopLoss(self,price,direction):
        loss_amount = 0.0085 * price
        '''
        forex_name = "EURUSD"
        price_to_usd = 1.0
        if(price < 1.0):
            forex_name = "USDEUR"
            price_to_usd = 1.0/price
        amount_to_loss = 10.0
        lot_size = 100000
        volume = 0.01
        entry_point = price
        price_in_usd = entry_point * price_to_usd
        volume_lot = volume * lot_size
        volume_lot_price = volume_lot * price_in_usd
        loss_amount = (amount_to_loss * price_in_usd)/volume_lot_price
        loss_amount = loss_amount/price_to_usd
        #print(win_amount)
        #print(loss_amount)
        #buy
        '''
        entry_point = price
        stoploss = entry_point - loss_amount
        
        if direction == 2:
            #sell
            stoploss = entry_point + loss_amount
            
        
        return stoploss

    def step(self,action_idx):
        self.wait100()
        #check punish
        beforeActionState = np.array(self.states,dtype=np.float32,copy=True)
        '''
        if self.openTradeDir == 1 and (self.stepIndex - self.startTradeStep) > (100 * 10) and self.stopTrade:
            action_idx = 2
        elif self.openTradeDir == 2 and (self.stepIndex - self.startTradeStep) > (100 * 10) and self.stopTrade:
            action_idx = 1
        
        '''
        
        stopLossPerc = 0.01
        if self.stopTrade:
            if self.openTradeDir == 1  :
                reward = self.closeUpTrade(beforeActionState)
                if abs(reward * 2.0) >= stopLossPerc:
                    action_idx = 2
                    #print('stop trade!')
            elif self.openTradeDir == 2 :
                reward = self.closeDownTrade(beforeActionState)
                if abs(reward * 2.0) >= stopLossPerc:
                    action_idx = 1
                    #print('stop trade!')
                    
        #end of punish action
        
        reward = 0
        done = False
        #punish no action
        if self.startTradeStep is None:
            if self.stepIndex >= (100 * 10) and self.punishAgent:
                loss = -0.00001
                done=True
                reward = loss
                action_idx = 0
                #close = beforeActionState[-1,self.header.index("close")]
                #close = close/(self.startClose*2.0)
                #action_idx = 1
                
                #if close > 0.5:
                #    action_idx = 2
                #else:
                #    action_idx = 1
        #end of punish no action
        if  self.startTradeStep is None:
            if action_idx == 2:
                action_idx = 0

        self.waitForTakeAction(action_idx)
        
        myState = self.waitForNewState()

       
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
            
            
            #check open trade
            if  self.openTradeDir == 0 :
                #self.openDownTrade(beforeActionState)
                None
            elif self.openTradeDir == 2:
                None
            else : # 1
                #close trade
                reward = self.closeUpTrade(beforeActionState)
                done = True

        data=None
        self.stepIndex+=1
        #add current reward :
        if(self.openTradeDir == 1):
            self.reward_queue.append(self.closeUpTrade(myState))
        elif (self.openTradeDir == 2):
            self.reward_queue.append(self.closeDownTrade(myState))
        else:
            self.reward_queue.append(reward)

        #enf of current reward :
        state = self.getState(myState)
        
        
        return state , reward , done ,data

        
    def getState(self,myState):
        state = myState[:,:6]
        actions = np.zeros((16,5),dtype=np.float32)
        #sep = np.zeros((16,1),dtype=np.float32)
        expectedDoubleReward = 0.01
        sltk = np.zeros((16,2),dtype=np.float32)
        sl=0
        tk=0
        if self.openTradeDir == 1:
            actions[:,0] = self.openTradeAsk
            tk = (self.openTradeAsk + (self.startClose * expectedDoubleReward))/2.0
            sl = (self.openTradeAsk - (self.startClose * expectedDoubleReward))/2.0
        if self.openTradeDir == 2:
            actions[:,1] = self.openTradeBid
            tk = (self.openTradeBid - (self.startClose * expectedDoubleReward))/2.0
            sl = (self.openTradeBid + (self.startClose * expectedDoubleReward))/2.0
        sltk[:,-2] = tk
        sltk[:,-1] = sl
        
        


        
        
        
        state = np.concatenate((state,actions),axis=1)
        state = (state/(self.startClose*2))
        state[:,-1] = np.array(self.reward_queue,dtype=np.float32,copy=True)
        #state[:,-2] = 0
        state[:,-3] = self.stepIndex/((12 * 21.0 * 24.0 * 4 * 1) * 2.0)
        if self.startTradeStep is not None :
            
            state[:,-2] = (self.stepIndex - self.startTradeStep)/(12 * 21.0 * 24.0 * 4 * 1)
        
        state = np.concatenate((state,sltk),axis=1)
        #state = np.concatenate((state,sep),axis=1)
        #state =  np.reshape( state,(-1,))
        return state

    def openUpTrade(self,myState):
        if self.openTradeDir == 1 or self.openTradeDir == 2:
            return
        self.openTradeDir = 1
        self.openTradeAsk = myState[-1,self.header.index("ask")]
        self.openTradeBid = myState[-1,self.header.index("bid")]
        self.startTradeStep = self.stepIndex
        self.stopLoss = self.calculateStopLoss(self.openTradeAsk,1)
        print('opening up trade start close : ',self.startClose,' open price ',self.openTradeAsk)

    def openDownTrade(self,myState):
        if self.openTradeDir == 1 or self.openTradeDir == 2:
            return
        self.openTradeDir = 2
        self.openTradeAsk = myState[-1,self.header.index("ask")]
        self.openTradeBid = myState[-1,self.header.index("bid")]
        self.startTradeStep = self.stepIndex
        self.stopLoss = self.calculateStopLoss(self.openTradeBid,2)
        print('opening down trade start close : ',self.startClose,' open price ',self.openTradeBid)


    def closeUpTrade(self,myState):
        if  self.openTradeDir == 0 or self.openTradeDir == 2:
            return
        currentBid = myState[-1,self.header.index("bid")]
        #print('closing up trade start close : ',self.startClose,' close price ',currentBid)
        return ((currentBid - self.openTradeAsk)/self.startClose)/2.0

    def closeDownTrade(self,myState):
        if  self.openTradeDir == 0 or self.openTradeDir == 1:
            return
        currentAsk = myState[-1,self.header.index("ask")]
        #print('closing down trade start close : ',self.startClose,' close price ',currentAsk)
        return ((self.openTradeBid - currentAsk)/self.startClose)/2.0


        

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random((int(time.time()*10000000)%2**31) if seed is None else seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
        #return [int(time.time()*1000000)%2**31,int(time.time()*1000000)%2**31]

