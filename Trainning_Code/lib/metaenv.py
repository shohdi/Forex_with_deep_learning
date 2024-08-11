
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


slval = 0.15
tkval = 0.15

class ForexMetaEnv(gym.Env):

    def __init__(self,statesCol,options,envName,punishAgent = True,stopTrade = True):

        self.states = statesCol
        self.envName = envName

        self.options = options
        self.punishAgent = punishAgent
        self.stopTrade = stopTrade
        self.action_space = gym.spaces.Discrete(n=3)
        
        
        self.startTradeStep = None
        self.startClose = None
        self.openTradeDir = 0
        self.lastTenData = collections.deque(maxlen=10)
        self.reward_queue = collections.deque(maxlen=99)
        while len(self.reward_queue) < 99:
            self.reward_queue.append(0.0)
        self.header = ("open","close","high","low","ask","bid")
        self.data = None
        self.startAsk = None
        self.startBid = None
        self.openTradeAsk = None
        self.openTradeBid = None
        self.stepIndex = 0
        self.stopLoss = None
        self.beforeActionState = None
        self.beforeActionTime = None
        self.nextAction = None
        self.nextProp = None
        
        
        
        self.reset()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(99,13), dtype=np.float32)




    



    def resetEnv(self,myState):
        self.startTradeStep = None
        self.stepIndex = 0
        
        self.startClose = myState[0,self.header.index("close")]

        self.openTradeDir = 0
        

        
        self.startAsk = myState[0,self.header.index("ask")]
        self.startBid = myState[0,self.header.index("bid")]
        self.openTradeAsk = None
        self.openTradeBid = None
        self.stopLoss = None
        self.nextAction = None
        self.nextProp = None
        self.reward_queue = collections.deque(maxlen=99)
        while len(self.reward_queue) < 99:
            self.reward_queue.append(0.0)


    def reset(self):
        self.options.takenAction = 12
        self.options.ActionAvailable = True
        if len(self.states) < 99 :
            return None
        myState = np.array(self.states,dtype=np.float32,copy=True)
        self.resetEnv(myState)

        return self.getState(myState)


        
    def normalizeVolume(self,arr):
        volIndex = self.header.index("volume")
        closeIndex = self.header.index("close")
        close = arr[:,closeIndex]
        normalizer = 10000000000.0
        arr[:,volIndex] = (arr[:,volIndex] * close)
        arr[:,volIndex] = arr[:,volIndex]/normalizer
        arr[:, volIndex] = np.where(arr[:, volIndex] > 1, 1, arr[:, volIndex])
        return arr

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
        
        if len(self.states) < 99:
            return None,None,None,None


        #automatic spread :
        beforeActionState = self.beforeActionState
        beforeActionState = self.fixSpreadToBeRandom(beforeActionState)
        myState = np.array(self.states,dtype=np.float32,copy=True)
        myState = self.fixSpreadToBeRandom(myState)
        #check punish
        
        '''
        if self.openTradeDir == 1 and (self.stepIndex - self.startTradeStep) > (100 * 10) and self.stopTrade:
            action_idx = 2
        elif self.openTradeDir == 2 and (self.stepIndex - self.startTradeStep) > (100 * 10) and self.stopTrade:
            action_idx = 1
        
        '''
        

                    
        #end of punish action
        
        reward = 0
        done = False
        #punish no action
        if self.startTradeStep is None:
            if self.stepIndex >= (1 * 10) and self.punishAgent:
                loss = 0.0#-0.00001
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
        #if  self.startTradeStep is None:
        #    if action_idx == 2:
        #        action_idx = 0

        
        
        
        if self.options.tradeDir == 0 and self.openTradeDir != 0  :
            #close trade dir
            self.resetEnv(myState)
            return self.getState(myState),0,False,None

       
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
                tradeStep = self.stepIndex - self.startTradeStep
                if tradeStep > 2:
                    reward = self.closeUpTrade(beforeActionState)
                    done = True

        data=None
		#if self.options.stateObjTimes[-1] != self.beforeActionTime:
        self.stepIndex+=1
                
        if self.stopTrade and not done:
            if self.openTradeDir == 1  :
                tradeStep = self.stepIndex - self.startTradeStep
                if tradeStep > 2:
                    reward = self.closeUpTrade(myState)

	                if reward > 0 and abs(reward * 2.0) >= tkval:
	                    done = True
                    
	                    #print('stop trade!')
	                if reward < 0 and abs(reward * 2.0) >= slval:
	                    done = True
                   
                    #print('stop trade!')
            elif self.openTradeDir == 2 :
                reward = self.closeDownTrade(myState)
                if reward > 0 and abs(reward * 2.0) >= tkval:
                    done = True
                    
                    #print('stop trade!')
                if reward < 0 and abs(reward * 2.0) >= slval:
                    done = True
            if not done:
                reward = 0
                    
                    #print('stop trade!')
        #add current reward :
        #if self.options.stateObjTimes[-1] != self.beforeActionTime:
        if(self.openTradeDir == 1):
            self.reward_queue.append(self.closeUpTrade(myState))
        elif (self.openTradeDir == 2):
            self.reward_queue.append(self.closeDownTrade(myState))
        else:
            self.reward_queue.append(reward)
        #else:
        #    if(self.openTradeDir == 1):
        #        self.reward_queue[-1]=self.closeUpTrade(myState)
        #    elif (self.openTradeDir == 2):
        #        self.reward_queue[-1]=self.closeDownTrade(myState)
        #    else:
        #        self.reward_queue[-1]=reward

        #enf of current reward :
        state = self.getState(myState)
        
        
        return state , reward , done ,data


    def fixSpreadToBeRandom(self,arr):
        bidIndex = self.header.index("bid")
        askIndex = self.header.index("ask")
        closeIndex = self.header.index("close")
        openIndex = self.header.index("open")
        for arrIndex in range(len(arr)):
            row = arr[arrIndex]
			
            op = row[closeIndex]
            if arrIndex < (len(arr)-1):
                op = arr[arrIndex+1,openIndex]
            cl = row[closeIndex]
            #Spread: 0.007 Spread * close = 0.007 * (close) 1.104 = 0.000298 = 0.00030
            point = (0.007 * cl)/30.0
            spread = 30.0 # float(np.random.randint(30,40))
            spread = point * spread
            spread = max(0.01,spread)
            row[bidIndex] = op
            row[askIndex] = row[bidIndex] + spread
        
        return arr
        
    def getState(self,myState):
        
        state = myState[:,:6]
        actions = np.zeros((99,5),dtype=np.float32)
        #sep = np.zeros((99,1),dtype=np.float32)
        
        sltk = np.zeros((99,2),dtype=np.float32)
        sl=0
        tk=0
        if self.openTradeDir == 1:
            actions[:,0] = self.openTradeAsk
            tk = (self.openTradeAsk + (self.startClose * tkval))/(2.0 * self.startClose)
            sl = (self.openTradeAsk - (self.startClose * slval))/(2.0 * self.startClose)
        if self.openTradeDir == 2:
            actions[:,1] = self.openTradeBid
            tk = (self.openTradeBid - (self.startClose * tkval))/(2.0 * self.startClose)
            sl = (self.openTradeBid + (self.startClose * slval))/(2.0 * self.startClose)
        sltk[:,-2] = tk
        sltk[:,-1] = sl
        
        #vol = newState[:,6:7]


        
        
        
        state = np.concatenate((state,actions),axis=1)
        state = (state/(self.startClose*2))
        state[:,-1] = np.array(self.reward_queue,dtype=np.float32,copy=True)
        #state[:,-2] = 0
        state[:,-3] = self.stepIndex/((12 * 21.0 * 24.0 * 4 * 1) * 2.0)
        if self.startTradeStep is not None :
            
            state[:,-2] = (self.stepIndex - self.startTradeStep)/(12 * 21.0 * 24.0 * 4 * 1)
        
        state = np.concatenate((state,sltk),axis=1)
        #state = np.concatenate((state,vol),axis=1)
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
        #print('opening up trade start close : ',self.startClose,' open price ',self.openTradeAsk)

    def openDownTrade(self,myState):
        if self.openTradeDir == 1 or self.openTradeDir == 2:
            return
        self.openTradeDir = 2
        self.openTradeAsk = myState[-1,self.header.index("ask")]
        self.openTradeBid = myState[-1,self.header.index("bid")]
        self.startTradeStep = self.stepIndex
        self.stopLoss = self.calculateStopLoss(self.openTradeBid,2)
        #print('opening down trade start close : ',self.startClose,' open price ',self.openTradeBid)


    def closeUpTrade(self,myState):
        if  self.openTradeDir == 0 or self.openTradeDir == 2:
            return 0.0
        currentBid = myState[-1,self.header.index("bid")]
        #print('closing up trade start close : ',self.startClose,' close price ',currentBid)
        reward = ((currentBid - self.openTradeAsk)/self.startClose)/2.0
        tradeStep = self.stepIndex - self.startTradeStep
        if self.stopTrade and tradeStep > 2:
            currentAsk = myState[-1,self.header.index("ask")]
            currentHigh = myState[-1,self.header.index("high")]
            currentLow = myState[-1,self.header.index("low")]
            spread = (currentAsk/self.startClose) - (currentBid/self.startClose)
            high = currentHigh / self.startClose
            low = currentLow /self.startClose
            tradeAsk = (self.openTradeAsk / self.startClose)
            sl = tradeAsk - slval
            tk = tradeAsk + tkval
            if  (low - spread) <= sl:
                
                reward = (-1 * slval)/2.0    
            elif (high + spread) >= tk:
                
                reward = tkval/2.0
        
            if(reward > (tkval/2.0)):
                reward = tkval/2.0
            elif (reward < ((-1 * slval)/2.0)):
                reward = (-1 * slval)/2.0
        
        return reward
    

    def closeDownTrade(self,myState):
        if  self.openTradeDir == 0 or self.openTradeDir == 1:
            return 0.0
        currentAsk = myState[-1,self.header.index("ask")]
        #print('closing down trade start close : ',self.startClose,' close price ',currentAsk)
        reward =  ((self.openTradeBid - currentAsk)/self.startClose)/2.0
        if self.stopTrade:
            currentBid = myState[-1,self.header.index("bid")]
            currentHigh = myState[-1,self.header.index("high")]
            currentLow = myState[-1,self.header.index("low")]
            spread = (currentAsk/self.startClose) - (currentBid/self.startClose)
            high = currentHigh / self.startClose
            low = currentLow /self.startClose
            tradeBid = (self.openTradeBid / self.startClose)
            sl = tradeBid + slval
            tk = tradeBid - tkval
            if (high - spread) >= sl:
                
                reward = (-1 * slval)/2.0
            elif  (low + spread) <= tk:
                
                reward = tkval /2.0 

            if(reward > (tkval/2.0)):
                reward = tkval/2.0
            elif (reward < ((-1 * slval)/2.0)):
                reward = (-1 * slval)/2.0

        return reward

        

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random((int(time.time()*10000000)%2**31) if seed is None else seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
        #return [int(time.time()*1000000)%2**31,int(time.time()*1000000)%2**31]

