from ssl import ALERT_DESCRIPTION_INSUFFICIENT_SECURITY
import gym
import gym.spaces
from gym.utils import seeding
import collections
import glob
import os
import numpy as np
try:
    testVar = np.zeros((3,3),dtype=np.bool)
except :
    np.bool = bool
    testVar = np.zeros((3,3),dtype=np.bool)
import csv
import time

slval = 0.15
tkval = 0.15

class ForexEnv(gym.Env):
    def __init__(self,filePath , haveOppsiteData:bool , punishAgent = True,stopTrade = True,startRandom=True):
        self.haveOppsiteData = haveOppsiteData
        self.punishAgent = punishAgent
        self.stopTrade = stopTrade
        self.filePath = filePath
        self.startRandom=startRandom
        self.action_space = gym.spaces.Discrete(n=3)
        
        
        self.startTradeStep = None
        self.startClose = None
        self.openTradeDir = 0
        self.lastTenData = collections.deque(maxlen=10)
        self.reward_queue = collections.deque(maxlen=99)
        while len(self.reward_queue) < 99:
            self.reward_queue.append(0.0)
        self.header = None
        self.data_arr = []
        self.data = None
        self.startAsk = None
        self.startBid = None
        self.openTradeAsk = None
        self.openTradeBid = None
        self.startIndex = 0
        self.stepIndex = 0
        self.stopLoss = None
        self.filesArr = []
        self.filesArr = self.readFilePaths(self.filePath)
        for fileIndex in range(len(self.filesArr)):
            currentFilePath = self.filesArr[fileIndex]
            with open(currentFilePath, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                self.header = next(reader)
                arrToppend = np.array(list(reader)).astype(np.float32)
                arrToppend = self.fixSpreadToBeRandom(arrToppend)
                #arrToppend = self.normalizeVolume(arrToppend)
                self.data_arr.append(arrToppend )
                if self.haveOppsiteData:
                    arrToppend = np.array(self.data_arr[len(self.data_arr)-1],copy=True)
                    self.data_arr.append(arrToppend)
                    self.data_arr[len(self.data_arr)-1] = 1/self.data_arr[len(self.data_arr)-1]
                    #self.data_arr[len(self.data_arr)-1][:,6] = 1/self.data_arr[len(self.data_arr)-1][:,6]
                    tempData = np.array(self.data_arr[len(self.data_arr)-1][:,4],copy=True)
                    self.data_arr[len(self.data_arr)-1][:,4] = np.array(self.data_arr[len(self.data_arr)-1][:,5],copy=True)
                    self.data_arr[len(self.data_arr)-1][:,5] = tempData
                    tempData = np.array(self.data_arr[len(self.data_arr)-1][:,2],copy=True)
                    self.data_arr[len(self.data_arr)-1][:,2] = np.array(self.data_arr[len(self.data_arr)-1][:,3],copy=True)
                    self.data_arr[len(self.data_arr)-1][:,3] = tempData

                self.data = self.data_arr[np.random.randint(len(self.data_arr))]
        
        test_state = self.reset()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=test_state.shape, dtype=np.float32)

    def readFilePaths(self,filePath):
        arrRet = []
        arrRet = glob.glob(filePath + os.sep +  r"*.csv")
        
        return arrRet
    
    def normalizeVolume(self,arr):
        volIndex = self.header.index("volume")
        closeIndex = self.header.index("close")
        close = arr[:,closeIndex]
        normalizer = 10000000000.0
        arr[:,volIndex] = (arr[:,volIndex] * close)
        arr[:,volIndex] = arr[:,volIndex]/normalizer
        arr[:, volIndex] = np.where(arr[:, volIndex] > 1, 1, arr[:, volIndex])
        return arr


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
            spread = float(np.random.randint(30,40))
            spread = point * spread
            spread = max(0.01,spread)
            row[bidIndex] = op
            row[askIndex] = row[bidIndex] + spread
        
        return arr


    def reset(self):
        self.lastTenData.append((self.startIndex,self.startTradeStep,self.startClose,self.startAsk,self.startBid,self.openTradeDir))
        #print(self.lastTenData[-1])
        self.data = self.data_arr[np.random.randint(len(self.data_arr))]
        self.startIndex = (self.startIndex + self.stepIndex+1)%(len(self.data)-(99 * 2))
        if self.startRandom:
            self.startIndex =np.random.randint(len(self.data)-(99 * 2))
        self.startTradeStep = None
        self.stepIndex = 0
        self.startClose = self.data[self.startIndex+ self.stepIndex][self.header.index("close")]

        self.openTradeDir = 0
        
        
        self.startAsk = self.data[self.startIndex+ self.stepIndex,self.header.index("ask")]
        self.startBid = self.data[self.startIndex+ self.stepIndex,self.header.index("bid")]
        self.openTradeAsk = None
        self.openTradeBid = None
        self.stopLoss = None
        self.reward_queue = collections.deque(maxlen=99)
        while len(self.reward_queue) < 99:
            self.reward_queue.append(0.0)
        return self.getState()
    
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
        #check punish
        '''
        if self.openTradeDir == 1 and (self.stepIndex - self.startTradeStep) > (100 * 10) and self.stopTrade:
            action_idx = 2
        elif self.openTradeDir == 2 and (self.stepIndex - self.startTradeStep) > (100 * 10) and self.stopTrade:
            action_idx = 1
        '''

        

        
        #end of punish action

        #punish no action
        reward = 0
        done = False
        if self.startTradeStep is None:
            if self.stepIndex >= (1 * 10) and self.punishAgent:
                loss = 0.0#-0.00001
                done=True
                reward = loss
                action_idx = 0
                #close = self.data[self.startIndex+self.stepIndex+99,self.header.index("close")]
                #close = close/(self.startClose*2.0)
                #action_idx = 1
                #print('open opposite trade as punish!')
                #if close > 0.5:
                #    action_idx = 2
                #    print('open opposite trade as punish!')
                #else:
                #    action_idx = 1
                #    print('open opposite trade as punish!')
        #end of punish no action

        
        
        if action_idx == 0:
            None
        elif action_idx == 1:
            #check open trade
            if self.openTradeDir == 0 :
                self.openUpTrade()
                #print('open up trade!')
            elif self.openTradeDir == 1:
                None
            else :
                #close trade
                reward = self.closeDownTrade()
                done = True
        else :#2 :
            
            
            #check open trade
            if  self.openTradeDir == 0 :
                #self.openDownTrade()
                None
                
            elif self.openTradeDir == 2:
                None
            else : # 1
                #close trade
                tradeStep = self.stepIndex - self.startTradeStep
                if tradeStep > 2:
                    reward = self.closeUpTrade()
                    done = True
        data=None
        if (self.stepIndex + self.startIndex + 99) >= (len(self.data) - 5) and not done:
            if self.openTradeDir == 1 :
                reward = self.closeUpTrade()
                #print('end of data!')
            elif self.openTradeDir == 2 :
                reward = self.closeDownTrade()
                #print('end of data!')
            else:
                reward = reward
            done = True
            data = True
        
        self.stepIndex+=1
        
        if self.stopTrade and not done:
            if self.openTradeDir == 1  :
                tradeStep = self.stepIndex - self.startTradeStep
                if tradeStep > 2:
                    reward = self.closeUpTrade()

                    if reward > 0 and abs(reward * 2.0) >= tkval:
                        done = True
                        
                        #print('stop trade!')
                    if reward < 0 and abs(reward * 2.0) >= slval:
                        done = True
                    
                        #print('stop trade!')
            elif self.openTradeDir == 2 :
                reward = self.closeDownTrade()
                if reward > 0 and abs(reward * 2.0) >= tkval:
                    done = True
                    
                    #print('stop trade!')
                if reward < 0 and abs(reward * 2.0) >= slval:
                    done = True
            if not done:
                reward = 0
                    
                    #print('stop trade!')
        #add current reward :
        if(self.openTradeDir == 1):
            self.reward_queue.append(self.closeUpTrade())
        elif (self.openTradeDir == 2):
            self.reward_queue.append(self.closeDownTrade())
        else:
            self.reward_queue.append(reward)

        #enf of current reward :
        state = self.getState()
        
        
        return state , reward , done ,data

    def getRawState(self):
        state = self.data[self.startIndex+self.stepIndex:(self.startIndex+self.stepIndex+99)]
        return state

    def getState(self):
        state = self.getRawState()[:,:6]
       

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
        
        

        #vol = self.getRawState()[:,6:7]
        
        
        
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

    def openUpTrade(self):
        if self.openTradeDir == 1 or self.openTradeDir == 2:
            return
        self.openTradeDir = 1
        self.openTradeAsk = self.data[self.startIndex+self.stepIndex+98,self.header.index("ask")]
        self.openTradeBid = self.data[self.startIndex+self.stepIndex+98,self.header.index("bid")]
        self.startTradeStep = self.stepIndex
        self.stopLoss = self.calculateStopLoss(self.openTradeAsk,1)

    def openDownTrade(self):
        #print('open down trade!')
        if self.openTradeDir == 1 or self.openTradeDir == 2:
            return
        self.openTradeDir = 2
        self.openTradeAsk = self.data[self.startIndex+self.stepIndex+98,self.header.index("ask")]
        self.openTradeBid = self.data[self.startIndex+self.stepIndex+98,self.header.index("bid")]
        self.startTradeStep = self.stepIndex
        self.stopLoss = self.calculateStopLoss(self.openTradeBid,2)

    def closeUpTrade(self):
        if  self.openTradeDir == 0 or self.openTradeDir == 2:
            return 0.0
        currentBid = self.data[self.startIndex+self.stepIndex+98,self.header.index("bid")]
        reward =  ((currentBid - self.openTradeAsk)/self.startClose)/2.0
        tradeStep = self.stepIndex - self.startTradeStep
        if self.stopTrade and tradeStep > 2:
            currentAsk = self.data[self.startIndex+self.stepIndex+98,self.header.index("ask")]
            currentHigh = self.data[self.startIndex+self.stepIndex+98,self.header.index("high")]
            currentLow = self.data[self.startIndex+self.stepIndex+98,self.header.index("low")]
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

    def closeDownTrade(self):
        if  self.openTradeDir == 0 or self.openTradeDir == 1:
            return 0.0
        currentAsk = self.data[self.startIndex+self.stepIndex+98,self.header.index("ask")]
        reward =  ((self.openTradeBid - currentAsk)/self.startClose)/2.0
        if self.stopTrade:
            currentBid = self.data[self.startIndex+self.stepIndex+98,self.header.index("bid")]
            currentHigh = self.data[self.startIndex+self.stepIndex+98,self.header.index("high")]
            currentLow = self.data[self.startIndex+self.stepIndex+98,self.header.index("low")]
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

    def analysisUpTrade(self):
        startStep = self.startIndex + self.stepIndex
        currentStep = startStep
        startAsk = self.data[startStep+99,self.header.index("ask")]
        currentBid = self.data[currentStep+99,self.header.index("bid")]
        diff = (startAsk - currentBid)
        while (( startAsk - currentBid) < (2*diff) and (currentBid - startAsk) < (diff) and currentStep < (len(self.data)-(500 * 1)) and (currentStep - startStep) < (200 * 1) ):
            currentStep += 1
            currentBid = self.data[currentStep+99,self.header.index("bid")]
        if currentStep == (len(self.data)-(500 * 1)) or ((currentStep - startStep) >= (200 * 1)) :
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
        while (( currentAsk - startBid) < (2*diff) and (startBid - currentAsk) < (diff) and currentStep < (len(self.data)-(500 * 1)) and (currentStep - startStep) < (200 * 1)):
            currentStep += 1
            currentAsk = self.data[currentStep+99,self.header.index("ask")]
        if currentStep == (len(self.data)-(500 * 1)) or ((currentStep - startStep) >= (200 * 1)):
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
    env = ForexEnv("minutes15_100/data/val")
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
    env = ForexEnv("minutes15_100/data/train",True)
    state = env.reset()
    print("start " , state[0])
    print("start close ",env.startClose)
    
    done = False
    while not done :
        print("last ",state[-1])
        action = int(input("action 0,1,2 : "))
        state,reward,done,_ = env.step(action)
        print('reward ',reward)
    





            
    


    




        



    

    
        



        
