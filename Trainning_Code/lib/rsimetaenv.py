
import gym
import gym.spaces
from gym.utils import seeding
import collections
import numpy as np
import time
from lib.IqOptionTrade import IqOptionTrade
import sys
from lib.metaenv import ForexMetaEnv




class RsiMetaEnv(ForexMetaEnv):

    def __init__(self,statesCol,options,punishAgent = True,stopTrade = True):
        super(RsiMetaEnv, self).__init__(statesCol,options,punishAgent ,stopTrade )
        self.old_actions = collections.deque([],4)




    def step(self,action_idx):
        self.wait100()
        
        action_idx = 0
        beforeActionState = np.array(self.states,dtype=np.float32,copy=True)
        self.waitForTakeAction(action_idx)
        
        myState = self.waitForNewState()
        
        noTrade = True
        #for i in range(len(self.old_actions)):
        #    if self.old_actions[i] != 0:
        #        noTrade = False
        
        if self.options.tradeDir == 1 and noTrade:
            #open trade up
            action_idx = 1
            self.iq.doUpTrade()
            print("open up order ......")
        
        if self.options.tradeDir == 2 and noTrade:
            action_idx = 2
            self.iq.doDownTrade()
            print("open down order ......")

        self.old_actions.append(action_idx)

        reward = 0
        done = False
        

        
        self.stepIndex+=1
        state = self.getState(myState)
        
        
        
        return state , reward , done ,None

        
