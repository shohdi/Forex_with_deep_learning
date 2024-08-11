
from lib.env import ForexEnv


#global init   
env = ForexEnv('minutes15_100/data/val',True,True,True,False)

slval = 0.15
tkval = 0.15
def testSlTkForBuyIsOk():
    try:
        #assign
        env.reset()

        #action
        expectedDone = False
        state,_,_,_ = env.step(0)
        state,reward,done,data = env.step(1)
        expectedDone,expectedReward = getTkSlExDone(state)
        

        

        while not expectedDone:
            state,reward,done,data = env.step(1)
            expectedDone,expectedReward = getTkSlExDone(state)
        


        #assert
        
        
        assert (expectedDone == done and reward == expectedReward),'touched tk or sl but not done , done : %s , expectedDone : %s , expectedReward : %0.6f , reward : %0.6f'%(str(done),str(expectedDone),expectedReward,reward)
       

        return True,"testSlTkForBuyIsOk : Success"
    except Exception as ex:
        return False,"testSlTkForBuyIsOk : %s"%(str(ex))
    

def testNoCloseBeforeStep2():
    try:
        #assign
        env.reset()

        #action
        expectedDone = False
        state,_,_,_ = env.step(0)
        state,reward,done,data = env.step(1)
        state,reward,done,data = env.step(2)

        #first assert
        expectedDone = False
        expectedOpenTradeDir = 1
        openTradeDir = env.openTradeDir
        assert (expectedOpenTradeDir == openTradeDir and expectedDone == done),'step %d close trade before two days , done : %s , expectedDone : %s , openTradeDir : %d , expectedOpenTradeDir : %d'%(1,str(done),str(expectedDone),openTradeDir,expectedOpenTradeDir)


        state,reward,done,data = env.step(2)
        #second assert
        expectedDone = False
        expectedOpenTradeDir = 1
        openTradeDir = env.openTradeDir
        assert (expectedOpenTradeDir == openTradeDir and expectedDone == done),'step %d close trade before two days , done : %s , expectedDone : %s , openTradeDir : %d , expectedOpenTradeDir : %d'%(2,str(done),str(expectedDone),openTradeDir,expectedOpenTradeDir)

        state,reward,done,data = env.step(2)
        #third assert
        expectedDone = True
        assert ( expectedDone == done),'step %d close trade before two days , done : %s , expectedDone : %s , openTradeDir : %d , expectedOpenTradeDir : %d'%(3,str(done),str(expectedDone))


        return True,"testNoCloseBeforeStep2 : Success"
    except Exception as ex:
        return False,"testNoCloseBeforeStep2 : %s"%(str(ex))
    
def testSlTkForSellIsOk():
    try:
        #assign
        env.reset()

        #action
        expectedDone = False
        state,_,_,_ = env.step(0)
        state,reward,done,data = env.step(2)
        expectedDone,expectedReward = getTkSlExDoneForSell(state)
        

        

        while not expectedDone:
            state,reward,done,data = env.step(2)
            expectedDone,expectedReward = getTkSlExDoneForSell(state)
        


        #assert
        
        
        assert (expectedDone == done and reward == expectedReward),'touched tk or sl but not done , done : %s , expectedDone : %s , expectedReward : %0.6f , reward : %0.6f'%(str(done),str(expectedDone),expectedReward,reward)
       

        return True,"testSlTkForSellIsOk : Success"
    except Exception as ex:
        return False,"testSlTkForSellIsOk : %s"%(str(ex))

def getTkSlExDone(state):
    expectedDone = False
    expectedReward = 0
    high = state[-1,2] * 2.0
    low = state[-1,3] * 2.0
    ask = state[-1,4] * 2.0
    bid = state[-1,5] * 2.0
    spread = ask-bid
        
    tradeAsk = (env.openTradeAsk / env.startClose)
    sl = tradeAsk - slval
    tk = tradeAsk + tkval
    if  (low - spread) <= sl:
        expectedDone = True
        expectedReward = (-1 * slval)/2.0    
    if (high + spread) >= tk:
        expectedDone = True
        expectedReward = tkval/2.0
    
    return expectedDone,expectedReward

def getTkSlExDoneForSell(state):
    expectedDone = False
    expectedReward = 0
    high = state[-1,2] * 2.0
    low = state[-1,3] * 2.0
    ask = state[-1,4] * 2.0
    bid = state[-1,5] * 2.0
    spread = ask-bid
    tradeBid = (env.openTradeBid / env.startClose)
    
    sl = tradeBid + slval
    tk = tradeBid - tkval
    if (high - spread) >= sl:
        expectedDone = True
        expectedReward = (-1 * slval)/2.0
    if  (low + spread) <= tk:
        expectedDone = True
        expectedReward = tkval /2.0
    return expectedDone,expectedReward
    

def SellIsNotWorking():
    try:
        #assign
        env.reset()

        #action
        state,_,_,_ = env.step(0)
        state,reward,done,data = env.step(2)

        
        


        #assert
        expectedTradeDir = 0
        foundTradeDir = env.openTradeDir
        
        assert (expectedTradeDir == foundTradeDir ),'expected dir is %d found %d'%(expectedTradeDir,foundTradeDir)
       

        return True,"SellIsNotWorking : Success"
    except Exception as ex:
        return False,"SellIsNotWorking : %s"%(str(ex))



def testStateShape():
    try:
        #assign
        env.reset()

        #action
        state,_,_,_ = env.step(0)

        #assert
        assert state.shape == (99 , 13)  , 'state shape is wrong %s'%(str(state.shape))

        return True,"testStateShape : Success"
    except Exception as ex:
        return False,"testStateShape : %s"%(str(ex))
    

def testSlIsIncluded():
    try:
        #assign
        env.reset()

        #action
        state,_,_,_ = env.step(0)
        state,_,_,_ = env.step(1)
        state,_,_,_ = env.step(1)
        #assert
        
        tk = (env.openTradeAsk + (env.startClose * tkval))/(2.0 * env.startClose)
        sl = (env.openTradeAsk - (env.startClose * slval))/(2.0 * env.startClose)
        

        assert str(round(state[-1,-2],6)) == str(round(tk,6)) and str(round(state[-1,-1],6)) == str(round(sl,6))  , 'expected tk : %0.6f , sl : %0.6f get tk : %0.6f , sl : %0.6f'%(tk,sl,state[-1,-2],state[-1,-1])

        return True,"testSlIsIncluded : Success"
    except Exception as ex:
        return False,"testSlIsIncluded : %s"%(str(ex))
    


def testStartCloseIsOkAndNotChangesAfterStep():
    try:
        #assign
        #global
        env.reset()
        
        expectedClose = env.data[env.stepIndex + env.startIndex ,1]
        #action
        env.step(0)
        startClose = env.startClose
        #assert
        if startClose != expectedClose:
            return False,"testStartCloseIsOkAndNotChangesAfterStep : start close expected : %.5f found : %.5f"%(expectedClose,startClose)
        else:
            return True,"testStartCloseIsOkAndNotChangesAfterStep : Success"
    except Exception as ex:
        return False,"testStartCloseIsOkAndNotChangesAfterStep : %s"%(str(ex))


def testNormalizeIsOk():
    try:
        #assign
        #global
        env.reset()
        startClose = env.startClose
        
        #action
        state,_,_,_ = env.step(0)
        state,_,_,_ = env.step(0)
        #assert
        lastOpen =state[-1,0]#[-12]#[-1,0]#[-14]
        lastOpenReal = env.data[(env.stepIndex+env.startIndex+99)-1,0]
        expected = (lastOpenReal/(startClose*2))
        if "%.5f"%lastOpen != "%.5f"%expected:
            return False,"testNormalizeIsOk : last open expected : %.5f found : %.5f"%(expected,lastOpen)
        else:
            return True,"testNormalizeIsOk : Success"
    except Exception as ex:
        return False,"testNormalizeIsOk : %s"%(str(ex))


def testReturnRewardWithoutDoneIs0():
    try:
        #assign
        #global
        env.reset()
        
        
        #action
        state,reward,_,_ = env.step(1)
        state,reward,_,_ = env.step(0)
        #state,reward,_,_ = env.step(0)
        #state,reward,_,_ = env.step(0)
        #state,reward,_,_ = env.step(0)
        #assert
        expected = 0
        
        if reward != expected:
            return False,"testReturnRewardWithoutDoneIs0 : reward expected : %.5f found : %.5f"%(expected,reward)
        else:
            return True,"testReturnRewardWithoutDoneIs0 : Success"
    except Exception as ex:
        return False,"testReturnRewardWithoutDoneIs0 : %s"%(str(ex))


def test200StepsReturnMinus0Point01():
    try:
        #assign
        #global
        env.reset()
        loss = 0.0 #-0.00001
        
        #action
        i  =0
        done = False

        while i< ((1 * 10)+1) and not done:
            state,reward,done,_ = env.step(0)
            i+=1

        expected = 0
        expectedReward = loss
        #if(state[-1,1] > 0.5):
        #    expected = 2
        
        found = env.openTradeDir
        #assert
        
        
        if  expected != found or reward != expectedReward:
            return False,"test200StepsReturnMinus0Point01 :  open trade direction expected %.5f , found %.5f , expectedReward %.6f reward %.6f "%(expected,found,expectedReward,reward)
        else:
            return True,"test200StepsReturnMinus0Point01 : Success"
    except Exception as ex:
        return False,"test200StepsReturnMinus0Point01 : %s"%(str(ex))


def test200StepsAfterTradeIsOkAndReturnRealReward():
    try:
        #assign
        #global
        env.reset()
        
        
        #action
        i  =0
        done = False
        reward = 0.0
        state,reward,done,data =env.step(1)
        
        rewardState = 0.0
        while i < ((100 * 10)+1) and not done:
            rewardState = state[-2]#[-1,-1]
            state,reward,done,data = env.step(0)
            
            i+=1
        
        expectedDone = True
        expectedReward = rewardState
        assert expectedDone == done , 'wrong done status expected done %s , done %s'%(str(expectedDone),str(done))
        strExpectedReward  = '%.6f'%(expectedReward)
        strReward = '%.6f'%(reward)
        
        
        
        assert ( strExpectedReward == strReward) ,'wrong reward expected reward %s , reward %s'%(strExpectedReward,strReward)
        return True,"test200StepsAfterTradeIsOkAndReturnRealReward : Success"
        
    except Exception as ex:
        return False,"test200StepsAfterTradeIsOkAndReturnRealReward : %s"%(str(ex))

def testRewardIsWrittenWithEachStep():
    try:
        #assign
        #global
        env.reset()
        
        
        #action
        i  =0
        done = False
        state,reward,_,_ = env.step(1)
        state,reward,_,_ = env.step(1)
        state,reward,_,_ = env.step(0)
        #state,reward,_,_ = env.step(0)
        
        #state,reward,_,_ = env.step(0)
        beforeDoneState = state
        
        bid = beforeDoneState[-1,-8]#[-7]#[-1,5]#[-9]
        openTradeAsk = beforeDoneState[-1,-7]#[-6]#[-1,6]#[-5]
        expectedReward = str(round( ((bid*2)-(openTradeAsk*2))/2.0,6))
        reward = str(round( state[-1,-3],6))

        previousReward = str(round( state[-2,-3],6))
        
        if  reward != expectedReward or reward == previousReward:
            return False,"testRewardIsWrittenWithEachStep :  reward expected %s , found %s , previousReward : %s"%(expectedReward,reward,previousReward)
        else:
            return True,"testRewardIsWrittenWithEachStep : Success"
    except Exception as ex:
        return False,"testRewardIsWrittenWithEachStep : %s"%(str(ex))

def testStepIsWrittenInState():
    try:
        #assign
        #global
        env.reset()
        
        
        #action
        i  =0
        done = False
        
        beforeDoneState = None
        after5stepsState = None
        while i< (((1))*10) and not done:
            state,reward,done,_ = env.step(0)
            if not done:
                beforeDoneState = state
            if i == 5:
                after5stepsState = state
            


            i+=1


        #assert
        expected = ((1) * 10)/((12 * 21.0 * 24.0 * 4 * 1) * 2.0)
        
        value = beforeDoneState[-1,-5]#[-4]#[-1,8]#[-3]
        
        expectedAfter5 = 6/((12 * 21.0 * 24.0 * 4 * 1) * 2)
        valueAfter5 = after5stepsState[-1,-5]#[-4]#[-1,8]#[-3]


        
        if "%.5f"%(value) != "%.5f"%(expected) :
            return False,"testStepIsWrittenInState : step index expected : %.5f found : %.5f "%(expected,value)
        
        if "%.5f"%(valueAfter5) != "%.5f"%(expectedAfter5):
            return False,"testStepIsWrittenInState : step index 6 expected : %.5f found : %.5f "%(expectedAfter5,valueAfter5)


        return True,"testStepIsWrittenInState : Success"
    except Exception as ex:
        return False,"testStepIsWrittenInState : %s"%(str(ex))









if __name__ == "__main__":
    #run tests
    with open('data/env_unit_tests_result.txt','w') as f:
        ret,msg = testStateShape()
        f.write("%r %s\r\n"%(ret,msg))
        print("%r %s\r\n"%(ret,msg))
        ret,msg = SellIsNotWorking()
        f.write("%r %s\r\n"%(ret,msg))
        print("%r %s\r\n"%(ret,msg))
        ret,msg = testSlTkForBuyIsOk()
        f.write("%r %s\r\n"%(ret,msg))
        print("%r %s\r\n"%(ret,msg))
        #ret,msg = testSlTkForSellIsOk()
        #f.write("%r %s\r\n"%(ret,msg))
        #print("%r %s\r\n"%(ret,msg))
        ret,msg = testSlIsIncluded()
        f.write("%r %s\r\n"%(ret,msg))
        print("%r %s\r\n"%(ret,msg))
        ret,msg = testStartCloseIsOkAndNotChangesAfterStep()
        f.write("%r %s\r\n"%(ret,msg))
        print("%r %s\r\n"%(ret,msg))
        ret,msg = testNormalizeIsOk()
        f.write("%r %s\r\n"%(ret,msg))
        print("%r %s\r\n"%(ret,msg))
        ret,msg = testReturnRewardWithoutDoneIs0()
        f.write("%r %s\r\n"%(ret,msg))
        print("%r %s\r\n"%(ret,msg))
        ret,msg = test200StepsReturnMinus0Point01()
        f.write("%r %s\r\n"%(ret,msg))
        print("%r %s\r\n"%(ret,msg))
        #ret,msg = test200StepsAfterTradeIsOkAndReturnRealReward()
        #f.write("%r %s\r\n"%(ret,msg))
        #print("%r %s\r\n"%(ret,msg))
        ret,msg = testRewardIsWrittenWithEachStep()
        f.write("%r %s\r\n"%(ret,msg))
        print("%r %s\r\n"%(ret,msg))
        ret,msg = testStepIsWrittenInState()
        f.write("%r %s\r\n"%(ret,msg))
        print("%r %s\r\n"%(ret,msg))
        ret,msg = testNoCloseBeforeStep2()
        f.write("%r %s\r\n"%(ret,msg))
        print("%r %s\r\n"%(ret,msg))




    
