from os import stat
from lib.env import ForexEnv


#global init   
env = ForexEnv('minutes15_100/data/test_data.csv',True)


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
        lastOpen =state[-1,0]#[-14]
        lastOpenReal = env.data[(env.stepIndex+env.startIndex+100)-1,0]
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
        state,reward,_,_ = env.step(0)
        state,reward,_,_ = env.step(0)
        state,reward,_,_ = env.step(0)
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
        
        
        #action
        i  =0
        done = False

        while i< (((100)+2) * 1) and not done:
            state,reward,done,_ = env.step(0)
            i+=1


        #assert
        expected = (((100) * 1)+1)
        expectedDone = True
        expectedReward = -0.02
        
        if i != expected or done != expectedDone or reward != expectedReward:
            return False,"test200StepsReturnMinus0Point01 : i expected : %.5f found : %.5f , done expected %s found %s , reward expected %.5f , found %.5f"%(expected,i,expectedDone,done,expectedReward,reward)
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
        env.step(1)
        beforeDoneState = None
        while i< (((100)+2)*1) and not done:
            state,reward,done,_ = env.step(0)
            if not done:
                beforeDoneState = state
            


            i+=1


        #assert
        expected = ((100 * 1)+1)
        expectedDone = True
        bid = beforeDoneState[-1,5]#[-9]
        openTradeAsk = beforeDoneState[-1,9]#[-5]
        expectedReward = str(round( (bid*2)-(openTradeAsk*2),5))
        reward = str(round(reward,5))

        
        if i != expected or done != expectedDone or reward != expectedReward:
            return False,"test200StepsAfterTradeIsOkAndReturnRealReward : i expected : %.5f found : %.5f , done expected %s found %s , reward expected %s , found %s"%(expected,i,expectedDone,done,expectedReward,reward)
        else:
            return True,"test200StepsAfterTradeIsOkAndReturnRealReward : Success"
    except Exception as ex:
        return False,"test200StepsAfterTradeIsOkAndReturnRealReward : %s"%(str(ex))



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
        while i< (((100)+2)*1) and not done:
            state,reward,done,_ = env.step(0)
            if not done:
                beforeDoneState = state
            if i == 5:
                after5stepsState = state
            


            i+=1


        #assert
        expected = ((100) * 1)/((12 * 21.0 * 24.0 * 4 * 1) * 2.0)
        
        value = beforeDoneState[-1,11]#[-3]
        
        expectedAfter5 = 6/((12 * 21.0 * 24.0 * 4 * 1) * 2)
        valueAfter5 = after5stepsState[-1,11]#[-3]


        
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
        ret,msg = test200StepsAfterTradeIsOkAndReturnRealReward()
        f.write("%r %s\r\n"%(ret,msg))
        print("%r %s\r\n"%(ret,msg))
        ret,msg = testStepIsWrittenInState()
        f.write("%r %s\r\n"%(ret,msg))
        print("%r %s\r\n"%(ret,msg))





    
