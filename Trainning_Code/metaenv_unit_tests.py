from os import stat
from lib.metaenv import ForexMetaEnv
from metarun import headers,options,stateObj,MetaTrade
from threading import Thread
from flask import Flask
from flask_restful import Resource, Api,reqparse


#global init
   
env = None 


def testStartCloseIsOkAndNotChangesAfterStep():
    try:
        #assign
        #global
        env.reset()
        
        expectedClose = env.states[env.stepIndex][1]
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
        lastOpen =state[-14]#[-1,0]
        lastOpenReal = env.states[-1][0]
        expected = (lastOpenReal/startClose)-1
        if lastOpen != expected:
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

        while i< (202 * 15) and not done:
            state,reward,done,_ = env.step(0)
            i+=1


        #assert
        expected = ((200 * 15)+1)
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
        while i< (202*15) and not done:
            state,reward,done,_ = env.step(0)
            if not done:
                beforeDoneState = state
            


            i+=1


        #assert
        expected = ((200 * 15)+1)
        expectedDone = True
        bid = beforeDoneState[-9]#[-1,5]
        openTradeAsk = beforeDoneState[-5]#[-1,9]
        expectedReward = str(round( bid-openTradeAsk,5))
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
        while i< (202*15) and not done:
            state,reward,done,_ = env.step(0)
            if not done:
                beforeDoneState = state
            if i == 5:
                after5stepsState = state
            


            i+=1


        #assert
        expected = (200 * 15)/(200 * 15 * 2)
        
        value = beforeDoneState[-3]#[-1,8]
        
        expectedAfter5 = 6/(200 * 15 * 2)
        valueAfter5 = after5stepsState[-3]#[-1,8]


        
        if value != expected  :
            return False,"testStepIsWrittenInState : step index expected : %.5f found : %.5f "%(expected,value)
        
        if str(round(valueAfter5,5)) != str(round(expectedAfter5,5)):
            return False,"testStepIsWrittenInState : step index 6 expected : %.5f found : %.5f "%(expectedAfter5,valueAfter5)


        return True,"testStepIsWrittenInState : Success"
    except Exception as ex:
        return False,"testStepIsWrittenInState : %s"%(str(ex))






def runTests():
    global env 
    env = ForexMetaEnv(stateObj,options)
    #run tests
    with open('data/metaenv_unit_tests_result.txt','w') as f:
        
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



if __name__ == "__main__":
    thread = Thread(target=runTests)
    thread.start()
    
    #start server
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(MetaTrade, '/')
    app.run()






    
