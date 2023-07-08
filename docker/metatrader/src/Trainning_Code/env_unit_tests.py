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
        lastOpen =state[-14]#[-1,0]
        lastOpenReal = env.data[(env.stepIndex+env.startIndex+100)-1,0]
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

        while i< (((21.0 * 24.0 * 4)+2) * 1) and not done:
            state,reward,done,_ = env.step(0)
            i+=1


        #assert
        expected = (((21.0 * 24.0 * 4) * 1)+1)
        expectedDone = True
        expectedReward = -0.02
        
        if i != expected or done != expectedDone or reward != expectedReward:
            return False,"test200StepsReturnMinus0Point01 : i expected : %.5f found : %.5f , done expected %s found %s , reward expected %.5f , found %.5f"%(expected,i,expectedDone,done,expectedReward,reward)
        else:
            return True,"test200StepsReturnMinus0Point01 : Success"
    except Exception as ex:
        return False,"test200StepsReturnMinus0Point01 : %s"%(str(ex))


def calculateStopLoss(price,direction):
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


def testStopLossWillStopTradeAndReturnNegativeValidReward():
    try:
        #assign
        #global
        state = env.reset()
        state = denormalizeState(state,env.startClose)
        
        
        state,_,done,_ = env.step(1)
        stopLoss = calculateStopLoss(env.openTradeAsk,1)#[-1,1])
        state = denormalizeState(state,env.startClose)
        while state[-11] > stopLoss :#[-1,3]
            state,_,done,_ = env.step(0)
            state = denormalizeState(state,env.startClose)
            if (done == True and state[-11] > stopLoss):#[-1,3]
                return False,"testStopLossWillStopTradeAndReturnNegativeValidReward :   done expected %s found %s "%(False,done)
        
        state,_,done,_ = env.step(0)
        state = denormalizeState(state,env.startClose)
        expectedDone = True
        if expectedDone != done:
            return False,"testStopLossWillStopTradeAndReturnNegativeValidReward :   done expected %s found %s "%(expectedDone,done)


        
        
        return True,"testStopLossWillStopTradeAndReturnNegativeValidReward : Success"
    except Exception as ex:
        return False,"testStopLossWillStopTradeAndReturnNegativeValidReward : %s"%(str(ex))

def denormalizeState(state,startClose):
    state = state + 1
    state = state * startClose
    return state



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
        while i< (((21.0 * 24.0 * 4)+2)*1) and not done:
            state,reward,done,_ = env.step(0)
            if not done:
                beforeDoneState = state
            if i == 5:
                after5stepsState = state
            


            i+=1


        #assert
        expected = ((21.0 * 24.0 * 4) * 1)/((12 * 21.0 * 24.0 * 4 * 1) * 2.0)
        
        value = beforeDoneState[-3]#[-1,11]
        
        expectedAfter5 = 6/((12 * 21.0 * 24.0 * 4 * 1) * 2)
        valueAfter5 = after5stepsState[-3]#[-1,11]


        
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
        ret,msg = testStopLossWillStopTradeAndReturnNegativeValidReward()
        f.write("%r %s\r\n"%(ret,msg))
        print("%r %s\r\n"%(ret,msg))
        ret,msg = testStepIsWrittenInState()
        f.write("%r %s\r\n"%(ret,msg))
        print("%r %s\r\n"%(ret,msg))





    
