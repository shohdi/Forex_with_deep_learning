from lib.env import ForexEnv


#global init   
env = ForexEnv('minutes15_100/data/test_data.csv')


def testStartCloseIsOkAndNotChangesAfterStep():
    try:
        #assign
        #global
        env.reset()
        startClose = env.startClose
        expectedClose = env.data[env.stepIndex + env.startIndex ,1]
        #action
        env.step(0)
        #assert
        if startClose != expectedClose:
            return False,"testStartCloseIsOkAndNotChangesAfterStep : start close expected : %.5f found : %.5f"%(expectedClose,startClose)
        else:
            return True,"testStartCloseIsOkAndNotChangesAfterStep : Success"
    except Exception as ex:
        return False,"testStartCloseIsOkAndNotChangesAfterStep : %s"%(str(ex))








if __name__ == "__main__":
    #run tests
    with open('data/env_unit_tests_result.txt','w') as f:
        
        ret,msg = testStartCloseIsOkAndNotChangesAfterStep()
        f.write("%r %s"%(ret,msg))



    
