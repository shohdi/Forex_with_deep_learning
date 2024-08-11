envSaveObj={
    'env':{
        'states':None, #(done save) #(done load)
        'envName':None, #(done save)#(done load)
        'options':None, #(done save)#(done load)
        'punishAgent':None, #(done save)(done load)
        'stopTrade':None, #(done save) (done load)
        'action_space':None, #(no need to save)(done load)
        'startTradeStep':None, #(done save) (done load)
        'startClose':None, #(done save) (done load)
        'openTradeDir':None, #(done save) (done load)
        'lastTenData':None, #(done save) (done load)
        'reward_queue':None, #(done save) (done load)
        'header':None, #(no need to save) (done load)
        'data':None, #(no need to save) (done load)
        'startAsk':None, #(done save) (done load)
        'startBid':None,#(done save) (done load)
        'openTradeAsk':None,#(done save)(done load)
        'openTradeBid':None,#(done save)(done load)
        'stepIndex':None,#(done save) (done load)
        'stopLoss':None,#(done save) (done load)
        'beforeActionState':None,#(done save) (done load)
        'beforeActionTime':None, #(done save)(done load)
        'observation_space':None, #(no need to save)
        'nextAction':None, #(done save) done load)
        'nextProp':None, #(done save) done load)

    },
    'envData':{
        'actions':None #(done save) (done load)
        ,'stateObj':None #(done save) (done load)
        ,'lastStepRet':{
            'stepState':None #(done save) (done load)
            ,'reward':None #(done save) (done load)
            ,'done':None #(done save) (done load)
            ,'dataItem':None #(done save) (done load)
        }
        ,'states':None #(done save) (done load)
        
    }   

}