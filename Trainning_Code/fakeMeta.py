import time
import numpy as np
try:
    testVar = np.zeros((3,3),dtype=np.bool)
except :
    np.bool = bool
    testVar = np.zeros((3,3),dtype=np.bool)
import csv
import requests



if __name__ == "__main__":
    data = None
    header = None
    tradeDir = 0
    with open('minutes15_100/data/val/EurUSD_val.csv', 'r') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        data = np.array(list(reader)).astype(np.float32)
    urlFormat = 'http://127.0.0.1:5000?open={open}&close={close}&high={high}&low={low}&ask={ask}&bid={bid}&volume={volume}&tradeDir={tradeDir}&env=test&time={time}'
    for i in range(len(data)):
        url = urlFormat.format(
            open=data[i,header.index("open")]
            ,close=data[i,header.index("close")]
            ,high=data[i,header.index("high")]
            ,low=data[i,header.index("low")]
            ,ask=data[i,header.index("ask")]
            ,bid=data[i,header.index("bid")]
            ,volume=0
            ,tradeDir=tradeDir
            ,time=time.time()
            )
        
        action = int(requests.get(url).text.strip().replace('"','').replace('\r','').replace('\n',''))
        if (i % 100) == 0:
            print('i : ' , i , 'action : ' , action)
        if action == 1 or action == 2:
            #if np.random.randint(2) == 1:
            tradeDir = action
            #else:
            #tradeDir = 0
        elif action == 12:
            tradeDir = 0

        
        
    
    

