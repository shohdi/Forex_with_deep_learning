import numpy as np
import csv
import requests



if __name__ == "__main__":
    data = None
    header = None

    with open('minutes15_100/data/test_data.csv', 'r') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        data = np.array(list(reader)).astype(np.float32)
    urlFormat = 'http://127.0.0.1:5000?open={open}&close={close}&high={high}&low={low}&ask={ask}&bid={bid}'
    for i in range(len(data)):
        url = urlFormat.format(
            open=data[i,header.index("open")]
            ,close=data[i,header.index("close")]
            ,high=data[i,header.index("high")]
            ,low=data[i,header.index("low")]
            ,ask=data[i,header.index("ask")]
            ,bid=data[i,header.index("bid")]
            )
        
        action = int(requests.get(url))
        print('action : ' , action)
        

        
        
    
    

