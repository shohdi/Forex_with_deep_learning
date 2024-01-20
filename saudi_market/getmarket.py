from SaudiMarket import SaudiMarket,Row
from YahooRet import Adjclose,Pre,Post,Regular,CurrentTradingPeriod,Meta,Quote,Indicators,Result,Chart,YahooRet
from YahooRequest import YahooRequest
import requests
import json


def writeDataToFile(dirPath,sym, start,end):
    req = YahooRequest(sym,start,end)
    ret = req.makeRequest()
    myMeta = ret.chart.result[0].meta
    quotes = ret.chart.result[0].indicators.quote
    with open(dirPath + '/{symbol}.csv'.format(symbol=sym),'w') as f:
        f.write("open;close;high;low;ask;bid;volume\n")
        for quoteIndex in range(len(quotes)):
            quote = quotes[quoteIndex]
            for dayIndex in range(len(quote.close)):
                
                op = quote.open[dayIndex]
                close = quote.close[dayIndex]
                if(close > 0.0):
                    high = quote.high[dayIndex]
                    low = quote.low[dayIndex]
                    ask = 0
                    bid = 0
                    volume = quote.volume[dayIndex]
                    f.write("{op};{close};{high};{low};{ask};{bid};{volume}\n".format(op=op,close=close,high=high,low=low,ask=ask,bid=bid,volume=volume))

if __name__=='__main__':
    url = 'https://www.mubasher.info/api/1/listed-companies?country=sa&size=500&start=1'
    stringRet = str(requests.get(url).text.strip())
    jsonObj = json.loads(stringRet)
    market = SaudiMarket.from_dict(jsonObj)
    for i in range(len(market.rows)):
        try :
        
            sym = str(market.rows[i].symbol)
            #print(sym)
            if(sym.find('.') >= 0):
                sym = sym[0:sym.index(".")]
            sym = sym  + ".SR"
            #print(sym)
            fr = 1705580381
            to = 1705580381
            req = YahooRequest(sym,fr,to)
            ret = req.makeRequest()
            myMeta = ret.chart.result[0].meta
            step = 86400
            noOfDays = (myMeta.regularMarketTime - myMeta.firstTradeDate)/step
            valDays = int(noOfDays * 0.1)
            
            valEnd = step * valDays
            valStart = myMeta.firstTradeDate
            valEnd = valStart + valEnd
            
            trainStart = valEnd
            trainEnd = myMeta.regularMarketTime
            if valDays > 364:
                writeDataToFile('minutes15_100/data/val',sym, valStart,valEnd)
                writeDataToFile('minutes15_100/data/train',sym,trainStart, trainEnd)
            



        except Exception as ex:
            print(ex)



        

    