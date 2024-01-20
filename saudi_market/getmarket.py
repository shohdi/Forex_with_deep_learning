from SaudiMarket import SaudiMarket,Row
from YahooRet import Adjclose,Pre,Post,Regular,CurrentTradingPeriod,Meta,Quote,Indicators,Result,Chart,YahooRet
import requests
import json


if __name__=='__main__':
    url = 'https://www.mubasher.info/api/1/listed-companies?country=sa&size=290&start=1'
    stringRet = str(requests.get(url).text.strip())
    jsonObj = json.loads(stringRet)
    market = SaudiMarket.from_dict(jsonObj)
    for i in range(len(market.rows)):
        sym = market.rows[i].symbol
        print(sym)
        sym = sym[0:sym.index(".")] + ".SR"
        print(sym)

    