from YahooRet import Adjclose,Pre,Post,Regular,CurrentTradingPeriod,Meta,Quote,Indicators,Result,Chart,YahooRet
import requests
import json


cookies = {
    'A3': 'd=AQABBKsNoWUCECgioJOFF67ibh00IH6N1ggFEgEBAQFfomWqZarHzyMA_eMAAA&S=AQAAAkG4P9ZfT0Fk8OajOOXa_4s',
    'A1': 'd=AQABBKsNoWUCECgioJOFF67ibh00IH6N1ggFEgEBAQFfomWqZarHzyMA_eMAAA&S=AQAAAkG4P9ZfT0Fk8OajOOXa_4s',
    'A1S': 'd=AQABBKsNoWUCECgioJOFF67ibh00IH6N1ggFEgEBAQFfomWqZarHzyMA_eMAAA&S=AQAAAkG4P9ZfT0Fk8OajOOXa_4s',
}

headers = {
    'authority': 'query1.finance.yahoo.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'no-cache',
    # 'cookie': 'A3=d=AQABBKsNoWUCECgioJOFF67ibh00IH6N1ggFEgEBAQFfomWqZarHzyMA_eMAAA&S=AQAAAkG4P9ZfT0Fk8OajOOXa_4s; A1=d=AQABBKsNoWUCECgioJOFF67ibh00IH6N1ggFEgEBAQFfomWqZarHzyMA_eMAAA&S=AQAAAkG4P9ZfT0Fk8OajOOXa_4s; A1S=d=AQABBKsNoWUCECgioJOFF67ibh00IH6N1ggFEgEBAQFfomWqZarHzyMA_eMAAA&S=AQAAAkG4P9ZfT0Fk8OajOOXa_4s',
    'pragma': 'no-cache',
    'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'none',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

params = {
    'symbol': '1050.SR',
    'period1': '1705579200',
    'period2': '1705579200',
    'useYfid': 'true',
    'interval': '1d',
    'includePrePost': 'true',
    'events': 'div|split|earn',
    'lang': 'en-US',
    'region': 'US',
    'crumb': 'JuMcKZM9MFA',
    'corsDomain': 'finance.yahoo.com',
}

class YahooRequest:
    def __init__ (self,symbol,fr,to):
        self.fr = fr
        self.to = to
        self.symbol = symbol
    
    def makeRequest(self):
        url = 'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'.format(symbol=self.symbol)
        params['symbol'] = self.symbol
        params['period1'] = self.fr
        params['period2'] = self.to
        response = requests.get(
            url,
            params=params,
            cookies=cookies,
            headers=headers,
        )
        stringRet = str(response.text.strip())
        jsonObj = json.loads(stringRet)
        yahooObj= YahooRet.from_dict(jsonObj)
        
        return yahooObj

