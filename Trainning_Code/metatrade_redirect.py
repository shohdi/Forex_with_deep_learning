

from flask import Flask,redirect
from flask_restful import Resource, Api,reqparse

import argparse

from requests import request







headers = ("open","close","high","low","ask","bid")

class MetaTrade(Resource):

    def get(self):
        
        parser = reqparse.RequestParser()
        parser.add_argument('open', type=float,location='args')
        parser.add_argument('close', type=float,location='args')
        parser.add_argument('high', type=float,location='args')
        parser.add_argument('low', type=float,location='args')
        parser.add_argument('ask', type=float,location='args')
        parser.add_argument('bid', type=float,location='args')
        parser.add_argument('tradeDir' , type=int,location='args')
        parser.add_argument('isCandle' , type=int,location='args')
        args = parser.parse_args()
        open = args.open
        close = args.close
        high = args.high
        low = args.low
        ask = args.ask
        bid = args.bid
        tradeDir = args.tradeDir
        isCandle = args.isCandle
        assert open > 0
        assert close > 0
        assert high > 0
        assert low > 0
        assert ask > 0
        assert bid > 0

        assert tradeDir == 0 or tradeDir == 1 or tradeDir == 2
        assert isCandle == 0 or isCandle == 1
        return redirect("http://127.0.0.1:5000/?open=%f&close=%f&high=%f&low=%f&ask=%f&bid=%f&tradeDir=%f&isCandle=%f"%(open,
        close,high,low,ask,bid,tradeDir,isCandle))
        

   









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--port", default=80, help="port number")
    args = parser.parse_args()
   
    
    #start server
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(MetaTrade, '/')
    app.run(host="0.0.0.0",port=args.port)




    









    

    
        



        
