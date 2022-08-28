//+------------------------------------------------------------------+
//|                                                       shohdi.mq4 |
//|                                                  Shohdy ElSheemy |
//|                                                 http://127.0.0.1 |
//+------------------------------------------------------------------+
#property copyright "Shohdy ElSheemy"
#property link      "http://127.0.0.1"
#property version   "1.00"
#property strict

#define MAGICMA  182182



//my functions
int CalculateCurrentOrders()
  {
   int buys=0,sells=0;
//---
   for(int i=0;i<OrdersTotal();i++)
     {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
      if(OrderSymbol()==Symbol() && OrderMagicNumber()==MAGICMA)
        {
         if(OrderType()==OP_BUY)  buys++;
         if(OrderType()==OP_SELL) sells++;
        }
     }
//--- return orders volume
   if(buys>0) return(buys);
   else       return(-sells);
  }
  
  
  
void openUp(double lots)
  {
     if (CalculateCurrentOrders() == 0)
     {
         Print("Opening Up Order !!");
        int res=OrderSend(Symbol(),OP_BUY,lots,Ask,5,0,0,"",MAGICMA,0,Green);
        
         return;
     }
     
  }



  void openDown(double lots)
  {
     if (CalculateCurrentOrders() == 0)
     {
         Print("Opening Down Order !!");
         int res=OrderSend(Symbol(),OP_SELL,lots,Bid,5,0,0,"",MAGICMA,0,Red);
         
         return;
     }
     
  }
  
  
  
  void closeDown()
  {
   
   for(int i=0;i<OrdersTotal();i++)
     {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
      if(OrderSymbol()==Symbol() && OrderMagicNumber()==MAGICMA)
        {
         //if(OrderType()==OP_BUY)  buys++;
         if(OrderType()==OP_SELL) {
            Print("Closing down order ",OrderTicket());
            OrderClose(OrderTicket(),OrderLots(),Ask,5,Red);
           
         }
        }
     }

  }
  
  
  
  void closeUp()
  {
   
   for(int i=0;i<OrdersTotal();i++)
     {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
      if(OrderSymbol()==Symbol() && OrderMagicNumber()==MAGICMA)
        {
         //if(OrderType()==OP_BUY)  buys++;
         if(OrderType()==OP_BUY) {
            Print("Closing up order ",OrderTicket());
            OrderClose(OrderTicket(),OrderLots(),Bid,5,Red);
            
         }
        }
     }

  }
  
  
  int OpenRequestGetAction(int i,bool history)
  {
      double open = iOpen(Symbol(),PERIOD_M15,i);
      double close = iClose(Symbol(),PERIOD_M15,i);
      double high = iHigh(Symbol(),PERIOD_M15,i); 
      
      double low = iLow(Symbol(),PERIOD_M15,i); 
      double ask = close + (Ask-Bid);
      double bid = close;
      if (!history)
      {  
         ask = Ask;
         bid = Bid;
         
      }
      
      string url = StringFormat("http://127.0.0.1/?open=%f&close=%f&high=%f&low=%f&ask=%f&bid=%f",open,close,high,low,ask,bid);
      
     string ret = createRequest(url);
     
     int action = StrToInteger(ret);
     return action;
      
  }
  
  
  string createRequest(string url)
  {
      string cookie=NULL,headers;
   char post[],result[];
   int res;


   ResetLastError();

   int timeout=0; //--- Timeout below 1000 (1 sec.) is not enough for slow Internet connection
   res=WebRequest("GET",url,cookie,NULL,timeout,post,0,result,headers);
//--- Checking errors
   if(res==-1)
     {
      Print("Error in WebRequest. Error code  =",GetLastError());
      //--- Perhaps the URL is not listed, display a message about the necessity to add the address
      MessageBox("Add the address '"+url+"' in the list of allowed URLs on tab 'Expert Advisors'","Error",MB_ICONINFORMATION);
      return "";
     }
   else
     {
          //--- Load successfully
         PrintFormat("The file has been successfully loaded, File size =%d bytes.",ArraySize(result));
         string ret = "";
         for (int i =0;i<ArraySize(result);i++)
         {
            ret = ret + result[i]; 
         }
         return ret;
     }
  }
  

void handleAction(int action)
{
      double modeMinLot = MarketInfo(Symbol(), MODE_MINLOT) ;
      
         
      if (action == 1)
      {
         //open buy
         //openUp(
      }
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   for (int i = 99;i>= 1;i--)
   {
      int action = OpenRequestGetAction(i,true);
      handleAction(action);
   }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   
  }
//+------------------------------------------------------------------+
