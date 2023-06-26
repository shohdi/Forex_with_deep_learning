//+------------------------------------------------------------------+

//|                                                 collect_data.mq4 |

//|                        Copyright 2022, MetaQuotes Software Corp. |

//|                                             https://www.mql5.com |

//+------------------------------------------------------------------+

#property copyright "Copyright 2022, MetaQuotes Software Corp."

#property link      "https://www.mql5.com"

#property version   "1.00"

#property strict



int myPeriod = PERIOD_M1;

int filehandle;

  datetime D1;



//+------------------------------------------------------------------+

//| Expert initialization function                                   |

//+------------------------------------------------------------------+

int OnInit()

  {

//---

Print("before FileHandle  ");

   filehandle=FileOpen("collected.csv",FILE_WRITE|FILE_CSV);

   if(filehandle<0)

     {

      Print("Failed to open the file by the absolute path ");

      Print("Error code ",GetLastError());

     }

     Print("Before writing title! " , filehandle);

   FileWrite(filehandle,"open","close","high","low","ask","bid","day","week","month");

   Print("after writing title! " , filehandle);

//---

   return(INIT_SUCCEEDED);

  }

//+------------------------------------------------------------------+

//| Expert deinitialization function                                 |

//+------------------------------------------------------------------+

void OnDeinit(const int reason)

  {

//---

      Print("Closing File Handle! ", filehandle);

      FileClose(filehandle);

      

  }

//+------------------------------------------------------------------+

//| Expert tick function                                             |

//+------------------------------------------------------------------+

void OnTick()

  {

//---

     if(D1!=iTime(Symbol(),myPeriod,0)) // new candle on D1

     {

         D1=iTime(Symbol(),myPeriod,0);

         

         double open = iOpen(Symbol(),myPeriod,1);

         double close = iClose(Symbol(),myPeriod,1);

         double high = iHigh(Symbol(),myPeriod,1);

         double low = iLow(Symbol(),myPeriod,1);

         double ask = Ask;

         double bid = Bid;

         double day = iClose(Symbol(),PERIOD_H1,1 * 24);
         double week = iClose(Symbol(),PERIOD_H1,1 * 24 * 5);
         double month = iClose(Symbol(),PERIOD_H1,1 * 24 * 5 * 4);

         if (open > 0 && close > 0 && high > 0 && low > 0 && ask > 0 && bid > 0 && day > 0 && week > 0 && month > 0)

         {

            //Print("Before writing candle! " , filehandle);

            FileWrite(filehandle,open,close,high,low,ask,bid,day,week,month);

            //Print("After writing candle! " , filehandle);

         }

         

     

         

     }

  }

//+------------------------------------------------------------------+

