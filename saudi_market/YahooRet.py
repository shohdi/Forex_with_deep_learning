from typing import List
from typing import Any
from dataclasses import dataclass
import json
@dataclass
class Adjclose:
    adjclose: List[float]

    @staticmethod
    def from_dict(obj: Any) -> 'Adjclose':
        _adjclose = [float(y if y is not None else 0) for y in obj.get("adjclose")]
        return Adjclose(_adjclose)

@dataclass
class Pre:
    timezone: str
    start: int
    end: int
    gmtoffset: int

    @staticmethod
    def from_dict(obj: Any) -> 'Pre':
        _timezone = str(obj.get("timezone"))
        _start = int(obj.get("start"))
        _end = int(obj.get("end"))
        _gmtoffset = int(obj.get("gmtoffset"))
        return Pre(_timezone, _start, _end, _gmtoffset)

@dataclass
class Regular:
    timezone: str
    start: int
    end: int
    gmtoffset: int

    @staticmethod
    def from_dict(obj: Any) -> 'Regular':
        _timezone = str(obj.get("timezone"))
        _start = int(obj.get("start"))
        _end = int(obj.get("end"))
        _gmtoffset = int(obj.get("gmtoffset"))
        return Regular(_timezone, _start, _end, _gmtoffset)

@dataclass
class Post:
    timezone: str
    start: int
    end: int
    gmtoffset: int

    @staticmethod
    def from_dict(obj: Any) -> 'Post':
        _timezone = str(obj.get("timezone"))
        _start = int(obj.get("start"))
        _end = int(obj.get("end"))
        _gmtoffset = int(obj.get("gmtoffset"))
        return Post(_timezone, _start, _end, _gmtoffset)

@dataclass
class CurrentTradingPeriod:
    pre: Pre
    regular: Regular
    post: Post

    @staticmethod
    def from_dict(obj: Any) -> 'CurrentTradingPeriod':
        _pre = Pre.from_dict(obj.get("pre"))
        _regular = Regular.from_dict(obj.get("regular"))
        _post = Post.from_dict(obj.get("post"))
        return CurrentTradingPeriod(_pre, _regular, _post)    
@dataclass
class Meta:
    currency: str
    symbol: str
    exchangeName: str
    instrumentType: str
    firstTradeDate: int
    regularMarketTime: int
    gmtoffset: int
    timezone: str
    exchangeTimezoneName: str
    regularMarketPrice: float
    chartPreviousClose: float
    priceHint: int
    currentTradingPeriod: CurrentTradingPeriod
    dataGranularity: str
    range: str
    validRanges: List[str]

    @staticmethod
    def from_dict(obj: Any) -> 'Meta':
        _currency = str(obj.get("currency"))
        _symbol = str(obj.get("symbol"))
        _exchangeName = str(obj.get("exchangeName"))
        _instrumentType = str(obj.get("instrumentType"))
        _firstTradeDate = int(obj.get("firstTradeDate"))
        _regularMarketTime = int(obj.get("regularMarketTime"))
        _gmtoffset = int(obj.get("gmtoffset"))
        _timezone = str(obj.get("timezone"))
        _exchangeTimezoneName = str(obj.get("exchangeTimezoneName"))
        _regularMarketPrice = float(obj.get("regularMarketPrice"))
        _chartPreviousClose = float(obj.get("chartPreviousClose"))
        _priceHint = int(obj.get("priceHint"))
        _currentTradingPeriod = CurrentTradingPeriod.from_dict(obj.get("currentTradingPeriod"))
        _dataGranularity = str(obj.get("dataGranularity"))
        _range = str(obj.get("range"))
        _validRanges = [str(y) for y in obj.get("validRanges")]
        return Meta(_currency, _symbol, _exchangeName, _instrumentType, _firstTradeDate, _regularMarketTime, _gmtoffset, _timezone, _exchangeTimezoneName, _regularMarketPrice, _chartPreviousClose, _priceHint, _currentTradingPeriod, _dataGranularity, _range, _validRanges)

@dataclass
class Quote:
    close: List[float]
    volume: List[int]
    open: List[float]
    low: List[float]
    high: List[float]

    @staticmethod
    def from_dict(obj: Any) -> 'Quote':
        _close = [float(y if y is not None else 0) for y in obj.get("close")]
        _volume = [int(y if y is not None else 0) for y in obj.get("volume")]
        _open = [float(y if y is not None else 0) for y in obj.get("open")]
        _low = [float(y if y is not None else 0) for y in obj.get("low")]
        _high = [float(y if y is not None else 0) for y in obj.get("high")]
        return Quote(_close, _volume, _open, _low, _high)


@dataclass
class Indicators:
    quote: List[Quote]
    adjclose: List[Adjclose]

    @staticmethod
    def from_dict(obj: Any) -> 'Indicators':
        _quote = [Quote.from_dict(y) for y in obj.get("quote")]
        _adjclose = [Adjclose.from_dict(y) for y in obj.get("adjclose")]
        return Indicators(_quote, _adjclose)

@dataclass
class Result:
    meta: Meta
    timestamp: List[int]
    indicators: Indicators

    @staticmethod
    def from_dict(obj: Any) -> 'Result':
        _meta = Meta.from_dict(obj.get("meta"))
        _timestamp = [int(y) for y in obj.get("timestamp")]
        _indicators = Indicators.from_dict(obj.get("indicators"))
        return Result(_meta, _timestamp, _indicators)


@dataclass
class Chart:
    result: List[Result]
    error: str

    @staticmethod
    def from_dict(obj: Any) -> 'Chart':
        _result = [Result.from_dict(y) for y in obj.get("result")]
        _error = str(obj.get("error"))
        return Chart(_result, _error)
















@dataclass
class YahooRet:
    chart: Chart

    @staticmethod
    def from_dict(obj: Any) -> 'YahooRet':
        _chart = Chart.from_dict(obj.get("chart"))
        return YahooRet(_chart)

# Example Usage
# jsonstring = json.loads(myjsonstring)
# YahooRet = YahooRet.from_dict(jsonstring)
