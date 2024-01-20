from typing import List
from typing import Any
from dataclasses import dataclass
import json


@dataclass
class Row:
    name: str
    url: str
    market: str
    sector: str
    marketUrl: str
    currency: str
    profileUrl: str
    symbol: str
    price: str
    changePercentage: str
    lastUpdate: str

    @staticmethod
    def from_dict(obj: Any) -> 'Row':
        _name = str(obj.get("name"))
        _url = str(obj.get("url"))
        _market = str(obj.get("market"))
        _sector = str(obj.get("sector"))
        _marketUrl = str(obj.get("marketUrl"))
        _currency = str(obj.get("currency"))
        _profileUrl = str(obj.get("profileUrl"))
        _symbol = str(obj.get("symbol"))
        _price = str(obj.get("price"))
        _changePercentage = str(obj.get("changePercentage"))
        _lastUpdate = str(obj.get("lastUpdate"))
        return Row(_name, _url, _market, _sector, _marketUrl, _currency, _profileUrl, _symbol, _price, _changePercentage, _lastUpdate)




@dataclass
class SaudiMarket:
    rows: List[Row]

    @staticmethod
    def from_dict(obj: Any) -> 'SaudiMarket':
        _rows = [Row.from_dict(y) for y in obj.get("rows")]
        return SaudiMarket(_rows)

# Example Usage
# jsonstring = json.loads(myjsonstring)
# SaudiMarket = SaudiMarket.from_dict(jsonstring)
