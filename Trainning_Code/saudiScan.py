import requests

def get_saudi_stock_symbols():
  """Gets a list of Saudi stock symbols from the Mubasher API.

  Returns:
    A list of strings, each representing a Saudi stock symbol.
  """

  # Make a GET request to the Mubasher API.
  response = requests.get("https://www.mubasher.info/api/1/listed-companies?country=sa&size=290&start=1")

  # Check if the request was successful.
  if response.status_code != 200:
    raise Exception("Error fetching Saudi stock symbols: {}".format(response.status_code))

  # Parse the JSON response.
  data = response.json()

  
  # Extract the symbol field values from the data.
  symbols = [item["symbol"] for item in data["rows"]]

  # Remove the dot and any character after it from each symbol.
  symbols = [symbol.split(".")[0] for symbol in symbols]

  # Add ".SR" to the end of each symbol.
  symbols = [symbol + ".SR" for symbol in symbols]

  # Return the list of symbols.
  return symbols
  
  

if __name__ == "__main__":
  # Get the list of Saudi stock symbols.
  symbols = get_saudi_stock_symbols()

  # Print the list of symbols.
  print(symbols)