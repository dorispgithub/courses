# iterate throught dictionary

def car_listing(car_prices):
  result = ""
  for key in car_prices:
    result += "{} costs {} dollars".format(key,car_prices[key]) + "\n"
  return result
