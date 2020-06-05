def fractional_part(numerator, denominator):
  if denominator>0:
    number = numerator/denominator
    return str(number-int(number))[1:]
  else:
    return 0
