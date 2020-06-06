def combine_guests(guests1, guests2):
  for key in guests1:
    if key in guests2:
      guests1[key] = (guests1[key],guests2[key])
    else:
      pass
  return guests1
  # Combine both dictionaries into one, with each key listed 
  # only once, and the value from guests1 taking precedence
