def count_letters(text):
  result = {}
  # Go through each letter in the text
  for letter in text.lower():
    # Check if the letter needs to be counted or not
    if letter.isalpha():
      if letter in result:
        result[letter]+=1
      else:
        result[letter]=1
    # Add or increment the value in the dictionary
  return result
