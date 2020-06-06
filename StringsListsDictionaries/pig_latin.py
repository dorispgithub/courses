def pig_latin(text):
  say = ""
  # Separate the text into words
  words = text.split()
  for word in words:
    # Create the pig latin word and add it to the list
    texts = word[1:] + word[0] + "ay" + " "
    say += texts
    # Turn the list back into a phrase
  return say
