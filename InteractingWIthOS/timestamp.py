import os
import datetime

def file_date(filename):
  # Create the file in the current directory
  with open(filename, 'w') as newfile:
    timestamp = datetime.datetime.now()
  # Convert the timestamp into a readable format, then into a string
    timestampStr = timestamp.strftime("%Y-%b-%d")
  # Return just the date portion 
  # Hint: how many characters are in “yyyy-mm-dd”? 
  #print('Current Timestamp : ', timestampStr)
  return ("{}".format(timestampStr))

print(file_date("newfile.txt")) 
# Should be today's date in the format of yyyy-mm-dd
