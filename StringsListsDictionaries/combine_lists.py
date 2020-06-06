def combine_lists(list1, list2):
  fulllist = list2[:]
  fulllist.extend(reversed(list1))
  return fulllist
  # Generate a new list containing the elements of list2
  # Followed by the elements of list1 in reverse order
