def group_list(group, users):
  members = ', '.join(users)
  return "{}: {}".format(group, members)
