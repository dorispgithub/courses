def skip_elements(elements):
	# Initialize variables
	new_list = []
	i = 0

	# Iterate through the list
	for i,j in enumerate(elements):
		# Does this element belong in the resulting list?
		if i%2==0:
		  new_list.append(j)
		else: 
		  continue
	return new_list
