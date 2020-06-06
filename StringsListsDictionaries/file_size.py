# just some random example of tuple

def file_size(file_info):
	name, type, size= file_info
	return("{:.2f}".format(size / 1024))
