# example rename filename in the list

filenames = ["program.c", "stdio.hpp", "sample.hpp", "a.out", "math.hpp", "hpp.out"]
newfilenames = []
for x in range(len(filenames)):
    old_name = filenames[x]
    if old_name.endswith(".hpp"):
      new_name = old_name.replace('.hpp','.h')
    else:
      new_name = old_name;
    newfilenames.append((new_name))
print (newfilenames)
