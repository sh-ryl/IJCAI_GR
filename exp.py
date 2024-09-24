required_objects = ['cloth', 'stick']
d = [{'cloth': '0.1', 'stick': '0.9'}, {'cloth': '0.5', 'stick': '0.5'}, {'cloth': '0.8', 'stick': '0.2'}, {'cloth': '0.3', 'stick': '0.7'}, {'cloth': '0.7',
                                                                                                                                              'stick': '0.3'}, {'cloth': '0.2', 'stick': '0.8'}, {'cloth': '0.4', 'stick': '0.6'}, {'cloth': '0.9', 'stick': '0.1'}, {'cloth': '0.6', 'stick': '0.4'}]
other = [x for x in range(len(d))]

combined = sorted(zip(d, other), key=lambda x: x[0][required_objects[0]])

# Step 2: Extract sorted `d` and `other_list`
sorted_d, sorted_other_list = zip(*combined)

# Convert back to lists if needed
sorted_d = list(sorted_d)
sorted_other_list = list(sorted_other_list)

print("Sorted `d`:", sorted_d)
print("Sorted `other_list`:", sorted_other_list)
# d = sorted(enumerate(d), key=lambda x: x[1][required_objects[0]])

# print(d)
