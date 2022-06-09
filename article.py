import numpy as np 


result = [1, 2, 3, 5]
array2 = [3, 5, 6, 7]

array = [[1,2,3,4], [1,2,3,4]]
for list in array:
    
    result = np.add(result, list)

result = np.divide(result, 2)
print(result)