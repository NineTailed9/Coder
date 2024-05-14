def xor(first_row, second_row):
	result_row = []
	for i in range(0, len(first_row)):
		if first_row[i] == second_row[i]:
			result_row.append(0)
		else:
			result_row.append(1)
		
	return result_row

def xor_multiple_rows(matrix):
	if len(matrix) == 0:
		return [0]
	if len(matrix) == 1:
		return matrix[0]
	
	buffer_row = []
	for i in range(0, len(matrix[0])):
		buffer_row.append(0)

	for i in range(0, len(matrix)):
		buffer_row = xor(buffer_row, matrix[i])
	
	return buffer_row

def first_processing(matrix):
	colums = len(matrix)
	arr_rev = 0

	result_ones_amount = []
	for i in range(0, len(matrix[0])):
		result_ones_amount.append(0)
	
	for i in range(0, 2**colums):
		arr_rev = (list(bin(i))[2:])
		print(arr_rev, end = '')
		print((colums - len(arr_rev)) * '     ', end='\t')
		
		rows_to_xor = []
		for j in range(0, len(arr_rev)):
			if arr_rev[j] == '1':
				rows_to_xor.append(matrix[j])

		xor_result = xor_multiple_rows(rows_to_xor)
		print(xor_result, end=' ')
		
		ones_counter = 0
		for elem in xor_result:
			if elem == 1:
				ones_counter += 1

		print(f'ones: {ones_counter}', end='')

		for j in range(0, len(result_ones_amount)):
			if ones_counter == j:
				result_ones_amount[j] += 1

		print()

	print()
	for i in range(0, len(result_ones_amount)):
		print(f' num of {i} = {result_ones_amount[i]}')
		
if __name__ == '__main__':
	
	G = [[0, 0, 0, 1, 1, 0, 1, 1],
		 [0, 1, 0, 0, 1, 1, 1, 1],
		 [0, 1, 0, 1, 1, 1, 1, 0],
		 [0, 1, 1, 1, 1, 0, 0, 0],
		 [1, 1, 0, 0, 0, 1, 0, 0]]
			
	first_processing(G)
	