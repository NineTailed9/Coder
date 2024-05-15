def dec_num_to_bin_list(num):
	return list(bin(num))[2:]

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
	# Calculation of n, k, r
	n = len(matrix[0])
	k = len(matrix)
	r = n - k

	# Calculation of weight coefficients
	bin_counter_vec = 0

	w = []
	for i in range(0, len(matrix[0])):
		w.append(0)
	
	for i in range(0, 2**k):
		bin_counter_vec = dec_num_to_bin_list(i)

		for i in range(len(dec_num_to_bin_list(2**k - 1)) - len(bin_counter_vec)):
			bin_counter_vec.insert(0, '0')

		print(bin_counter_vec, end = '\t')
		
		rows_to_xor = []
		for j in range(0, len(bin_counter_vec)):
			if bin_counter_vec[j] == '1':
				rows_to_xor.append(matrix[j])

		xor_result = xor_multiple_rows(rows_to_xor)
		print(xor_result, end=' ')
		
		ones_counter = 0
		for elem in xor_result:
			if elem == 1:
				ones_counter += 1

		print(ones_counter)

		for j in range(0, len(w)):
			if ones_counter == j:
				w[j] += 1

	# Calculation d, qo, qi
	for i, num in enumerate(w[1:]):
		if num != 0:
			d = i + 1
			break
	qo = d - 1
	qi = round(qo / 2)
	
	# Results
	print(f'\nn = {n}')
	print(f'k = {k}')
	print(f'r = {r}')
	print(f'd = {d}')
	print(f'qo = {qo}')
	print(f'qi = {qi}')
	print(f'w = {w}')
		
if __name__ == '__main__':

	G = []

	with open('Data.txt') as file:
		file_data = file.readlines()
		counter = 0
		for i in range(file_data.index('# 1\n') + 1, len(file_data)):
			current_line = list(file_data[i])
			G.append([])
			for j in range(0, len(current_line)):
				if j % 2 == 0:
					G[counter].append(int(current_line[j]))
			counter += 1
		
	for item in G:
		print(item)
	print()
			
	first_processing(G)
