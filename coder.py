from math import log

# Global var init
bin_code_comb = []

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

def recursive_vertor_division_shanfano(vector):
    if len(vector) == 1:
        return vector
    
    counter = 0
    sum_elems = 0
    sum_1 = 0
    sum_2 = 0
    vector_match = 0

    for i in range(0, len(vector)):
        vector_match = vector_match + vector[i][0]
    vector_match /= 2
    
    while True:
        sum_elems += vector[counter][0]

        if sum_elems >= vector_match:
            sum_1 = recursive_vertor_division_shanfano(vector[:counter + 1])
            sum_2 = recursive_vertor_division_shanfano(vector[counter + 1:])
            break

        counter += 1

    for item in sum_1:
        bin_code_comb[item[1]].append(1)
    for item in sum_2:
        bin_code_comb[item[1]].append(0)
    
    return sum_1 + sum_2

def shannon_fano(vector):
    vector.sort()
    vector.reverse()
    vector_len = len(vector)
	
    vector_new = []
    for i in range(0, len(vector)):
        vector_new.append([vector[i], i])
    vector = vector_new

    global bin_code_comb
    bin_code_comb = []
    for i in range(len(vector)):
        bin_code_comb.append([])

    recursive_vertor_division_shanfano(vector)
	
    avg_len = 0
    for i in range(0, len(bin_code_comb)):
        bin_code_comb[i] = bin_code_comb[i][::-1]
        avg_len += len(bin_code_comb[i])

        print(vector[i][0], bin_code_comb[i])
    return (avg_len/len(vector))
	
def huffman(vector):
    vector_prim = vector.copy()
    vector.sort()
    vector.reverse()

    vector_new = []
    for i in range(0, len(vector)):
        vector_new.append([vector[i], i])
    vector = vector_new

    global bin_code_comb
    bin_code_comb = []
    for i in range(len(vector)):
        bin_code_comb.append([])

    minimal_1 = []
    minimal_1_index = 0
    minimal_2 = []
    minimal_2_index = 0
    while len(vector) != 1:
        minimal_1 = [2, 0]
        minimal_1_index = 1000
        minimal_2 = [2, 0]
        minimal_2_index = 1000

        for i in range(0, len(vector)):
            if vector[i][0] < minimal_1[0]:
                minimal_1 = vector[i]
                minimal_1_index = i
        vector.pop(minimal_1_index)

        for i in range(0, len(vector)):
            if vector[i][0] < minimal_2[0]:
                minimal_2 = vector[i]
                minimal_2_index = i
        vector.pop(minimal_2_index)

        if minimal_1[0] == minimal_2[0]:
            minimal_1, minimal_2 = minimal_2, minimal_1

        min_1_type = type(minimal_1[1])
        min_2_type = type(minimal_2[1])

        if isinstance(minimal_1[1], list) and isinstance(minimal_2[1], list):
            vector.append([round(minimal_1[0] + minimal_2[0], 2), minimal_1[1] + minimal_2[1]])

            for item in minimal_1[1]:
                bin_code_comb[item].append(0)
            for item in minimal_2[1]:
                bin_code_comb[item].append(1)

        if isinstance(minimal_1[1], list) and isinstance(minimal_2[1], int):
            vector.append([round(minimal_1[0] + minimal_2[0], 2), minimal_1[1] + [minimal_2[1]]])

            for item in minimal_1[1]:
                bin_code_comb[item].append(0)   
            bin_code_comb[minimal_2[1]].append(1)

        if isinstance(minimal_1[1], int) and isinstance(minimal_2[1], list):
            vector.append([round(minimal_1[0] + minimal_2[0], 2), [minimal_1[1]] + minimal_2[1]])

            bin_code_comb[minimal_1[1]].append(0)
            for item in minimal_2[1]:
                bin_code_comb[item].append(1)

        if isinstance(minimal_1[1], int) and isinstance(minimal_2[1], int):
            vector.append([round(minimal_1[0] + minimal_2[0], 2), [minimal_1[1]] + [minimal_2[1]]])

            bin_code_comb[minimal_1[1]].append(0)
            bin_code_comb[minimal_2[1]].append(1)

    avg_len = 0
    for i in range(0, len(bin_code_comb)):
        bin_code_comb[i] = bin_code_comb[i][::-1]
        avg_len += len(bin_code_comb[i])
        print(vector_prim[i], bin_code_comb[i])
    return (avg_len/len(vector_prim))

def first_processing(matrix):
	print('Task 1')
	print('Matrix G')
	for item in matrix:
		print(item)
	print()
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

def fifth_processing(vector, sistem_base = 3):
	print('\nTask 5')
	print('Vector', vector)
	print('Standart code R =', round(log(len(vector), sistem_base), 2))

	print('Huffman code')
	print('R =', round(huffman(vector), 2))
	
	print('\nShannon-Fano code')
	print('R =', round(shannon_fano(vector), 2))

	# TODO: Repeat actions with combining two by two

if __name__ == '__main__':

	G = []

	with open('Data.txt') as file:
		file_data = file.readlines()
		counter = 0
		for i in range(file_data.index('# 1\n') + 1, file_data.index('# 5\n')):
			current_line = list(file_data[i])
			G.append([])
			for j in range(0, len(current_line)):
				if j % 2 == 0:
					G[counter].append(int(current_line[j]))
			counter += 1
        
		vector_tsk_5 = ''
		for i in range(file_data.index('# 5\n') + 1, len(file_data)):
			current_line = file_data[i]
			vector_tsk_5 = current_line.split()
			for j in range(0, len(vector_tsk_5)):
				vector_tsk_5[j] = float(vector_tsk_5[j])

	first_processing(G)
	fifth_processing(vector_tsk_5, 2)
