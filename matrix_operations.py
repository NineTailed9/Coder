from math import log, factorial
from copy import deepcopy
from random import randint

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


def xor_simb(num_1, num_2):
    if num_1 != num_2:
        return 1
    else:
        return 0


def matrix_print(matrix, new_line=True):
    for row in matrix:
        print(row)
    if new_line:
        print()


def matrix_transpose(primary_matrix: list):
    columns = len(primary_matrix[0])
    rows = len(primary_matrix)

    transposed = []
    for column_index in range(0, columns):
        transposed.append([])
        for row_index in range(0, rows):
            transposed[column_index].append(0)

    for column_index in range(0, columns):
        for row_index in range(0, rows):
            transposed[column_index][row_index] = primary_matrix[row_index][
                column_index
            ]

    return transposed


def div_matrix_in_two(primary_matrix: list, div_index: int):
    matrix = deepcopy(primary_matrix)
    columns = len(matrix[0])
    rows = len(matrix)

    first_matrix = []
    second_matrix = []
    for row_index in range(0, rows):
        first_matrix.append([])
        second_matrix.append([])
        for column_index in range(0, columns):
            if column_index <= div_index:
                first_matrix[row_index].append(matrix[row_index][column_index])
            else:
                second_matrix[row_index].append(matrix[row_index][column_index])

    return first_matrix, second_matrix


def join_matrix_in_one(first_matrix: list, second_matrix: list):
    rows_first = len(first_matrix)
    columns_first = len(first_matrix[0])
    rows_second = len(second_matrix)
    columns_second = len(second_matrix[0])

    if (rows_first > rows_second) and (columns_first == columns_second):
        second_matrix = deepcopy(second_matrix)

        for i in range(1, rows_first - rows_second + 1):
            rows_second = len(second_matrix)
            for row_index in range(0, rows_second):
                second_matrix[row_index].append(0)

            second_matrix.append([])
            for j in range(0, len(second_matrix[0]) - 1):
                second_matrix[len(second_matrix) - 1].append(0)
            second_matrix[len(second_matrix) - 1].append(1)
        columns_second = len(second_matrix[0])
    if columns_second > rows_first:
        second_matrix = deepcopy(second_matrix)

        for row_index in range(0, rows_second):
            second_matrix[row_index].pop()
        second_matrix.pop()

        rows_second = len(second_matrix)
        columns_second = len(second_matrix[0])

    combined_matrix = []
    for row_index in range(0, rows_first):
        combined_matrix.append([])
        for column_index in range(0, columns_first):
            combined_matrix[row_index].append(first_matrix[row_index][column_index])
        for column_index in range(0, columns_second):
            combined_matrix[row_index].append(second_matrix[row_index][column_index])

    return combined_matrix


def get_single_num_columns_index(primary_matrix: list):
    matrix = deepcopy(primary_matrix)

    columns = len(matrix[0])
    rows = len(matrix)

    single_num_columns_position = []
    for i in range(0, columns):
        single_num_columns_position.append([])

    total_single_num_columns = 0

    for column_index in range(0, columns):

        ones_counter = [0, 0]
        for row_index in range(0, rows):
            if matrix[row_index][column_index] == 1:
                ones_counter[0] += 1
                ones_counter[1] = row_index

        if ones_counter[0] == 1:
            single_num_columns_position[column_index] = ones_counter[1]
            total_single_num_columns += 1

    return total_single_num_columns, single_num_columns_position


def matrix_to_single_num_columns(
    primary_matrix: list, operations_to_solve=10, IQ_edge_return=True
):
    columns = len(primary_matrix[0])
    rows = len(primary_matrix)

    matrix = []

    while True:
        matrix = deepcopy(primary_matrix)
        operations = 1
        log_list = []

        while True:
            first_random = randint(0, rows - 1)
            second_random = randint(0, rows - 1)
            while first_random == second_random:
                first_random = randint(0, rows - 1)
                second_random = randint(0, rows - 1)

            # xor rows
            for i in range(0, len(matrix[0])):
                if matrix[first_random][i] == matrix[second_random][i]:
                    matrix[first_random][i] = 0
                else:
                    matrix[first_random][i] = 1

            log_list.append([first_random, second_random])

            num_single_ones = get_single_num_columns_index(matrix)[0]

            # if amount of columns with single 1 is right, or too many operations
            if (num_single_ones == rows) or (operations > operations_to_solve):
                break
            operations += 1

        if (num_single_ones == rows) and (operations <= operations_to_solve):
            columns_position = get_single_num_columns_index(matrix)[1]
            equal_position_error = False
            for index, elem in enumerate(columns_position):
                if index == len(columns_position) - 1:
                    break
                if (elem != []) and (elem in columns_position[index + 1 :]):
                    equal_position_error = True
            if equal_position_error == True:
                pass
            else:
                print("XORed rows:")
                for swap in log_list:
                    print(f"{swap[0]} row = {swap[0]} xor {swap[1]}")
                print()
                return matrix


def matrix_to_unit_matrix(primary_matrix: list):
    columns = len(primary_matrix[0])
    rows = len(primary_matrix)

    new_matrix = []
    for row_index in range(0, rows):
        new_matrix.append([])
        for column_index in range(0, columns):
            new_matrix[row_index].append(0)

    counter = 0
    ones_position = get_single_num_columns_index(primary_matrix)[1]
    while counter < rows:
        current_pos = ones_position.index(counter)

        for row in range(0, rows):
            new_matrix[row][counter] = primary_matrix[row][current_pos]
        counter += 1

    if rows == columns:
        return new_matrix

    for i in range(0, columns - rows):
        if i == 0:
            non_ones_column = ones_position.index([])
        else:
            non_ones_column = ones_position.index([], non_ones_column + 1)

        for row in range(0, rows):
            new_matrix[row][rows + i] = primary_matrix[row][non_ones_column]

    return new_matrix


def matrix_columns_swap_index(first_matrix: list, second_matrix: list):
    columns = len(first_matrix[0])
    rows = len(first_matrix)

    old_columns = []
    for column_index in range(0, columns):
        old_columns.append([column_index, []])
        for row_index in range(0, rows):
            old_columns[column_index][1].append(first_matrix[row_index][column_index])

    p = []
    for column_index in range(0, columns):
        current_column = []
        for row_index in range(0, rows):
            current_column.append(second_matrix[row_index][column_index])

        for num, column in old_columns:
            if current_column == column:
                p.append([num, column_index])
                old_columns.remove([num, column])
                break

    return sorted(p, key=lambda x: x[0])


def matrix_columns_swap_by_list(primary_matrix: list, p=list):
    columns = len(primary_matrix[0])
    rows = len(primary_matrix)

    new_matrix = []
    for row_index in range(0, rows):
        new_matrix.append([])
        for column_index in range(0, columns):
            new_matrix[row_index].append(0)

    for i in range(0, len(p)):
        prev_column = p[i][0]
        future_column = p[i][1]

        for row_index in range(0, rows):
            new_matrix[row_index][future_column] = primary_matrix[row_index][
                prev_column
            ]

    return new_matrix


def matrix_accuracy_HG(
    first_matrix: list, second_matrix_pr: list, print_messages=False
):
    second_matrix = matrix_transpose(second_matrix_pr)

    columns_first = len(first_matrix[0])
    rows_first = len(first_matrix)
    columns_second = len(second_matrix[0])
    rows_second = len(second_matrix)

    accuracy_matrix = []
    for column_index in range(0, columns_second):
        accuracy_matrix.append([])
        for row_index in range(0, rows_first):
            accuracy_matrix[column_index].append(0)

    for row_first_index in range(0, rows_first):
        summ_current_iter = 0
        for column_second_index in range(0, columns_second):
            for bit_index in range(0, columns_first):
                if print_messages:
                    print(
                        f"{first_matrix[row_first_index][bit_index]} * {second_matrix[bit_index][column_second_index]} + ",
                        end="",
                    )
                summ_current_iter = xor_simb(
                    summ_current_iter,
                    first_matrix[row_first_index][bit_index]
                    * second_matrix[bit_index][column_second_index],
                )
            if print_messages:
                print(
                    f"= {summ_current_iter} | {column_second_index} row {row_first_index} column"
                )
            accuracy_matrix[column_second_index][row_first_index] = summ_current_iter
    if print_messages:
        print("\nH*G transposed")
    return accuracy_matrix


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
            sum_1 = recursive_vertor_division_shanfano(vector[: counter + 1])
            sum_2 = recursive_vertor_division_shanfano(vector[counter + 1 :])
            break

        counter += 1

    for item in sum_1:
        bin_code_comb[item[1]].append(1)
    for item in sum_2:
        bin_code_comb[item[1]].append(0)

    return sum_1 + sum_2


def shannon_fano(vector_prim: list):
    vector = deepcopy(vector_prim)
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

    print("n | p(x) | code")
    print("---------------------")
    avg_len = 0
    for i in range(0, len(bin_code_comb)):
        bin_code_comb[i] = bin_code_comb[i][::-1]
        avg_len += len(bin_code_comb[i])

        j = vector_prim.index(vector[i][0])
        print(f"{(j // 3)}{j % 3} | {vector[i][0]} | {bin_code_comb[i]}")
    return avg_len / len(vector)


def huffman(vector_prim: list):
    vector = deepcopy(vector_prim)
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
            vector.append(
                [round(minimal_1[0] + minimal_2[0], 2), minimal_1[1] + minimal_2[1]]
            )

            for item in minimal_1[1]:
                bin_code_comb[item].append(0)
            for item in minimal_2[1]:
                bin_code_comb[item].append(1)

        if isinstance(minimal_1[1], list) and isinstance(minimal_2[1], int):
            vector.append(
                [round(minimal_1[0] + minimal_2[0], 2), minimal_1[1] + [minimal_2[1]]]
            )

            for item in minimal_1[1]:
                bin_code_comb[item].append(0)
            bin_code_comb[minimal_2[1]].append(1)

        if isinstance(minimal_1[1], int) and isinstance(minimal_2[1], list):
            vector.append(
                [round(minimal_1[0] + minimal_2[0], 2), [minimal_1[1]] + minimal_2[1]]
            )

            bin_code_comb[minimal_1[1]].append(0)
            for item in minimal_2[1]:
                bin_code_comb[item].append(1)

        if isinstance(minimal_1[1], int) and isinstance(minimal_2[1], int):
            vector.append(
                [round(minimal_1[0] + minimal_2[0], 2), [minimal_1[1]] + [minimal_2[1]]]
            )

            bin_code_comb[minimal_1[1]].append(0)
            bin_code_comb[minimal_2[1]].append(1)

    vector_prim_sorted = deepcopy(vector_prim)
    vector_prim_sorted.sort()
    vector_prim_sorted.reverse()

    print("n | p(x) | code")
    print("---------------------")
    avg_len = 0
    for i in range(0, len(bin_code_comb)):
        bin_code_comb[i] = bin_code_comb[i][::-1]
        avg_len += len(bin_code_comb[i])

        j = vector_prim.index(vector_prim_sorted[i])
        print(f"{(j // 3)}{j % 3} | {vector_prim_sorted[i]} | {bin_code_comb[i]}")
    return avg_len / len(vector_prim)
