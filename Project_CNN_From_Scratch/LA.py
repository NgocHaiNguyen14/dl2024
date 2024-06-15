import random
class Helper:
    @staticmethod
    def dot_product(matrix1, matrix2):

        result = []
        for i in range(len(matrix1)):
            row = []
            for j in range(len(matrix2[0])):
                sum_val = 0
                for k in range(len(matrix1[0])):
                    sum_val += matrix1[i][k] * matrix2[k][j]
                row.append(sum_val)
            result.append(row)
        return result
    @staticmethod
    def add(matrix1, matrix2):
        result = []
        for i in range(len(matrix1)):
            row = []
            for j in range(len(matrix1[0])):
                row.append(matrix1[i][j] + matrix2[i][j])
            result.append(row)
        return result
    @staticmethod
    def transpose(matrix):
        result = []
        for j in range(len(matrix[0])):
            row = []
            for i in range(len(matrix)):
                row.append(matrix[i][j])
            result.append(row)
        return result

    @staticmethod
    def apply_function(func, matrix):
        if isinstance(matrix[0][0], list):  
            result = []
            for d in range(len(matrix)):
                depth_slice = []
                for i in range(len(matrix[0])):
                    row = []
                    for j in range(len(matrix[0][0])):
                        row.append(func(matrix[d][i][j]))
                    depth_slice.append(row)
                result.append(depth_slice)
            return result
        else:
            result = []
            for i in range(len(matrix)):
                row = []
                for j in range(len(matrix[0])):
                    row.append(func(matrix[i][j]))
                result.append(row)
            return result
    @staticmethod
    def elementwise_multiply(matrix1, matrix2):
        if isinstance(matrix1[0][0], (int, float)):  
            result = []
            for i in range(len(matrix1)):
                row = []
                for j in range(len(matrix1[0])):
                    row.append(matrix1[i][j] * matrix2[i][j])
                result.append(row)
            return result
        elif isinstance(matrix1[0][0], list): 
            result = []
            for i in range(len(matrix1)):
                slice_ = []
                for j in range(len(matrix1[0])):
                    row = []
                    for k in range(len(matrix1[0][0])):
                        row.append(matrix1[i][j][k] * matrix2[i][j][k])
                    slice_.append(row)
                result.append(slice_)
            return result

