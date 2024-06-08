import random
class Helper:
    @staticmethod
    def dot_product(matrix1, matrix2):
        # Check if dimensions are compatible for dot product
        if len(matrix1[0]) != len(matrix2):
            raise ValueError("Matrix dimensions are not compatible for dot product")

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
        # Check if dimensions are compatible for addition
        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            raise ValueError("Matrix dimensions are not compatible for addition")

        result = []
        for i in range(len(matrix1)):
            row = []
            for j in range(len(matrix1[0])):
                row.append(matrix1[i][j] + matrix2[i][j])
            result.append(row)
        return result
    @staticmethod
    def transpose(matrix):
        # Transpose the matrix
        result = []
        for j in range(len(matrix[0])):
            row = []
            for i in range(len(matrix)):
                row.append(matrix[i][j])
            result.append(row)
        return result
    @staticmethod
    def print_3d_matrix(matrix):
        depth = len(matrix)
        height = len(matrix[0]) 
        width = len(matrix[0][0])
        print(f"Dimension: Depth: {depth}, Height: {height}, Width: {width}")
        for d in range(len(matrix)):
            print("    [")
            for h in range(len(matrix[d])):
                row = ', '.join(f"{value:.2f}" for value in matrix[d][h])
                print(f"        [{row}],")
            if d < len(matrix) - 1:
                print("    ],")
            else:
                print("    ]")


    @staticmethod
    def apply_function(func, matrix):
        if isinstance(matrix[0][0], list):  # Check if the matrix has depth
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
    """
    @staticmethod
    def apply_function(func, matrix):
        if isinstance(matrix[0][0], list):  # Check if the matrix has depth
            return [[[func(matrix[d][i][j]) for j in range(len(matrix[0][0]))] for i in range(len(matrix[0]))] for d in range(len(matrix))]
        else:
            return [[func(matrix[i][j]) for j in range(len(matrix[0]))] for i in range(len(matrix))]
    """
    @staticmethod
    def elementwise_multiply(matrix1, matrix2):
        if isinstance(matrix1[0][0], (int, float)):  # Check if matrix1 and matrix2 are 2D matrices
            if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
                raise ValueError("Matrices must have the same dimensions for elementwise multiplication.")
            result = []
            for i in range(len(matrix1)):
                row = []
                for j in range(len(matrix1[0])):
                    row.append(matrix1[i][j] * matrix2[i][j])
                result.append(row)
            return result
        elif isinstance(matrix1[0][0], list):  # Check if matrix1 and matrix2 are 3D tensors
            if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]) or len(matrix1[0][0]) != len(matrix2[0][0]):
                raise ValueError("Tensors must have the same dimensions for elementwise multiplication.")
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
        else:
            raise ValueError("Unsupported input types. matrix1 and matrix2 should either be normal matrices or tensors.")

