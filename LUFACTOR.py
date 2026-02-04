def lu_factorization(A):
    n = len(A)

    # Initialize L and U
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        # Compute U
        for j in range(i, n):
            total = 0.0
            for k in range(i):
                total += L[i][k] * U[k][j]
            U[i][j] = A[i][j] - total

        # Compute L
        L[i][i] = 1.0
        for j in range(i + 1, n):
            total = 0.0
            for k in range(i):
                total += L[j][k] * U[k][i]
            L[j][i] = (A[j][i] - total) / U[i][i]

    return L, U


# ---- Main program ----
n = int(input("Enter the size of the square matrix: "))

A = []
print("Enter the matrix row by row:")
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)

L, U = lu_factorization(A)

print("\nMatrix L:")
for row in L:
    print(row)

print("\nMatrix U:")
for row in U:
    print(row)

