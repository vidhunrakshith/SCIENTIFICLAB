import numpy as np

# Number of equations
n = int(input("Enter number of equations: "))

# Read matrix A
print("Enter coefficient matrix A:")
A = np.zeros((n, n))
for i in range(n):
    A[i] = list(map(float, input().split()))

# Read vector b  (FIXED)
print("Enter constant vector b:")
b = np.array(list(map(float, input().split())))

# Scaling factors
s = np.zeros(n)
for i in range(n):
    s[i] = max(abs(A[i]))

# Forward elimination with scaled pivoting
for k in range(n - 1):

    max_ratio = 0
    pivot_row = k
    for i in range(k, n):
        ratio = abs(A[i][k]) / s[i]
        if ratio > max_ratio:
            max_ratio = ratio
            pivot_row = i

    # Swap rows
    if pivot_row != k:
        A[[k, pivot_row]] = A[[pivot_row, k]]
        b[[k, pivot_row]] = b[[pivot_row, k]]
        s[k], s[pivot_row] = s[pivot_row], s[k]

    # Elimination
    for i in range(k + 1, n):
        factor = A[i][k] / A[k][k]
        for j in range(k, n):
            A[i][j] -= factor * A[k][j]
        b[i] -= factor * b[k]

# Back substitution
x = np.zeros(n)
for i in range(n - 1, -1, -1):
    total = 0
    for j in range(i + 1, n):
        total += A[i][j] * x[j]
    x[i] = (b[i] - total) / A[i][i]

# Output
print("\nSolution:")
for i in range(n):
    print(f"x{i+1} = {x[i]}")

