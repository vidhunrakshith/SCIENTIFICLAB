import numpy as np

def full_qr_via_gram_schmidt(A, tol=1e-10):
    m, n = A.shape
    
    # Step 1: Initialize Q (m x m) and R (m x n)
    Q = np.zeros((m, m))
    R = np.zeros((m, n))
    
    # Step 2: Classical Gram-Schmidt for first n columns
    for j in range(n):
        v = A[:, j].copy()
        
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        
        R[j, j] = np.linalg.norm(v)
        
        if R[j, j] > tol:
            Q[:, j] = v / R[j, j]
    
    # Step 3: Complete Q to full orthonormal basis
    for j in range(n, m):
        v = np.zeros(m)
        v[j] = 1  # start with standard basis vector
        
        for i in range(j):
            v = v - np.dot(Q[:, i], v) * Q[:, i]
        
        norm = np.linalg.norm(v)
        if norm > tol:
            Q[:, j] = v / norm
    
    return Q, R


# ---------------- USER INPUT ----------------

print("Full QR Decomposition using Gram-Schmidt")
m = int(input("Enter number of rows (m): "))
n = int(input("Enter number of columns (n): "))

print("Enter the matrix row by row (space-separated):")
A = np.zeros((m, n))
for i in range(m):
    A[i] = list(map(float, input(f"Row {i+1}: ").split()))

# Compute Full QR
Q, R = full_qr_via_gram_schmidt(A)

# Output
print("\nMatrix A:")
print(A)

print("\nMatrix Q (m x m):")
print(Q)

print("\nMatrix R (m x n):")
print(R)

print("\nVerification (Q R):")
print(Q @ R)

