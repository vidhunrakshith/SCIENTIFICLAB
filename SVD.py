import numpy as np

def svd_via_ata_no_svd(A, tol=1e-10):
    m, n = A.shape

    # Step 1: A^T A
    ATA = A.T @ A

    # Step 2: Eigenvalues and eigenvectors of A^T A
    eigenvalues, eigenvectors = np.linalg.eigh(ATA)

    # Step 3: Keep positive eigenvalues (rank r)
    positive = eigenvalues > tol
    lambdas = eigenvalues[positive]
    V_r = eigenvectors[:, positive]

    # Step 4: Singular values
    sigmas = np.sqrt(lambdas)

    # Step 5: Sort in descending order
    idx = np.argsort(sigmas)[::-1]
    sigmas = sigmas[idx]
    V_r = V_r[:, idx]

    r = len(sigmas)

    # Step 6: Normalize right singular vectors
    for i in range(r):
        V_r[:, i] /= np.linalg.norm(V_r[:, i])

    # Step 7: Complete V using remaining eigenvectors of A^T A
    V_null = eigenvectors[:, ~positive]
    V = np.hstack((V_r, V_null))

    # Step 8: Sigma matrix (m x n)
    Sigma = np.zeros((m, n))
    for i in range(r):
        Sigma[i, i] = sigmas[i]

    # Step 9: Left singular vectors u_i = A v_i / sigma_i
    U_r = np.zeros((m, r))
    for i in range(r):
        U_r[:, i] = (A @ V_r[:, i]) / sigmas[i]

    # Step 10: Complete U using eigenvectors of A A^T
    AAT = A @ A.T
    eigvals_u, eigvecs_u = np.linalg.eigh(AAT)

    zero = eigvals_u < tol
    U_null = eigvecs_u[:, zero]

    U = np.hstack((U_r, U_null))

    return U, Sigma, V.T


# ---------------- USER INPUT ----------------

print("Singular Value Decomposition using A^T A (no built-in SVD)")
m = int(input("Enter number of rows (m): "))
n = int(input("Enter number of columns (n): "))

print("Enter the matrix row by row (space-separated):")
A = np.zeros((m, n))
for i in range(m):
    A[i] = list(map(float, input(f"Row {i+1}: ").split()))

# Compute SVD
U, Sigma, Vt = svd_via_ata_no_svd(A)

# Output
print("\nMatrix A:")
print(A)

print("\nMatrix U:")
print(U)

print("\nMatrix Sigma:")
print(Sigma)

print("\nMatrix V^T:")
print(Vt)

print("\nVerification (U Σ V^T):")
print(U @ Sigma @ Vt)

