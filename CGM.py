import numpy as np

# ---------- INPUT ----------
n = int(input("Enter the size of the matrix (n): "))

print("Enter the matrix A row by row:")
A = []
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)
A = np.array(A)

# ---------- SPD CHECK USING NUMPY ----------

# Step 1: Check symmetry
if not np.allclose(A, A.T):
    print("\nMatrix is NOT symmetric.")
    print("Conjugate Gradient method cannot be applied.")
    exit()

# Step 2: Compute eigenvalues
eigenvalues = np.linalg.eigvals(A)

print("\nEigenvalues of A:")
print(eigenvalues)

# Step 3: Check if all eigenvalues are positive
if not np.all(eigenvalues > 0):
    print("\nMatrix is NOT positive definite.")
    print("Conjugate Gradient method cannot be applied.")
    exit()

print("\nMatrix is Symmetric Positive Definite (SPD). Proceeding...\n")

# ---------- CG INPUT ----------
print("Enter the vector b:")
b = np.array(list(map(float, input().split())))

tol = float(input("Enter tolerance (e.g., 1e-6): "))
max_iter = int(input("Enter maximum iterations: "))

# ---------- CONJUGATE GRADIENT ----------
x = np.zeros(n)
r = b - A @ x
p = r.copy()

for k in range(max_iter):
    if np.linalg.norm(r) < tol * np.linalg.norm(b):
        print(f"Converged at iteration {k}")
        break
    
    Apk = A @ p
    
    alpha = np.dot(r, r) / np.dot(p, Apk)
    
    x = x + alpha * p
    
    r_new = r - alpha * Apk
    
    beta = - np.dot(p, A @ r_new) / np.dot(p, Apk)
    
    p = r_new + beta * p
    
    r = r_new
    
    print(f"Iteration {k}: alpha={alpha:.6f}, beta={beta:.6f}, ||r||={np.linalg.norm(r):.6e}")

print("\nFinal solution x:")
print(x)
