import numpy as np
import matplotlib.pyplot as plt
import os

# -------- INPUT --------
n = int(input("Enter dimension n: "))

print("\nEnter matrix A row by row:")
A = []
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)
A = np.array(A)

print("\nEnter vector b:")
b = np.array(list(map(float, input().split())))

print("\nEnter initial point x0:")
x = np.array(list(map(float, input().split())))

max_iter = int(input("\nEnter maximum iterations: "))
tol = float(input("Enter tolerance (example 1e-6): "))

# -------- STEEPEST DESCENT --------
points = [x.copy()]

for k in range(max_iter):

    grad = A @ x - b
    p = -grad

    if np.linalg.norm(grad) < tol:
        print(f"Converged in {k} iterations")
        break

    alpha = (p @ p) / (p @ (A @ p))
    x = x + alpha * p

    points.append(x.copy())

points = np.array(points)

print("\nApproximate solution:", x)

# -------- PLOT --------
if n == 2:

    x_vals = np.linspace(-10, 10, 500)
    y_vals = np.linspace(-10, 10, 500)

    X, Y = np.meshgrid(x_vals, y_vals)

    Z = 0.5*(A[0,0]*X**2 + 2*A[0,1]*X*Y + A[1,1]*Y**2) - b[0]*X - b[1]*Y

    plt.contour(X, Y, Z, 30)
    plt.plot(points[:,0], points[:,1], 'ro-', label="Iterations")

    plt.title("Steepest Descent Convergence")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_dir, "steepest_descent_plot.png")
    
    plt.savefig(filename)
    plt.show()

else:
    plt.plot(range(len(points)))
    plt.title("Iterations completed")
    plt.show()
