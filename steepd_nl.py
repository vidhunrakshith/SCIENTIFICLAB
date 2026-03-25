import numpy as np

# -------- INPUT --------
n = int(input("Enter number of variables: "))

print("Enter function f(x1,...,xn) using x[i]:")
f_str = input()

def f(x):
    return eval(f_str)

# -------- NUMERICAL GRADIENT --------
def grad_f(x, h=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += h
        x2[i] -= h
        g[i] = (f(x1) - f(x2)) / (2*h)
    return g

# -------- phi'(alpha) --------
def phi_prime(x, p, alpha):
    return np.dot(grad_f(x + alpha*p), p)

# -------- phi''(alpha) --------
def phi_double_prime(x, p, alpha, h=1e-5):
    phi = lambda a: f(x + a*p)
    return (phi(alpha + h) - 2*phi(alpha) + phi(alpha - h)) / (h**2)

# -------- FIND ALPHA USING NEWTON --------
def find_alpha(p, x):
    y = 0.0   # initial alpha (as in board)
    TOL = 1e-6
    
    for i in range(1000):
        y_old = y
        
        d1 = phi_prime(x, p, y)
        d2 = phi_double_prime(x, p, y)
        
        if abs(d2) < 1e-10:
            break
        
        # Newton update
        y = y - d1 / d2
        
        # stopping for Newton
        if abs(y - y_old) / max(1, abs(y)) < TOL:
            break
    
    return y   # this is alpha_k

# -------- INITIAL GUESS --------
print("Enter initial guess:")
x_old = np.array(list(map(float, input().split())))

TOL = float(input("Enter tolerance (e.g., 1e-6): "))
MaxIter = 1000

# -------- MAIN LOOP --------
for i in range(MaxIter):
    
    # Step 1: pk = -∇f(xk)
    pk = -grad_f(x_old)
    
    # Step 2: αk = findalpha(pk, xk, f)
    alpha_k = find_alpha(pk, x_old)
    
    # Step 3: x = x_old + αk pk
    x = x_old + alpha_k * pk
    
    print(f"Iter {i}: x = {x}, f(x) = {f(x):.6f}, alpha = {alpha_k:.6f}")
    
    # -------- STOPPING CRITERIA --------
    
    # (i) function change
    cond1 = abs(f(x) - f(x_old)) / max(1, abs(f(x_old))) < TOL
    
    # (ii) step change
    cond2 = np.linalg.norm(x - x_old) / max(1, np.linalg.norm(x)) < TOL
    
    if cond1 or cond2:
        print("\nConverged")
        break
    
    # update
    x_old = x

# -------- OUTPUT --------
print("\nMinimum point:", x)
print("Minimum value:", f(x))
