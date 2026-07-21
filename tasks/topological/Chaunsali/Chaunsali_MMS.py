import sympy as sp

eps, K, delta, gamma, Gamma, q, omega = sp.symbols(
	"eps K delta gamma Gamma q omega"
)

I = sp.I
r = sp.symbols("r")  # r represents exp(i theta)

X, Y, Xc, Yc = sp.symbols("X Y Xc Yc")
Xz, Yz, Xcz, Ycz = sp.symbols("Xz Yz Xcz Ycz")

em = sp.exp(-I*q)
ep = sp.exp(I*q)

u = X*r + Xc/r
v = Y*r + Yc/r

v_minus = em*(Y - eps*Yz)*r + ep*(Yc - eps*Ycz)/r
u_plus = ep*(X + eps*Xz)*r + em*(Xc + eps*Xcz)/r

Ru = (
	K*(1 - eps*delta)*(v_minus - u)
	+ eps*Gamma*(v_minus - u)**3
	+ K*(1 + eps*gamma)*(v - u)
	- eps*Gamma*(v - u)**3
)

Rv = (
	K*(1 + eps*gamma)*(u - v)
	- eps*Gamma*(u - v)**3
	+ K*(1 - eps*delta)*(u_plus - v)
	+ eps*Gamma*(u_plus - v)**3
)

def fundamental(expr):
	return sp.expand(expr).coeff(r, 1)

Ru_fund = sp.series(fundamental(Ru), eps, 0, 2).removeO().expand()
Rv_fund = sp.series(fundamental(Rv), eps, 0, 2).removeO().expand()

Ru_O1 = sp.expand(Ru_fund.coeff(eps, 0))
Ru_Oeps = sp.expand(Ru_fund.coeff(eps, 1))

Rv_O1 = sp.expand(Rv_fund.coeff(eps, 0))
Rv_Oeps = sp.expand(Rv_fund.coeff(eps, 1))

print("u equation O(1):")
print(Ru_O1)

print("\nu equation O(eps):")
print(Ru_Oeps)

print("\nv equation O(1):")
print(Rv_O1)

print("\nv equation O(eps):")
print(Rv_Oeps)