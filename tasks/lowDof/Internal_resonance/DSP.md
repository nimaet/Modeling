# Time-Varying Damped Oscillator as a Digital Filter

We consider the weakly damped oscillator

$$
m\ddot{x}(t) + c\dot{x}(t) + k(t)x(t) = 0
$$

with:
- mass \(m\)
- damping \(c\)
- time-varying stiffness \(k(t)\)

Define the instantaneous natural frequency

$$
\omega_n(t) = \sqrt{\frac{k(t)}{m}}
$$

and damping ratio

$$
\zeta(t) = \frac{c}{2\sqrt{mk(t)}}
$$

---

# 1. Bilinear Transform (Tustin) Method

## Continuous-Time Transfer Function

Assuming locally constant stiffness over one sample:

$$
H(s) = \frac{1}{ms^2 + cs + k}
$$

Normalize:

$$
H(s) = \frac{1/m}{s^2 + \frac{c}{m}s + \frac{k}{m}}
$$

Define:

$$
a = \frac{c}{m}
$$

$$
b = \frac{k}{m}
$$

Then:

$$
H(s) = \frac{1/m}{s^2 + as + b}
$$

---

## Bilinear Transform

Apply:

$$
s \rightarrow \frac{2}{T}\frac{1-z^{-1}}{1+z^{-1}}
$$

where:
- \(T\) = sample period
- \(f_s = 1/T\)

Define:

$$
\alpha = \frac{2}{T}
$$

Then:

$$
s = \alpha\frac{1-z^{-1}}{1+z^{-1}}
$$

Substitute into the denominator:

$$
s^2 + as + b
$$

giving

$$
\alpha^2\left(\frac{1-z^{-1}}{1+z^{-1}}\right)^2
+
a\alpha\left(\frac{1-z^{-1}}{1+z^{-1}}\right)
+
b
$$

Multiply by \((1+z^{-1})^2\):

$$
\alpha^2(1-z^{-1})^2
+
a\alpha(1-z^{-1})(1+z^{-1})
+
b(1+z^{-1})^2
$$

Expand:

$$
(1-z^{-1})^2 = 1 - 2z^{-1} + z^{-2}
$$

$$
(1-z^{-1})(1+z^{-1}) = 1 - z^{-2}
$$

$$
(1+z^{-1})^2 = 1 + 2z^{-1} + z^{-2}
$$

Thus:

$$
A_0 + A_1 z^{-1} + A_2 z^{-2}
$$

with

$$
A_0 = \alpha^2 + a\alpha + b
$$

$$
A_1 = -2\alpha^2 + 2b
$$

$$
A_2 = \alpha^2 - a\alpha + b
$$

---

## Discrete-Time Difference Equation

The resulting filter is

$$
y[n]
=
-b_1 y[n-1]
-b_2 y[n-2]
+
g x[n]
+
2g x[n-1]
+
g x[n-2]
$$

where

$$
b_1 = \frac{A_1}{A_0}
$$

$$
b_2 = \frac{A_2}{A_0}
$$

$$
g = \frac{1/m}{A_0}
$$

---

## Time-Varying Stiffness

For time-varying stiffness:

$$
k \rightarrow k[n]
$$

so coefficients become:

$$
A_0[n] = \alpha^2 + a\alpha + \frac{k[n]}{m}
$$

$$
A_1[n] = -2\alpha^2 + 2\frac{k[n]}{m}
$$

$$
A_2[n] = \alpha^2 - a\alpha + \frac{k[n]}{m}
$$

Update coefficients each sample.

This yields a time-varying IIR resonator.

---

# 2. State-Transition Exponential Method

## State-Space Form

Define state:

$$
\mathbf{x}(t)
=
\begin{bmatrix}
x(t) \\
v(t)
\end{bmatrix}
$$

with velocity

$$
v(t) = \dot{x}(t)
$$

Then:

$$
\dot{\mathbf{x}}(t)
=
A(t)\mathbf{x}(t)
$$

where

$$
A(t)
=
\begin{bmatrix}
0 & 1 \\
-\frac{k(t)}{m} & -\frac{c}{m}
\end{bmatrix}
$$

---

## Exact Local Discretization

Assume \(A(t)\) constant over one sample interval:

$$
A_n = A(nT)
$$

Then the exact update is

$$
\mathbf{x}_{n+1}
=
e^{A_nT}\mathbf{x}_n
$$

where

$$
\Phi_n = e^{A_nT}
$$

is the state-transition matrix.

Thus:

$$
\mathbf{x}_{n+1}
=
\Phi_n \mathbf{x}_n
$$

---

# Closed-Form Matrix Exponential

Define:

$$
\omega_n = \sqrt{\frac{k_n}{m}}
$$

and

$$
\gamma = \frac{c}{2m}
$$

The damped frequency is

$$
\omega_d
=
\sqrt{\omega_n^2 - \gamma^2}
$$

For the underdamped case:

$$
\Phi_n
=
e^{-\gamma T}
\begin{bmatrix}
\cos(\omega_dT)
+
\frac{\gamma}{\omega_d}\sin(\omega_dT)
&
\frac{1}{\omega_d}\sin(\omega_dT)
\\
-\frac{\omega_n^2}{\omega_d}\sin(\omega_dT)
&
\cos(\omega_dT)
-
\frac{\gamma}{\omega_d}\sin(\omega_dT)
\end{bmatrix}
$$

---

# Recursive Update Equations

The update becomes

$$
x_{n+1}
=
\phi_{11}x_n
+
\phi_{12}v_n
$$

$$
v_{n+1}
=
\phi_{21}x_n
+
\phi_{22}v_n
$$

where:

$$
\Phi_n
=
\begin{bmatrix}
\phi_{11} & \phi_{12} \\
\phi_{21} & \phi_{22}
\end{bmatrix}
$$

and all coefficients vary with \(k_n\).

---

# Notes

## Bilinear Method
Advantages:
- simple biquad implementation
- unconditional stability
- standard DSP structure

Disadvantages:
- frequency warping
- not structure-preserving

---

## Exponential Method
Advantages:
- exact local solution
- excellent modal accuracy
- preserves damping/frequency behavior
- ideal for resonators

Disadvantages:
- more expensive coefficient computation
- requires trig/exponential evaluation

---

# Practical Recommendation

For real-time DSP:
- use bilinear transform for simplicity
- use exponential integrator for highest resonator quality

For rapidly modulated stiffness:
- smooth coefficient interpolation is important
- abrupt coefficient jumps can inject energy