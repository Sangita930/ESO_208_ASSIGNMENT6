import numpy as np
import matplotlib.pyplot as plt

T = np.array([0, 8, 16, 24, 32, 40], dtype=float)
O = np.array([14.621, 11.843, 9.870, 8.418, 7.305, 6.413], dtype=float)

n = len(T)


DD = np.zeros((n, n))

DD[:, 0] = O

for j in range(1, n):
    for i in range(n - j):
        DD[i, j] = (
            DD[i + 1, j - 1] - DD[i, j - 1]
        ) / (T[i + j] - T[i])


coeff = DD[0, :]

print("\n========================================")
print("PART (a): Newton Interpolation")
print("========================================")

print("\nDivided Difference Table:")
print(DD)

print("\nNewton coefficients:")
for i in range(n):
    print("a{} = {:.12e}".format(i, coeff[i]))

def newton_polynomial(x):
    result = coeff[n - 1]

    for i in range(n - 2, -1, -1):
        result = coeff[i] + (x - T[i]) * result

    return result


print("\nNewton Polynomial:")
print("P(T) = {:.12f}".format(coeff[0]))

for i in range(1, n):
    print("     + ({:.12e}) * product".format(coeff[i]))


m = n - 1

A = []
B = []


for i in range(m):

    row = np.zeros(3 * m)


    row[3 * i] = 1

    A.append(row)
    B.append(O[i])


for i in range(m):

    h = T[i + 1] - T[i]

    row = np.zeros(3 * m)

    row[3 * i] = 1
    row[3 * i + 1] = h
    row[3 * i + 2] = h ** 2

    A.append(row)
    B.append(O[i + 1])


for i in range(m - 1):

    h = T[i + 1] - T[i]

    row = np.zeros(3 * m)

    row[3 * i + 1] = 1
    row[3 * i + 2] = 2 * h


    row[3 * (i + 1) + 1] = -1

    A.append(row)
    B.append(0)


row = np.zeros(3 * m)
row[2] = 1

A.append(row)
B.append(0)


A = np.array(A, dtype=float)
B = np.array(B, dtype=float)


print("\n========================================")
print("PART (c): Gaussian Elimination")
print("========================================")


N = len(B)


for k in range(N - 1):

    pivot_row = k + np.argmax(np.abs(A[k:, k]))

    if pivot_row != k:

        A[[k, pivot_row]] = A[[pivot_row, k]]
        B[[k, pivot_row]] = B[[pivot_row, k]]

    for i in range(k + 1, N):

        factor = A[i, k] / A[k, k]

        A[i, k:] = A[i, k:] - factor * A[k, k:]
        B[i] = B[i] - factor * B[k]

x = np.zeros(N)

for i in range(N - 1, -1, -1):

    x[i] = (
        B[i] - np.dot(A[i, i + 1:], x[i + 1:])
    ) / A[i, i]

spline_coeff = x.reshape(m, 3)


print("\nQuadratic Spline Coefficients:")
print("----------------------------------------")

for i in range(m):

    a = spline_coeff[i, 0]
    b = spline_coeff[i, 1]
    c = spline_coeff[i, 2]

    print(
        "S{}: a = {:.12f}, b = {:.12f}, c = {:.12f}".format(
            i + 1, a, b, c
        )
    )


def quadratic_spline(t):

    t = np.asarray(t, dtype=float)

    result = np.zeros_like(t)

    for i in range(m):

        left = T[i]
        right = T[i + 1]

        if i == m - 1:
            mask = (t >= left) & (t <= right)
        else:
            mask = (t >= left) & (t < right)

        dt = t[mask] - left

        a = spline_coeff[i, 0]
        b = spline_coeff[i, 1]
        c = spline_coeff[i, 2]

        result[mask] = (
            a
            + b * dt
            + c * dt ** 2
        )

    return result


print("\nQuadratic Spline Equations:")
print("----------------------------------------")

for i in range(m):

    a = spline_coeff[i, 0]
    b = spline_coeff[i, 1]
    c = spline_coeff[i, 2]

    print(
        "\nFor {} <= T <= {}:".format(
            int(T[i]), int(T[i + 1])
        )
    )

    print(
        "S{}(T) = {:.12f} + ({:.12f})(T-{:.0f}) + "
        "({:.12f})(T-{:.0f})^2".format(
            i + 1,
            a,
            b,
            T[i],
            c,
            T[i]
        )
    )



T_plot = np.linspace(0, 40, 500)

O_newton = newton_polynomial(T_plot)

O_spline = quadratic_spline(T_plot)


plt.figure(figsize=(9, 6))

plt.plot(
    T_plot,
    O_newton,
    label="5th-order Newton polynomial"
)

plt.plot(
    T_plot,
    O_spline,
    label="Quadratic spline"
)

plt.scatter(
    T,
    O,
    label="Given data points",
    zorder=3
)

plt.xlabel("Temperature, T (°C)")
plt.ylabel("Dissolved Oxygen, O (mg/L)")

plt.title(
    "Dissolved Oxygen vs Temperature"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()