import numpy as np
import matplotlib.pyplot as plt


flow = np.array([
    2704899, 3105421, 1705109, 3149538, 1899785, 2192683,
    2986978, 1827716, 3083510, 1813893, 2296806, 3035596,
    2866723, 1701996, 2832348, 2978316, 2094881, 2598303,
    2279577, 1890624, 2689544, 2468717, 2914777, 2833078,
    2203512, 1337041, 2106024, 2027347, 1118162, 1700049,
    2400749, 1560946, 2575401, 1859169, 1442183, 1821173,
    2059865, 1989162, 1639846, 1878229, 1700540, 2408338,
    2044065, 2190356, 1658232, 2249649, 2872648, 1894235,
    1056160, 1413513, 1883692, 3021141, 2063239, 1715627,
    1995966, 1501497, 2836282, 1311121, 1473949, 2490962,
    1329035, 1737860, 1854468, 1943841, 2408563, 2525583,
    2008769, 2380085, 2365929, 2172743, 1697608, 1064570,
    2465552, 2424480, 2236322, 1273686, 2184465, 2964330,
    3447166, 2707678, 2787777, 1639881, 1908933, 1556879,
    1490701, 1883312, 1596099, 2463142, 1596736, 2491106,
    2507554, 2908217, 1959082, 2076743, 1841910, 1592335,
    894755, 1972020, 1296720, 1917613, 2055741, 1859322,
    2545406, 2501931, 2047398, 3521671, 1257761, 1684747,
    2805866, 2292377, 2101271, 2162486, 1528744, 2498397
], dtype=float)


N = len(flow)
t = np.arange(1, N + 1)

print("Number of data points =", N)
print("First year = 1906")
print("Last year  = 2019")


F = np.zeros(N, dtype=complex)


for k in range(N):

    for j in range(N):

        F[k] = F[k] + flow[j] * np.exp(
            -1j * 2 * np.pi * k * t[j] / N
        )


frequency = np.arange(N) / N


power = np.abs(F) ** 2


print("\n========================================")
print("FOURIER COEFFICIENTS")
print("========================================")

for k in range(N):

    print(
        "k = {:3d}, f = {:.6f}, "
        "F = {:.4e} + {:.4e}i".format(
            k,
            frequency[k],
            F[k].real,
            F[k].imag
        )
    )



k_plot = np.arange(0, N // 2 + 1)

frequency_plot = frequency[k_plot]

power_plot = power[k_plot]


plt.figure(figsize=(9, 6))

plt.plot(
    frequency_plot,
    power_plot
)

plt.xlabel("Frequency (cycles/year)")
plt.ylabel("Power = |F(k)|^2")

plt.title("Power vs Frequency")

plt.grid(True)

plt.tight_layout()

plt.show()


candidate = np.arange(1, N // 2 + 1)

sorted_k = candidate[
    np.argsort(power[candidate])[::-1]
]

top3 = sorted_k[:3]


print("\n========================================")
print("THREE MOST DOMINANT FREQUENCIES")
print("========================================")


for k in top3:

    f = frequency[k]

    period = 1 / f

    print("\n----------------------------")

    print("k =", k)

    print(
        "Frequency = {:.9f} cycles/year".format(f)
    )

    print(
        "Period = {:.6f} years".format(period)
    )

    print(
        "Power = {:.6e}".format(power[k])
    )


plt.figure(figsize=(9, 6))

plt.plot(
    frequency_plot,
    power_plot,
    label="Power spectrum"
)

plt.scatter(
    frequency[top3],
    power[top3],
    zorder=3,
    label="Three dominant frequencies"
)


plt.xlabel("Frequency (cycles/year)")
plt.ylabel("Power = |F(k)|^2")

plt.title("Power Spectrum of Annual River Flow")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()