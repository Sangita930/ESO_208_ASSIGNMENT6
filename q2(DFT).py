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

print("Number of observations =", N)
print("First year = 1906")
print("Last year  = 2019")


A0 = np.mean(flow)

print("\nMean annual flow (A0) =", A0)



k_values = np.arange(1, N // 2 + 1)

frequency = []
period = []
A = []
B = []
power = []


for k in k_values:

    theta = 2 * np.pi * k * t / N

   
    Ak = (2 / N) * np.sum(flow * np.cos(theta))

    Bk = (2 / N) * np.sum(flow * np.sin(theta))

    f = k / N

    T = 1 / f

    P = Ak**2 + Bk**2

    frequency.append(f)
    period.append(T)
    A.append(Ak)
    B.append(Bk)
    power.append(P)

frequency = np.array(frequency)
period = np.array(period)
A = np.array(A)
B = np.array(B)
power = np.array(power)


print("\nFourier Analysis Results")
print("-" * 80)

print("   k        Frequency        Period        A_k        B_k        Power")
print("-" * 80)

for i in range(len(k_values)):
    print(
        f"{k_values[i]:4d}   "
        f"{frequency[i]:12.4f}   "
        f"{period[i]:10.2f}   "
        f"{A[i]:10.2f}   "
        f"{B[i]:10.2f}   "
        f"{power[i]:12.2e}"
    )


important_k = [10, 39, 2]


print("\nImportant Frequencies")
print("-" * 50)

for k in important_k:

    index = k - 1

    print(
        f"k = {k:2d}   "
        f"Frequency = {frequency[index]:.4f} cycles/year   "
        f"Period = {period[index]:.2f} years   "
        f"Power = {power[index]:.4e}"
    )




plt.figure(figsize=(10, 6))

plt.plot(
    frequency,
    power,
    linewidth=1.5
)

plt.scatter(
    frequency,
    power,
    s=15
)

for k in important_k:

    index = k - 1

    f = frequency[index]
    P = power[index]
    T = period[index]

    plt.scatter(
        f,
        P,
        s=70
    )

    plt.annotate(
        f"f = {f:.4f}\nT ≈ {T:.1f} years",
        xy=(f, P),
        xytext=(10, 15),
        textcoords="offset points",
        fontsize=10,
        arrowprops=dict(arrowstyle="->")
    )



plt.xlabel("Frequency (cycles/year)", fontsize=12)

plt.ylabel(
    r"Power $(A_k^2 + B_k^2)$",
    fontsize=12
)

plt.title(
    "Power Spectrum of Annual River Flow (1906–2019)",
    fontsize=14
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()