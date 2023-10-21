import numpy as np

# Vs1 = -10 + 0.9(0.9 * Vs2 + 0.1Vs1 )
#     = -10 + 0.81 Vs2 + 0.09 Vs1
#  10 = 0.81 Vs2 + (0.09 - 1) Vs1


# Vs2 = 10 + 0.9(0.8 * Vs2 + 0.2Vs1 )
# Vs2 = 10 + 0.72Vs2 + 0.18Vs1
# -10 = (0.72 - 1) Vs2 + 0.18Vs1


A = np.array( [[(0.09 - 1), 0.81],
               [0.18, (0.72 - 1)]] )

b = np.array( [[ 10],
               [-10]] )

print(A)
print(b)

r = np.linalg.solve(A, b)

print(r)

vs1 = r[0, 0]
vs2 = r[1, 0]

q_faim_manger = 0.9 * vs2 + 0.1 * vs1
q_faim_tv = vs1
q_satis_dormir = 0.8 * vs2 + 0.2 * vs1
q_satis_exercice = vs1

print(q_faim_manger, q_faim_tv, q_satis_dormir, q_satis_exercice)





