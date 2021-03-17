import dill
import os
import time
import numpy as np
start = time.time()
krigefile = os.path.join(os.path.dirname(__file__), "cw21_kde_krige.pkl")
with open(krigefile, "rb") as f:
    file = dill.load(f)
    krige = file["krige"]
    Upec_var_threshold = file["Upec_var_threshold"]
    Vpec_var_threshold = file["Vpec_var_threshold"]
    file = None
x = np.random.normal(loc=4, scale=1, size=100)
y = np.random.normal(loc=5, scale=1, size=100)
for i in range(100):
    _, *_ = krige(x,y)
    # if i % 100 == 0:
    #     print(i)
    print(i)
    # krige = Upec_var_threshold = Vpec_var_threshold = None
end = time.time()
print("Done")
print(end-start)