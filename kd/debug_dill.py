import dill
import os

for i in range(10000):
    krigefile = os.path.join(os.path.dirname(__file__), "cw21_kde_krige.pkl")
    with open(krigefile, "rb") as f:
        file = dill.load(f)
        krige = file["krige"]
        Upec_var_threshold = file["Upec_var_threshold"]
        Vpec_var_threshold = file["Vpec_var_threshold"]
        file = None
    _, *_ = krige(4,5)
    if i % 100 == 0:
        print(i)
    krige = Upec_var_threshold = Vpec_var_threshold = None

print("Done")