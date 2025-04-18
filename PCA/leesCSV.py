import pandas as pd
from pprint import pp
import warnings
import statistics

warnings.filterwarnings("ignore")

data = pd.read_csv("gegevens.csv")

methode = "SVD"
ds = "ACM"

methodes = ["wKSVD","SVD","KPCA","KSVD"]
dss = ["ACM","DBLP","wiki"]



metrs = [ "avgNMI", "avgPMI", "avgF1", "avgCA", "avgARI" ]

for metr in metrs:
    h = [""] + dss
    uit = [h]
    for methode in methodes:
        ll = [methode]
        for ds in dss:
            td = data[data["alg"] == methode][data["data"] == ds]
            tdl = td[td["redenCl"] == "doc"]
            gmm = statistics.median(list(tdl["gamma"]))
            tdl = td[td["gamma"] == gmm]
            h = max(list(tdl[metr]))
            ll += [h]
        uit += [ll]

    print(metr)
    for i in uit:
        pp(i)
    print()
