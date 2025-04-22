import pandas as pd

data = pd.read_csv("gegevens0003.csv")

df = data

dfs =   { "PubMed":df[df['data'] == "PubMed"]
        , "wiki":  df[df['data'] == "wiki"]
        , "ACM":   df[df['data'] == "ACM"]
        , "DBLP":  df[df['data'] == "DBLP"] }

paperKSVDNMI =  { "PubMed": 0.33
                , "wiki":   0.48
                , "ACM":    0.68
                , "DBLP":   0.28 }

for i in ["PubMed","wiki","ACM","DBLP"]:
    print(i+', max. NMI')
    td = df[df['alg']=="KSVD"]
    print("Qinghua  KSVD:\t"+str(paperKSVDNMI[i]))
    td = df[df['alg']=="KSVD"]
    td = td[td['data']==i]
    h = td.loc[td['avgNMI'].idxmax()]
    print("HeNCLer  KSVD:\t"+str(h["avgNMI"]))
    td = df[df['alg']=="wKSVD"]
    td = td[td['data']==i]
    h = td.loc[td['avgNMI'].idxmax()]
    print("HeNCLer wKSVD:\t"+str(h["avgNMI"]))
    print()

# for i in ["PubMed","wiki","ACM","DBLP"]:
    # print(i)
    # print("max NMI KSVD")
    # td = df[df['alg']=="KSVD"]
    # print(td.loc[td['avgNMI'].idxmax()])
    # print("max NMI wKSVD")
    # td = df[df['alg']=="wKSVD"]
    # print(td.loc[td['avgNMI'].idxmax()])
    # print()
    # print()
