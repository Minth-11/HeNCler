import os
import scipy as sp
import numpy as np
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh
import cupy.linalg as cla
import cupy as cp
import pandas as pd
import gc
from metrics import average_pmi_per_cluster

dataPath = os.path.join("..", "BCOT-main", "data")
wikiPath = os.path.join(dataPath, "wiki.mat")
acmPath  = os.path.join(dataPath,  "acm.mat")
pubmedPath  = os.path.join(dataPath,  "pubmed.mat")
dblpPath  = os.path.join(dataPath, "dblp.mat")

wikiMat   = sp.io.loadmat(  wikiPath)
acmMat    = sp.io.loadmat(   acmPath)
pubmedMat = sp.io.loadmat(pubmedPath)
dblpMat   = sp.io.loadmat(  dblpPath)

dataDict = {"wiki":wikiMat,"ACM":acmMat, "DBLP":dblpMat, "PubMed":pubmedMat}
termDict = {"wiki":23,"ACM":18, "DBLP":2, "PubMed":3}
docDict  = {"wiki":17,"ACM":3, "DBLP":4, "PubMed":3}

n_components = 1000
nruns = 10

def compaIze(RD,CD):
    if (RD.shape[1]<CD.shape[1]):
        X = RD @ np.random.randn(RD.shape[1], CD.shape[1])
        Y = CD
    else:
        X = RD
        Y = CD @ np.random.randn(CD.shape[1], RD.shape[1])
    return X, Y

def nCl(i,d):
    if (i == "doc"):
        return docDict[d]
    if (i=="term"):
        return termDict[d]
    return i

gegevens = pd.DataFrame()
teller = -1
mxit = 3*4*10*10
for dataStr in ["wiki","DBLP","ACM","PubMed"]: # 4
    for alg in ["SVD","KPCA","KSVD"]: # 3
        for nClustersV in ["doc","term", 8,16,32,64,128,256,512,1024]: # 10
            for gam in [0.0001,0.0004,0.0016,0.0064,0.0265,0.1024,0.4096,1.6384,3.2768,13.1072]: #10 [0.0004]:#
                teller+=1
                nClusters = nCl(nClustersV,dataStr)

                # FORMATTEER DATA
                data = dataDict[dataStr]
                features= data["fea"].astype(float)
                if not sp.sparse.issparse(features):
                    features = sp.sparse.csr_matrix(features)

                labels = data['gnd'].reshape(-1) - 1
                n_classes = len(np.unique(labels))

                # VIND EMBEDDING
                if (alg=="KSVD"):
                    RowData = features
                    ColData = features.transpose()
                    X, Y = compaIze(RowData,ColData)
                    K = rbf_kernel(X,Y,gam)
                    n, m = K.shape
                    # K = K / K.sum(0) # lijkt te veel op weging...
                    K = (np.eye(n) - np.ones((n,n))/n) @ K @ (np.eye(m) - np.ones((m,m))/m)
                    U, _, Vh = svd(K)
                    DocEmbedding = U[:,:min(n_components, n, m)]
                    TermEmbedding = Vh[:,:min(n_components, n, m)]

                if (alg=="SVD"):
                    n, m = features.shape
                    K = features
                    Kc = (np.eye(n) - np.ones((n,n))/n) @ K @ (np.eye(m) - np.ones((m,m))/m)
                    U, _, Vh = svd(Kc)
                    DocEmbedding = U[:,:min(n_components, n, m)]
                    TermEmbedding = Vh[:,:min(n_components, n, m)]

                if (alg=="KPCA"):
                    X = features
                    # print("Kernelmatrix...",end="",flush=True)
                    K = rbf_kernel(X,gamma=gam)
                    # print("KLAAR",flush=True)
                    n = K.shape[0]
                    if dataStr == "PubMed":
                        # K = (np.eye(n) - np.ones((n,n))/n) @ K @ (np.eye(n) - np.ones((n,n))/n)
                        # U, _, _ = gpuSVD(K)
                        N = cp.ones((n,n))/n
                        N = cp.eye(n) - N
                        Kg = cp.asarray(K)
                        Kg = N @ Kg @ N
                        del N
                        gc.collect()
                        Ug, _, _ = cla.svd(Kg)
                        K = cp.asnumpy(Kg)
                        del Kg
                        gc.collect()
                        U = cp.asnumpy(Ug)
                        del Ug
                        gc.collect()
                    else:
                        K = (np.eye(n) - np.ones((n,n))/n) @ K @ (np.eye(n) - np.ones((n,n))/n)
                        U, _, _ = svd(K)
                    DocEmbedding = U[:,:min(n_components, n)]
                    X = features.transpose()
                    K = rbf_kernel(X,gamma=gam)
                    n = K.shape[0]
                    # K = (np.eye(n) - np.ones((n,n))/n) @ K @ (np.eye(n) - np.ones((n,n))/n)
                    if dataStr == "PubMed":
                        # K = (np.eye(n) - np.ones((n,n))/n) @ K @ (np.eye(n) - np.ones((n,n))/n)
                        # U, _, _ = gpuSVD(K)
                        N = cp.ones((n,n))/n
                        N = cp.eye(n) - N
                        Kg = cp.asarray(K)
                        Kg = N @ Kg @ N
                        del N
                        gc.collect()
                        # Kg = cp.asarray(K)
                        # Kg = (cp.eye(n) - cp.ones((n,n))/n) @ Kg @ (cp.eye(n) - cp.ones((n,n))/n)
                        Ug, _, _ = cla.svd(Kg)
                        K = cp.asnumpy(Kg)
                        del Kg
                        gc.collect()
                        U = cp.asnumpy(Ug)
                        del Ug
                        gc.collect()
                    else:
                        K = (np.eye(n) - np.ones((n,n))/n) @ K @ (np.eye(n) - np.ones((n,n))/n)
                        U, _, _ = svd(K)
                    TermEmbedding = U[:,:min(n_components, n)]

                # VIND NMI (a.d.h.v. K-means)
                NMIs = []
                for i in range(nruns):
                    kmeans = KMeans(n_clusters=nClusters,verbose=0)
                    kmeans.fit(DocEmbedding)
                    fts = kmeans.predict(DocEmbedding)
                    NMIs = NMIs + [nmi(labels, fts)]
                avgNmi = np.mean(np.array(NMIs))
                stdNmi = np.std( np.array(NMIs))

                PMIs = []
                # VIND PMI
                for i in range(nruns):
                    kmeans = KMeans(n_clusters=nClusters,verbose=0)
                    kmeans.fit(TermEmbedding)
                    fts = kmeans.predict(TermEmbedding)
                    with np.errstate(divide='ignore'):
                        PMIs = PMIs + [ average_pmi_per_cluster(features.T,fts) ]
                with np.errstate(divide='ignore'):
                    avgPmi = np.mean(np.array(PMIs))
                    stdPmi = np.std( np.array(PMIs))

                reden = ''
                if nClustersV == "doc":
                    reden="doc"
                elif nClustersV == "term":
                    reden="term"
                else:
                    reden="sweep"

                if (teller%20 == 0):
                    print()
                    print(" %\t alg\t data\t nClss\t reden\t gamma\t avgNMI\t\t\t stdNMI\t\t\t avgPMI\t\t\t stdPMI")
                print(str(round(1000*teller/mxit)/10)+"%",end='\t')
                print(alg,end='\t')
                print(dataStr,end='\t')
                print(nClusters,end='\t')
                print(reden,end='\t')
                print(gam,end='\t')
                print(avgNmi,end='\t')
                print(stdNmi,end='\t')
                print(avgPmi,end='\t')
                print(stdPmi,end='\t')
                print("",flush=True)

                ldta =  {   "alg":alg
                        ,   "data":dataStr
                        ,   "nClusters":nClusters
                        ,   "reden":reden
                        ,   "gamma":gam
                        ,   "avgNMI":avgNmi
                        ,   "stdNMI":stdNmi
                        ,   "avgPMI":avgPmi
                        ,   "stdPMI":stdPmi }
                gegevens = gegevens._append(ldta,ignore_index=True)

print(gegevens)
gegevens.to_csv("gegevens.csv")



















