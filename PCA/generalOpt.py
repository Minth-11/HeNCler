import os
import scipy as sp
import numpy as np
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
# from sklearn.metrics import f1_score as f1
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh
from sklearn.metrics import adjusted_rand_score
import sklearn.decomposition as skd
import cupy.linalg as cla
import cupy as cp
import pandas as pd
import gc
from metrics import average_pmi_per_cluster
from sklearn.metrics import cluster

clear = lambda: os.system('clear')

pd.set_option("display.max_rows", 100000)

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

def pairwise_accuracy(y_true, y_pred):
    """Computes pairwise accuracy of two clusterings.

    Args:
      y_true: An (n,) int ground-truth cluster vector.
      y_pred: An (n,) int predicted cluster vector.

    Returns:
      Accuracy value computed from the true/false positives and negatives.
    """
    true_pos, false_pos, false_neg, true_neg = _pairwise_confusion(y_true, y_pred)
    return (true_pos + false_pos) / (true_pos + false_pos + false_neg + true_neg)

def _pairwise_confusion(
    y_true,
    y_pred):
  """Computes pairwise confusion matrix of two clusterings.

  Args:
    y_true: An (n,) int ground-truth cluster vector.
    y_pred: An (n,) int predicted cluster vector.

  Returns:
    True positive, false positive, true negative, and false negative values.
  """
  contingency = cluster.contingency_matrix(y_true, y_pred)
  same_class_true = np.max(contingency, 1)
  same_class_pred = np.max(contingency, 0)
  diff_class_true = contingency.sum(axis=1) - same_class_true
  diff_class_pred = contingency.sum(axis=0) - same_class_pred
  total = contingency.sum()

  true_positives = (same_class_true * (same_class_true - 1)).sum()
  false_positives = (diff_class_true * same_class_true * 2).sum()
  false_negatives = (diff_class_pred * same_class_pred * 2).sum()
  true_negatives = total * (
      total - 1) - true_positives - false_positives - false_negatives

  return true_positives, false_positives, false_negatives, true_negatives

def pairwise_precision(y_true, y_pred):
    """Computes pairwise precision of two clusterings.

    Args:
    y_true: An [n] int ground-truth cluster vector.
    y_pred: An [n] int predicted cluster vector.

    Returns:
    Precision value computed from the true/false positives and negatives.
    """
    true_positives, false_positives, _, _ = _pairwise_confusion(y_true, y_pred)
    return true_positives / (true_positives + false_positives)


def pairwise_recall(y_true, y_pred):
  """Computes pairwise recall of two clusterings.

  Args:
    y_true: An (n,) int ground-truth cluster vector.
    y_pred: An (n,) int predicted cluster vector.

  Returns:
    Recall value computed from the true/false positives and negatives.
  """
  true_positives, _, false_negatives, _ = _pairwise_confusion(y_true, y_pred)
  return true_positives / (true_positives + false_negatives)


def f1_score(y_true, y_pred):
    precision = pairwise_precision(y_true, y_pred)
    recall = pairwise_recall(y_true, y_pred)
    return 2 * precision * recall / (precision + recall)

def compaIze(RD,CD):
    if (RD.shape[1]<CD.shape[1]):
        X = RD @ np.random.randn(RD.shape[1], CD.shape[1])
        Y = CD
    else:
        X = RD
        Y = CD @ np.random.randn(CD.shape[1], RD.shape[1])
    return X, Y

def compaIzeKlein(RD,CD):
    if (RD.shape[1]<CD.shape[1]):
        X = RD
        Y = CD @ np.random.randn(CD.shape[1], RD.shape[1])
    else:
        X = RD @ np.random.randn(RD.shape[1], CD.shape[1])
        Y = CD
    return X, Y

def compaIzePCA(RD,CD):
    if (RD.shape[1]<CD.shape[1]):
        X = RD
        skpca = skd.PCA(n_components=RD.shape[1])
        cdd = np.asarray( CD.todense() )
        ft = skpca.fit(cdd)
        Y = ft.transform(cdd)
    else:
        skpca = skd.PCA(n_components=CD.shape[1])
        rdd = np.asarray( RD.todense() )
        ft = skpca.fit(rdd)
        X = ft.transform(rdd)
        Y = CD
    return X, Y

def nCl(i,d):
    if (i == "doc"):
        return docDict[d]
    if (i=="term"):
        return termDict[d]
    return i

gegevens = pd.DataFrame()
teller = 0

dataStrs = ["ACM","DBLP","wiki","PubMed"]
dataStrs = ["PubMed","wiki","DBLP","ACM"]
algs = ["wKSVD","KSVD","SVD","KPCA"]
#algs = ["wKSVD","KSVD"]
nClustersVs = ["doc","term", 8,16,32,64,128,256,512,1024]
# gams = [0.0001/64,0.0001/16,0.0001/4,0.0001,0.0004,0.0016,0.0064,0.0265,0.1024,0.4096,1.6384,3.2768,13.1072]
# gams = ["nonsense!"]
gamScale = [10**i for i in range(-4,7)]
ncp = [1000,"doc","term"]

compa = ["std","klein","pca"]
compa = ["pca"]

mxit =len(dataStrs) * len(algs) * len(nClustersVs) * len(gamScale) * len(ncp)
print("Experimenten: \t"+str(mxit))
for dataStr in dataStrs:
    for alg in algs:
        for nClustersV in nClustersVs:
            # for gam in gams:
            for gs in gamScale:
                for nc in ncp:
                    for cpi in compa:
                        teller+=1
                        nClusters = nCl(nClustersV,dataStr)

                        # FORMATTEERgams DATA
                        data = dataDict[dataStr]
                        features= data["fea"].astype(float)
                        if not sp.sparse.issparse(features):
                            features = sp.sparse.csr_matrix(features)

                        labels = data['gnd'].reshape(-1) - 1
                        n_classes = len(np.unique(labels))

                        npFeatures = features.todense()
                        gammaROT1 = np.mean(npFeatures.var(0))
                        gammaROT2 = np.mean(npFeatures.var(1))
                        gammmaROT = (gammaROT1 + gammaROT2)/2

                        gam = gammmaROT * gs

                        n_components = nc
                        if nc == "doc":
                            n_components = docDict[dataStr]
                        if nc == "term":
                            n_components = termDict[dataStr]


                        # VIND EMBEDDING
                        if (alg=="KSVD"):
                            RowData = features
                            ColData = features.transpose()
                            if cpi == "std":
                                X, Y = compaIze(RowData,ColData)
                            if cpi == "pca":
                                X, Y = compaIzePCA(RowData,ColData)
                            if cpi == "klein":
                                X, Y = compaIzeKlein(RowData,ColData)
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

                        if (alg=="wKSVD"):
                            RowData = features
                            ColData = features.transpose()
                            if cpi == "std":
                                X, Y = compaIze(RowData,ColData)
                            if cpi == "pca":
                                X, Y = compaIzePCA(RowData,ColData)
                            if cpi == "klein":
                                X, Y = compaIzeKlein(RowData,ColData)
                            K = rbf_kernel(X,Y,gam) #NOTE: only positive kernel values are supported (e.g. between 0 and 1)
                            n, m = K.shape


                            d_i, d_j = K.sum(1), K.sum(0)
                            eps = 1e-3
                            d_i, d_j = np.clip(d_i, eps, None), np.clip(d_j, eps, None)

                            w_i, w_j = np.power(d_i, -1), np.power(d_j, -1)

                            Kc = (np.eye(n) - np.ones((n,n)) @ np.diag(w_i)/w_i.sum()) @ K @ (np.eye(m) - np.ones((m,m)) @ np.diag(w_j)/w_j.sum())

                            U, _ ,Vh = svd(Kc)
                            DocEmbedding = U[:,:min(n_components, n, m)]
                            TermEmbedding = Vh[:,:min(n_components, n, m)]
                            # DocEmbedding = U[:,:nClusters-1]
                            # TermEmbedding = Vh[:,:nClusters-1]

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

                        # VIND NMI en F1 (a.d.h.v. K-means)
                        NMIs = []
                        F1s = []
                        CAs = []
                        ARIs = []
                        for i in range(nruns):
                            kmeans = KMeans(n_clusters=nClusters,verbose=0,init='random',max_iter=10000)
                            kmeans.fit(DocEmbedding)
                            fts = kmeans.predict(DocEmbedding)
                            NMIs = NMIs + [nmi(labels, fts)]
                            F1s = F1s + [f1_score(labels, fts)]
                            CAs = CAs + [ pairwise_accuracy(labels, fts) ]
                            ARIs = ARIs + [ adjusted_rand_score(labels, fts) ]
                        avgNmi = np.mean(np.array(NMIs))
                        maxNmi = np.max(np.array(NMIs))
                        stdNmi = np.std( np.array(NMIs))
                        avgF1 = np.mean(np.array(F1s))
                        maxF1 = np.max(np.array(F1s))
                        stdF1 = np.std( np.array(F1s))
                        avgCA =  np.mean(np.array(CAs))
                        maxCA = np.max(np.array(CAs))
                        stdCA =  np.std(np.array(CAs))
                        avgARI =  np.mean(np.array(ARIs))
                        maxARI = np.max(np.array(ARIs))
                        stdARI =  np.std(np.array(ARIs))

                        PMIs = []
                        if TermEmbedding.shape[0] > nClusters:
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
                        else:
                            avgPmi = float('nan')
                            stdPmi = float('nan')

                        reden = ''
                        if nClustersV == "doc":
                            reden="doc"
                        elif nClustersV == "term":
                            reden="term"
                        else:
                            reden="sweep"

                        ldta =  {   "pct":round(100*teller/mxit,2)
                                ,   "alg":alg
                                ,   "compaIze":cpi
                                ,   "data":dataStr
                                ,   "nClusters":nClusters
                                ,   "nComponents":n_components
                                ,   "nClusters":nc
                                ,   "redenCl":reden
                                ,   "gamma":gam
                                ,   "gammaScale":gs
                                ,   "maxNMI":maxNmi
                                ,   "avgNMI":avgNmi
                                ,   "stdNMI":stdNmi
                                ,   "avgPMI":avgPmi
                                ,   "stdPMI":stdPmi
                                ,   "maxF1":maxF1
                                ,   "avgF1":avgF1
                                ,   "stdF1":stdF1
                                ,   "maxCA":maxCA
                                ,   "avgCA":avgCA
                                ,   "stdCA":stdCA
                                ,   "maxARI":maxARI
                                ,   "avgARI":avgARI
                                ,   "stdARI":stdARI} # was stdCA
                        gegevens = gegevens._append(ldta,ignore_index=True)

                        print()
                        clear()
                        print(gegevens)
                        gegevens.to_csv("gegevens0003.csv")




