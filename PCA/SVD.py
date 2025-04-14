import os
import scipy
import zipfile
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans
from bayes_opt import BayesianOptimization
from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from scipy.linalg import svd

# inlezen data van BCOT

dataPath = os.path.join("..", "BCOT-main", "data")
wikiPath = os.path.join(dataPath, "wiki.mat")
acmPath  = os.path.join(dataPath,  "acm.mat")
pubmedPath  = os.path.join(dataPath,  "pubmed.mat")
dblpPath  = os.path.join(dataPath, "dblp.mat")

wikiMat   = scipy.io.loadmat(  wikiPath)
acmMat    = scipy.io.loadmat(   acmPath)
pubmedMat = scipy.io.loadmat(pubmedPath)
dblpMat   = scipy.io.loadmat(  dblpPath)

#print(wikiMat["fea"]) # features # shape (2405, 4973)
#print(wikiMat["gnd"]) # labels
#print(wikiMat["W"]) # ???

data = pubmedMat

ii=0
l = ["Wiki","ACM", "DBLP", "PubMed"]
ll = [30,70,35,8]
nclstrs = [17,3,4,3]
for data in [wikiMat, acmMat,dblpMat,pubmedMat]:
    print()
    print()
    print("\t//////////////// "+((ii+1)*"I")+". "+l[ii]+" ////////////////")
    print()
    print()
    ii += 1
    # acm gamma rond 0.000417022004702574
    # pubmed gamma rond 0.000417022004702574
    # dblp gamma rond 0.000417022004702574

    features= data["fea"].astype(float)
    if not sp.sparse.issparse(features):
        features = sp.sparse.csr_matrix(features)
    labels = data['gnd'].reshape(-1) - 1 # [0...16]
    n_classes = len(np.unique(labels))

    RowData = features#.transpose()
    print("Shape features:\t"+str(features.shape))
    ColData = features.transpose()

    # print(min(labels))

    kleinste = min(features.shape)
    grootste = max(features.shape)

    # compabiliteitsmatrix
    C = np.random.rand(kleinste,grootste)

    def prt(x):
        print(x,end='',flush=True)

    def toOpt(
        gam,
        n):
        nclstrs = 17
        nclstrs = round(n)

        kernel='rbf'
        gamma = gam
        #gamma=None
        n_components = 1000

        n = kleinste
        m = grootste

        #todo: transformeer de data a.d.h.v. KPCA
        Y=(ColData @ C)
        X= RowData
        #print(X.shape)
        #print(Y.shape)
        #print(5*'\n',flush=True)
        K = X @ Y
        Kc = (np.eye(n) - np.ones((n,n))/n) @ K @ (np.eye(m) - np.ones((m,m))/m)
        U, _, Vh = svd(Kc)

        colEmbedding = U[:,:1000]

        NMIs = []
        for i in tqdm(range(10)):
            kmeans = KMeans(n_clusters=nclstrs,verbose=0)
            kmeans.fit(colEmbedding)
            fts = kmeans.predict(colEmbedding)
            NMIs = NMIs + [nmi(labels, fts)]

        #print(NMIs)
        avgNmi = np.mean(np.array(NMIs))
        stdNmi = np.std( np.array(NMIs))
        print()
        print( "gamma:\t\t"+str(gam))
        print("mean NMI:\t"+str(avgNmi))
        print("std  NMI:\t"+str(stdNmi))
        return avgNmi

    #toOpt(None,17)

    pbounds = {
        'gam': (0, 0.001000),
        'n': (10, 2000)}
    optimizer = BayesianOptimization(
        f=toOpt,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(
        init_points=2,
        n_iter=ll[ii-1],
    )
    print("\n\n\n")
    print("mx:")
    print(optimizer.max)
    print("\n\n\n")



