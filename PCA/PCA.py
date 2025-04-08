import os
import scipy
import zipfile
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans

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

data = wikiMat

features= data["fea"].astype(float)
if not sp.sparse.issparse(features):
      features = sp.csr_matrix(features)
labels = data['gnd'].reshape(-1) - 1 # [0...16]
n_classes = len(np.unique(labels))

inData = features#.transpose()

print(min(labels))

def prt(x):
    print(x,end='',flush=True)

nclstrs = 17


prt("fit...")
pca = PCA()
pca.fit(inData)
prt("KLAAR\n")

print(pca.singular_values_)

prt("transform...")
ntr = pca.transform(inData)
prt("KLAAR\n")
print(ntr)

NMIs = []
for i in range(10):
    prt("KMeans "+str(i)+"...")
    kmeans = KMeans(n_clusters=nclstrs,verbose=0)
    kmeans.fit(inData)
    prt("KLAAR\n")

    fts = kmeans.predict(inData)
    # print(fts)

    NMIs = NMIs + [nmi(labels, fts)]

print(NMIs)
avgNmi = np.mean(np.array(NMIs))
stdNmi = np.std( np.array(NMIs))
print("mean NMI:\t"+str(avgNmi))
print("std  NMI:\t"+str(stdNmi))

#wikiIn =
