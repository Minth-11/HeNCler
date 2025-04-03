import os
import scipy
import zipfile
import scipy as sp

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

print(wikiMat["fea"]) # features # shape (2405, 4973)
#print(wikiMat["gnd"]) # labels
#print(wikiMat["W"]) # ???

data = wikiMat

features= data["fea"].astype(float)
if not sp.issparse(features):
      features = sp.csr_matrix(features)
labels = data['gnd'].reshape(-1) - 1
n_classes = len(np.unique(labels))

print(features)

#wikiIn =
