import os
import scipy
import zipfile

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
