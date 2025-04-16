from sklearn import metrics
from sklearn.metrics import confusion_matrix, silhouette_score, davies_bouldin_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import numpy as np
import cupy as cp
from sklearn.metrics import adjusted_rand_score as ari
import time as time
import gc
import scipy

def ordered_confusion_matrix(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    w = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(w)
    conf_mat = conf_mat[row_ind, :]
    conf_mat = conf_mat[:, col_ind]
    return conf_mat

def clustering_accuracy(y_true, y_pred):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)

def pmi(df, positive=True):
  #<class 'scipy.sparse._csr.csr_matrix'>
  col_totals = df.sum(axis=0)
  #col_totals = np.array(col_totals)
  #print(np.array(col_totals))
  total = col_totals.sum()
  row_totals = df.sum(axis=1)
  #row_totals = np.array(row_totals)



  # hevel = np.outer(row_totals, col_totals)
  # expected = hevel / total
  # df = df / expected

  dfg = cp.asarray(scipy.sparse.csr_matrix.toarray(df))

  RTg = cp.asarray(row_totals)
  CTg = cp.asarray(col_totals)
  hevelg = cp.outer(RTg, CTg)
  del RTg
  del CTg
  gc.collect()
  expectedg = hevelg / total
  del hevelg
  gc.collect()
  dfg = dfg / expectedg
  del expectedg
  gc.collect()

  # Silence distracting warnings about log(0):
  with np.errstate(divide='ignore'):
    #df = np.log1p(df - np.divide(df,df))
    # GPU
    df = cp.log1p(dfg - cp.divide(dfg,dfg))
    df = cp.asnumpy(dfg)
    del dfg
    gc.collect()


  df[~np.isfinite(df)] = 0.0  # log(0) = 0
  if positive:
    df[df < 0] = 0.0
  return df  

def average_pmi_per_cluster(x, labels):
  values = 0
  pmi_mat = pmi(x @ x.T, positive=False)
  for c in np.unique(labels):
    intra = pmi_mat[labels == c][:, labels == c]
    inter = pmi_mat[labels == c][:, labels != c]
    v = np.mean(intra) - np.mean(inter)
    values += v * np.sum(labels == c) / len(labels)
  return values

