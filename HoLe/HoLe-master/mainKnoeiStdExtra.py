"""Main
"""
# pylint: disable=line-too-long,invalid-name,
import argparse
import random

import pandas as pd
from tqdm import tqdm

from bayes_opt import BayesianOptimization

import scipy.sparse as sp
import torch
from graph_datasets import load_data

from models import HoLe
from models import HoLe_batch
from utils import check_modelfile_exists
from utils import csv2file
from utils import evaluation
from utils import get_str_time
from utils import set_device

import time
import numpy as np

import sys, os

if __name__ == "__main__":

    clear = lambda: os.system('clear')

    parser = argparse.ArgumentParser(
        prog="HoLe",
        description=
        "Homophily-enhanced Structure Learning for Graph Clustering",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="Cora",
        help="Dataset used in the experiment",
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        type=int,
        default=0,
        help="gpu id",
    )
    args = parser.parse_args()

    final_params = {}

    dim = 500
    n_lin_layers = 1
    dump = True
    device = set_device(str(args.gpu_id))

    def toOpt(datasets):

        gamma = -11
        xi = -11
        eta = -11

        myValue = 0.001

        lr = {
            "Cora": 0.001,
            "Citeseer": 0.001,
            "ACM": 0.001,
            "Pubmed": 0.001,
            "BlogCatalog": 0.001,
            "Flickr": 0.001,
            "Reddit": 2e-5,

            "texas": myValue,
            "cornell": myValue,
            "wisconsin": myValue,
            "chameleon": myValue,
            "squirrel": myValue,
            "roman-empire": myValue,
            "amazon-ratings": myValue,
            "minesweeper": myValue,
            "tolokers": myValue,
        }

        myValue = [3]

        n_gnn_layers = {
            "Cora": [8],
            "Citeseer": [3],
            "ACM": [3],
            "Pubmed": [35],
            "BlogCatalog": [1],
            "Flickr": [1],
            "Reddit": [3],

            "texas": myValue,
            "cornell": myValue,
            "wisconsin": myValue,
            "chameleon": myValue,
            "squirrel": myValue,
            "roman-empire": myValue,
            "amazon-ratings": myValue,
            "minesweeper": myValue,
            "tolokers": myValue,
        }

        myValue = [300]

        pre_epochs = {
            "Cora": [150],
            "Citeseer": [150],
            "ACM": [200],
            "Pubmed": [50],
            "BlogCatalog": [150],
            "Flickr": [300],
            "Reddit": [3],

            "texas": myValue,
            "cornell": myValue,
            "wisconsin": myValue,
            "chameleon": myValue,
            "squirrel": myValue,
            "roman-empire": myValue,
            "amazon-ratings": myValue,
            "minesweeper": myValue,
            "tolokers": myValue,
        }

        myValue = 150

        epochs = {
            "Cora": 50,
            "Citeseer": 150,
            "ACM": 150,
            "Pubmed": 200,
            "BlogCatalog": 150,
            "Flickr": 150,
            "Squirrel": 150,
            "Reddit": 3,

            "texas": myValue,
            "cornell": myValue,
            "wisconsin": myValue,
            "chameleon": myValue,
            "squirrel": myValue,
            "roman-empire": myValue,
            "amazon-ratings": myValue,
            "minesweeper": myValue,
            "tolokers": myValue,
        }

        myValue = (lambda x: x)

        inner_act = {
            "Cora": lambda x: x,
            "Citeseer": torch.sigmoid,
            "ACM": lambda x: x,
            "Pubmed": lambda x: x,
            "BlogCatalog": lambda x: x,
            "Flickr": lambda x: x,
            "Squirrel": lambda x: x,
            "Reddit": lambda x: x,

            "texas": myValue,
            "cornell": myValue,
            "wisconsin": myValue,
            "chameleon": myValue,
            "squirrel": myValue,
            "roman-empire": myValue,
            "amazon-ratings": myValue,
            "minesweeper": myValue,
            "tolokers": myValue,
        }

        myValue = 40

        udp = {
            "Cora": 10,
            "Citeseer": 40,
            "ACM": 40,
            "Pubmed": 10,
            "BlogCatalog": 40,
            "Flickr": 40,
            "Squirrel": 40,
            "Reddit": 40,

            "texas": myValue,
            "cornell": myValue,
            "wisconsin": myValue,
            "chameleon": myValue,
            "squirrel": myValue,
            "roman-empire": myValue,
            "amazon-ratings": myValue,
            "minesweeper": myValue,
            "tolokers": myValue,
        }

        myValue = [gamma]

        node_ratios = { # 0.1 - 1
            "Cora": [1],
            "Citeseer": [0.3],
            "ACM": [0.3],
            "Pubmed": [0.5],
            "BlogCatalog": [1],
            "Flickr": [0.3],
            "Squirrel": [0.3],
            "Reddit": [0.01],

            "texas": myValue,
            "cornell": myValue,
            "wisconsin": myValue,
            "chameleon": myValue,
            "squirrel": myValue,
            "roman-empire": myValue,
            "amazon-ratings": myValue,
            "minesweeper": myValue,
            "tolokers": myValue,
        }

        myValue = xi

        add_edge_ratio = { # 0.1 - 1
            "Cora": 0.5,
            "Citeseer": 0.5,
            "ACM": 0.5,
            "Pubmed": 0.5,
            "BlogCatalog": 0.5,
            "Flickr": 0.5,
            "Reddit": 0.005,

            "texas": myValue,
            "cornell": myValue,
            "wisconsin": myValue,
            "chameleon": myValue,
            "squirrel": myValue,
            "roman-empire": myValue,
            "amazon-ratings": myValue,
            "minesweeper": myValue,
            "tolokers": myValue,
        }

        myValue = [eta]

        del_edge_ratios = { # 0.001 - 0.01
            "Cora": [0.01],
            "Citeseer": [0.005],
            "ACM": [0.005],
            "Pubmed": [0.005],
            "BlogCatalog": [0.005],
            "Flickr": [0.005],
            "Reddit": [0.02],

            "texas": myValue,
            "cornell": myValue,
            "wisconsin": myValue,
            "chameleon": myValue,
            "squirrel": myValue,
            "roman-empire": myValue,
            "amazon-ratings": myValue,
            "minesweeper": myValue,
            "tolokers": myValue,
        }

        myValue = [8]

        gsl_epochs_list = {
            "Cora": [5],
            "Citeseer": [5],
            "ACM": [10],
            "Pubmed": [3],
            "BlogCatalog": [10],
            "Flickr": [10],
            "Reddit": [1],

            "texas": myValue,
            "cornell": myValue,
            "wisconsin": myValue,
            "chameleon": myValue,
            "squirrel": myValue,
            "roman-empire": myValue,
            "amazon-ratings": myValue,
            "minesweeper": myValue,
            "tolokers": myValue,
        }

        myValue = 0

        regularization = {
            "Cora": 1,
            "Citeseer": 0,
            "ACM": 0,
            "Pubmed": 1,
            "BlogCatalog": 0,
            "Flickr": 0,
            "Reddit": 0,

            "texas": myValue,
            "cornell": myValue,
            "wisconsin": myValue,
            "chameleon": myValue,
            "squirrel": myValue,
            "roman-empire": myValue,
            "amazon-ratings": myValue,
            "minesweeper": myValue,
            "tolokers": myValue,
        }


        source = {
            "Cora": "dgl",
            "Citeseer": "dgl",
            "ACM": "sdcn",
            "Pubmed": "dgl",
            "BlogCatalog": "cola",
            "Flickr": "cola",
            "Reddit": "dgl",

            "texas": "pyg",
            "cornell": "pyg",
            "wisconsin": "pyg",
            "chameleon": "pyg",
            "squirrel": "pyg",
            "roman-empire": "critical",
            "amazon-ratings": "critical",
            "minesweeper": "critical",
            "tolokers": "critical",
        }

        datasets = [datasets]

        for ds in datasets:
            if ds == "Reddit":
                hole = HoLe_batch
            else:
                hole = HoLe

            for gsl_epochs in gsl_epochs_list[ds]:
                runs = 1

                for n_gnn_layer in n_gnn_layers[ds]:
                    for pre_epoch in pre_epochs[ds]:
                        for del_edge_ratio in del_edge_ratios[ds]:
                            for node_ratio in node_ratios[ds]:
                                final_params["dim"] = dim
                                final_params["n_gnn_layers"] = n_gnn_layer
                                final_params["n_lin_layers"] = n_lin_layers
                                final_params["lr"] = lr[ds]
                                final_params["pre_epochs"] = pre_epoch
                                final_params["epochs"] = epochs[ds]
                                final_params["udp"] = None #udp[ds]
                                final_params["inner_act"] = inner_act[ds]
                                final_params["add_edge_ratio"] = add_edge_ratio[ds]
                                final_params["node_ratio"] = node_ratio
                                final_params["del_edge_ratio"] = del_edge_ratio
                                final_params["gsl_epochs"] = gsl_epochs

                                time_name = get_str_time()
                                save_file = f"results/hole/hole_{ds}_gnn_{n_gnn_layer}_gsl_{gsl_epochs}_{time_name[:9]}.csv"

                                graph, labels, n_clusters = load_data(
                                    dataset_name=ds,
                                    source=source[ds],
                                    verbosity=2,
                                )
                                features = graph.ndata["feat"]
                                if ds in ("Cora", "Pubmed"):
                                    graph.ndata["feat"][(features -
                                                        0.0) > 0.0] = 1.0
                                adj_csr = graph.adj_external(scipy_fmt="csr")
                                adj_sum_raw = adj_csr.sum()

                                edges = graph.edges()
                                features_lil = sp.lil_matrix(features)

                                final_params["dataset"] = ds

                                warmup_filename = f"hole_{ds}_run_gnn_{n_gnn_layer}"

                                if not check_modelfile_exists(warmup_filename):
                                    # print("warmup first")
                                    model = hole(
                                        hidden_units=[dim],
                                        in_feats=features.shape[1],
                                        n_clusters=n_clusters,
                                        n_gnn_layers=n_gnn_layer,
                                        n_lin_layers=n_lin_layers,
                                        lr=lr[ds],
                                        n_pretrain_epochs=pre_epoch,
                                        n_epochs=epochs[ds],
                                        norm="sym",
                                        renorm=True,
                                        tb_filename=
                                        f"{ds}_gnn_{n_gnn_layer}_node_{node_ratio}_{add_edge_ratio[ds]}_{del_edge_ratio}_pre_ep{pre_epoch}_ep{epochs[ds]}_dim{dim}_{random.randint(0, 999999)}",
                                        warmup_filename=warmup_filename,
                                        inner_act=inner_act[ds],
                                        udp=udp[ds],
                                        regularization=regularization[ds],
                                    )

                                    model.fit(
                                        graph=graph,
                                        device=device,
                                        add_edge_ratio=add_edge_ratio[ds],
                                        node_ratio=node_ratio,
                                        del_edge_ratio=del_edge_ratio,
                                        gsl_epochs=0,
                                        labels=labels,
                                        adj_sum_raw=adj_sum_raw,
                                        load=False,
                                        dump=dump,
                                    )

                                seed_list = [
                                    random.randint(0, 999999) for _ in range(runs)
                                ]
                                for run_id in range(runs):
                                    final_params["run_id"] = run_id
                                    seed = seed_list[run_id]
                                    final_params["seed"] = seed
                                    # Citeseer needs reset to overcome over-fitting
                                    reset = ds == "Citeseer"

                                    model = hole(
                                        hidden_units=[dim],
                                        in_feats=features.shape[1],
                                        n_clusters=n_clusters,
                                        n_gnn_layers=n_gnn_layer,
                                        n_lin_layers=n_lin_layers,
                                        lr=lr[ds],
                                        n_pretrain_epochs=pre_epoch,
                                        n_epochs=epochs[ds],
                                        norm="sym",
                                        renorm=True,
                                        tb_filename=
                                        f"{ds}_gnn_{n_gnn_layer}_node_{node_ratio}_{add_edge_ratio[ds]}_{del_edge_ratio}_gsl_{gsl_epochs}_pre_ep{pre_epoch}_ep{epochs[ds]}_dim{dim}_{random.randint(0, 999999)}",
                                        warmup_filename=warmup_filename,
                                        inner_act=inner_act[ds],
                                        udp=udp[ds],
                                        reset=reset,
                                        regularization=regularization[ds],
                                        seed=seed,
                                    )

                                    model.fit(
                                        graph=graph,
                                        device=device,
                                        add_edge_ratio=add_edge_ratio[ds],
                                        node_ratio=node_ratio,
                                        del_edge_ratio=del_edge_ratio,
                                        gsl_epochs=gsl_epochs,
                                        labels=labels,
                                        adj_sum_raw=adj_sum_raw,
                                        load=True,
                                        dump=dump,
                                    )

                                    with torch.no_grad():
                                        z_detached = model.get_embedding()
                                        Q = model.get_Q(z_detached)
                                        q = Q.detach().data.cpu().numpy().argmax(1)
                                    (
                                        ARI_score,
                                        NMI_score,
                                        AMI_score,
                                        ACC_score,
                                        Micro_F1_score,
                                        Macro_F1_score,
                                        purity,
                                    ) = evaluation(labels, q)

                                    # print("\n"
                                    #     f"ARI:{ARI_score}\n"
                                    #     f"NMI:{ NMI_score}\n"
                                    #     f"AMI:{ AMI_score}\n"
                                    #     f"ACC:{ACC_score}\n"
                                    #     f"Micro F1:{Micro_F1_score}\n"
                                    #     f"Macro F1:{Macro_F1_score}\n"
                                    #     f"purity_score:{purity}\n")

                                    final_params["qARI"] = ARI_score
                                    final_params["qNMI"] = NMI_score
                                    final_params["qACC"] = ACC_score
                                    final_params["qMicroF1"] = Micro_F1_score
                                    final_params["qMacroF1"] = Macro_F1_score
                                    final_params["qPurity"] = Macro_F1_score

                                    if save_file is not None:
                                        csv2file(
                                            target_path=save_file,
                                            thead=list(final_params.keys()),
                                            tbody=list(final_params.values()),
                                        )
                                        # print(f"write to {save_file}")

        return (final_params)

    def toOptSilentF1(gamma,xi,eta):
        sys.stdout = open(os.devnull, 'w')
        uit = toOpt(gamma,xi,eta)
        sys.stdout = sys.__stdout__
        return uit["qMicroF1"]

    def toOptSilentNMI(gamma,xi,eta):
        sys.stdout = open(os.devnull, 'w')
        uit = toOpt(gamma,xi,eta)
        sys.stdout = sys.__stdout__
        return uit["qNMI"]


    prs = {}
    prs[('texas',"F1")] = {'eta': 0.09056786017067009, 'gamma': 0.4661603526428678, 'xi': 0.5721751294498395}
    prs[('texas',"NMI")] = {'eta': 0.09867920983459957, 'gamma': 0.6750119490629675, 'xi': 0.3794930623374809}
    prs[('cornell',"F1")] = {'eta': 0.034283258885109115, 'gamma': 0.9343399667400427, 'xi': 0.947548945539555}
    prs[('cornell',"NMI")] = {'eta': 0.0036917623609216787, 'gamma': 0.46808308319677383, 'xi': 0.20152568819948655}
    prs[('wisconsin',"F1")] = {'eta': 0.02566839062191928, 'gamma': 0.177693836316196, 'xi': 0.9145609670094857}
    prs[('wisconsin',"NMI")] = {'eta': 0.052877722512764375, 'gamma': 0.9533101229831927, 'xi': 0.6427064682746172}
    prs[('chameleon',"F1")] = {'eta': 0.002433162696718917, 'gamma': 0.5099598817204188, 'xi': 0.2665007133346956}
    prs[('chameleon',"NMI")] = {'eta': 0.042825396854477366, 'gamma': 0.1855019928861414, 'xi': 0.48725955512305286}
    prs[('squirrel',"F1")] = {'eta': 0.036597973132358484, 'gamma': 0.9991425640349296, 'xi': 0.6103549855250833}
    prs[('squirrel',"NMI")] = {'eta': 0.03400239103063429, 'gamma': 0.813070431201588, 'xi': 0.7762092665391783}
    prs[('minesweeper',"F1")] = {'eta': 0.002838703911552943, 'gamma': 0.5952780812172738, 'xi': 0.8459821609256446}
    prs[('minesweeper',"NMI")] = {'eta': 0.00317637986772471, 'gamma': 0.38731235450303136, 'xi': 0.5414278164133848}
    prs[('tolokers',"F1")] = {'eta': 0.02966067223565122, 'gamma': 0.4355679101573471, 'xi': 0.9101530379759138}
    prs[('tolokers',"NMI")] = {'eta': 0.08722, 'gamma': 0.8878, 'xi': 0.7451}

    aantal = 5;
    datasets = ['chameleon','squirrel','tolokers','minesweeper']
    metrieken = ["NMI"]
    tot = len(datasets) * len(metrieken)
    gegevens = pd.DataFrame()
    teller = 0
    for d in datasets:
        for m in metrieken:
            ldta = {}
            teller += 1
            clear()
            print(gegevens)
            ldta["pct"] = round(100*teller/tot)
            ldta["dataset"] = d
            ldta["metriek"] = m
            hevel = []
            tijd = []
            for i in tqdm(range(aantal)):
                sys.stdout = open(os.devnull, 'w') # stil
                beginTijd = time.time()
                h = toOpt(d)
                eindTijd = time.time()
                sys.stdout = sys.__stdout__
                tijd = tijd = [ eindTijd - beginTijd ]
                if m == 'F1':
                    hevel = hevel + [ h['qMicroF1'] ]
                if m == 'NMI':
                    hevel = hevel + [ h['qNMI'] ]
            hnp = np.array(hevel)
            tnp = np.array(tijd)
            ldta["avg"] = np.mean(hnp)
            ldta["std"] = np.std(hnp)
            ldta["avgTijd [s]"] = np.mean(tnp)
            ldta["stdTijd [s]"] = np.std(tnp)
            gegevens = gegevens._append(ldta,ignore_index=True)
            gegevens.to_csv("gegevensHoLeCoraCiteseerPubmed"+str(aantal)+".csv")

    clear()
    print("Finaal:")
    print(gegevens)
    gegevens.to_csv("gegevensHoLeChmlnSqrrlTlkrsMnswpr"+str(aantal)+".csv")

    # titel = '\n' + sys.argv[2] + ": F1" + '\n'
    # print(titel)
    # punten = 60
    # bounds = {'gamma':(0.1,1.0),'xi':(0.1,1.0),'eta':(0.001,0.1)}
    # boF1 = BayesianOptimization(f=toOptSilentF1,pbounds=bounds)
    # boF1.maximize(n_iter=punten)
    # fl = open("besteParams.txt",'a')
    # fl.write(str(sys.argv))
    # fl.write('\n')
    # fl.write("F1\n")
    # fl.write(str(boF1.max))
    # fl.write('\n')
    # fl.write('\n')
    # fl.close
    # titel = '\n' + sys.argv[2] + ": NMI" + '\n'
    # print(titel)
    # boNMI = BayesianOptimization(f=toOptSilentNMI,pbounds=bounds)
    # boNMI.maximize(n_iter=punten)
    # fl = open("besteParams.txt",'a')
    # fl.write(str(sys.argv))
    # fl.write('\n')
    # fl.write("NMI\n")
    # fl.write(str(boNMI.max))
    # fl.write('\n')
    # fl.write('\n')
    # fl.close
