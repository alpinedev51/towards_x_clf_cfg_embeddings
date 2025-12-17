import os
import pickle
import random

import gensim

# from angrutils import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.estimator import Estimator
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing
from sklearn.manifold import TSNE


def loadCFG(cfg_path, n_prog=0):
    # load CFG files

    if not os.path.exists(cfg_path):
        print("Path of the file is Invalid")

    file_paths = []
    file_names = []

    Graphs = []
    i = 0
    for root, dirs, files in os.walk(cfg_path):
        # load all the programs
        if n_prog == 0:
            n_prog = n_prog + len(files)

        print(root)
        for file in files:
            print(i)
            print(file)

            if i >= n_prog:
                break
            else:
                f_path = root + file  # file path

                try:
                    with open(f_path, "rb") as f_G:
                        G = pickle.load(f_G)
                    # G = nx.read_gpickle(f_path)

                    # nx.draw_networkx(G)
                    # plt.show()
                    Graphs.append(G)  # needs a list of graphs
                    file_names.append(file[:-8])  # get rid of ".gpickle" extension

                    i = i + 1

                    # nx.to_edgelist(cfg, "test.edgelist")
                    # nx.write_edgelist(G, "./benign_graphs_test/edgelist/"+file+ ".edgelist",data=True)
                    # nx.write_adjlist(cfg.graph, "test.adjlist")
                    print(
                        "It has %d nodes and %d edges"
                        % (len(G.nodes()), len(G.edges()))
                    )

                except Exception as e:
                    print("ERROR!!!!!!! : %s" % e)
                    print("File %s skipped" % file)

    return Graphs, file_names


def loadTestCFG(cfg_path, n_malware=0, n_benign=0):
    # Load CFG dataset from the given cfg path

    malware_cfg_path = cfg_path + "Malware_CFG/"
    benign_cfg_path = cfg_path + "Benign_CFG/"

    # Load Malware .gpickle CFG files
    Malware_graphs, Malware_names = loadCFG(malware_cfg_path, n_malware)
    Benign_graphs, Benign_names = loadCFG(benign_cfg_path, n_benign)

    n_malware = len(Malware_names)
    n_benign = len(Benign_names)

    Malware_Labels = ["Malware"] * n_malware
    Benign_Labels = ["Benign"] * n_benign

    ## Combining the Malware and Benign
    graph_list = Malware_graphs + Benign_graphs
    file_names = Malware_names + Benign_names
    n_prog = n_malware + n_benign
    Labels = Malware_Labels + Benign_Labels

    return graph_list, file_names, Labels, n_prog


def createWLhash(graph_list, param):
    # TODO: parallel implementation
    documents = []

    for graph in graph_list:
        G = graph
        G = param._check_graph(G)

        document = WeisfeilerLehmanHashing(
            G, param.wl_iterations, param.attributed, param.erase_base_features
        )

        documents.append(document)

    documents = [
        TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
        for i, doc in enumerate(documents)
    ]

    return documents


def trainD2Vmodel(train_corpus, param):
    d2v_model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

    d2v_model.build_vocab(train_corpus)

    d2v_model = Doc2Vec(
        train_corpus,
        vector_size=param.dimensions,
        window=0,
        min_count=param.min_count,
        dm=0,
        sample=param.down_sampling,
        workers=param.workers,
        epochs=param.epochs,
        alpha=param.learning_rate,
        seed=param.seed,
    )

    d2v_model.train(
        train_corpus, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs
    )

    return d2v_model


def DocShuffle(documents, labels, names):
    # https://pynative.com/python-random-shuffle/
    # To Shuffle two List at once with the same order
    mapIndexPosition = list(zip(documents, labels, names))
    random.shuffle(mapIndexPosition)
    # make list separate
    documents_shuffled, labels_shuffled, names_shuffled = zip(*mapIndexPosition)

    return documents_shuffled, labels_shuffled, names_shuffled


def TSNE_2D_plot(vector, labels, n_vec, dimensions, return_plot=False):
    twoD_embedded_graphs = TSNE(n_components=2).fit_transform(vector)

    idx_malware = [i for i in range(n_vec) if labels[i] == "Malware"]
    idx_benign = [i for i in range(n_vec) if labels[i] == "Benign"]

    # plt.subplot(1,2,1)
    plt.plot(
        twoD_embedded_graphs[idx_malware, 0],
        twoD_embedded_graphs[idx_malware, 1],
        "r*",
        label="malware",
    )
    plt.legend(loc="upper left")
    # plt.subplot(1,2,2)
    plt.plot(
        twoD_embedded_graphs[idx_benign, 0],
        twoD_embedded_graphs[idx_benign, 1],
        "b*",
        label="benign",
    )
    plt.legend(loc="upper left")
    plt.suptitle(
        "Graph2Vec (" + str(dimensions) + " dims) \n TSNE visualization of input graphs"
    )
    # plt.legend(bbox_to_anchor=(1.05, 1))

    fig1 = plt.gcf()
    plt.show()

    if return_plot:
        return twoD_embedded_graphs, fig1
    else:
        plt.clf()
        return twoD_embedded_graphs


# def plot_Malware_type(twoD_tsne_vector, test_vector_labels, test_vector_names,info_path ):
def plot_Malware_type(file_names, info_path):
    malware_info_path = info_path  # + 'malware.csv'

    malware_data = pd.read_csv(malware_info_path, names=["Name", "label", "Type"])

    columns = malware_data.columns
    df_malware = pd.DataFrame(columns=["Name", "Malware_Type"])

    M_types = malware_data["Type"].unique()
    j = 0
    for M_type in M_types[1:]:
        i = 0
        idx = []
        colr = []
        print(M_type)
        for file in file_names:
            # file_hash = file[:-3]      # get rid of ".exe"
            temp = file.split(".")
            file_hash = temp[0]
            file_entry = malware_data[malware_data["Name"] == file_hash]

            if file_entry.size != 0:
                idx.append(i)
            i = i + 1

        plt.subplot(ceildiv(len(M_types), 2), 2, j + 1)
        plt.plot()
        # plt.scatter(twoD_tsne_vector[idx, 0], twoD_tsne_vector[idx, 1], c=colr, label=M_type)
        plt.title(M_type)
        j = j + 1

    # plt.colorbar(orientation='horizontal')
    plt.suptitle("Malware types")
    plt.figure(figsize=(3, 5))
    fig = plt.gcf()

    plt.show()

    return 0


def read_Dike_file_info(twoD_tsne_vector, info_path, file_names):
    # load malware information
    malware_info_path = info_path + "malware.csv"

    malware_data = pd.read_csv(malware_info_path)

    columns = malware_data.columns
    df_malware = pd.DataFrame(columns=["Name", "Malware_Type"])

    # malware_idx = []

    for file in file_names:
        # file_hash = file[:-3]      # get rid of ".exe"
        temp = file.split(".")
        file_hash = temp[0]
        file_entry = malware_data[malware_data["hash"] == file_hash]

        if file_entry.size != 0:
            # print(file_entry)
            t = file_entry.values[0, 3:]
            max_ind = np.argmax(t)

            # print(columns[max_ind+3])

            df_malware = df_malware.append(
                {"Name": file, "Malware_Type": columns[max_ind + 3]}, ignore_index=True
            )
        else:
            print("Not found")

    j = 0
    for M_type in columns[2:]:
        i = 0
        idx = []
        colr = []
        print(M_type)
        for file in file_names:
            # file_hash = file[:-3]      # get rid of ".exe"
            temp = file.split(".")
            file_hash = temp[0]
            file_entry = malware_data[malware_data["hash"] == file_hash]

            if file_entry.size != 0:
                if file_entry.iloc[0][M_type] != 0:
                    idx.append(i)
                    colr.append(file_entry.iloc[0][M_type])
            i = i + 1

        plt.subplot(2, 5, j + 1)
        plt.plot()
        plt.scatter(
            twoD_tsne_vector[idx, 0], twoD_tsne_vector[idx, 1], c=colr, label=M_type
        )
        plt.title(M_type)
        j = j + 1

    plt.colorbar(orientation="horizontal")
    plt.suptitle("Malware types")
    plt.show()

    print("dONE")

    return 0


def trainG2V(train_graphs, train_labels, train_names, param):
    ## Create WL hash word documents
    print("Creating WL hash words for training set")
    train_documents = createWLhash(train_graphs, param)

    ## Shuffling of the data
    print("Shuffling data")
    train_corpus, train_labels, train_names = DocShuffle(
        train_documents, train_labels, train_names
    )

    ## Doc2Vec model training
    print("D2V training")
    d2v_model = trainD2Vmodel(train_corpus, param)

    return d2v_model


def inferG2V(test_graphs, test_labels, test_names, param):
    ## Create WL hash word documents for testing set
    print("Creating WL hash words for testing set")
    test_documents = createWLhash(test_graphs, param)

    ## Shuffling of the data (** optional)
    print("Shuffling data")
    test_corpus, test_labels, test_names = DocShuffle(
        test_documents, test_labels, test_names
    )

    ## Doc2Vec inference
    print("Doc2Vec inference")
    test_vector = np.array([d2v_model.infer_vector(d.words) for d in test_corpus])

    return test_vector, test_labels, test_names


def train_test_divide(
    Malware_graphs,
    Benign_graphs,
    Malware_names,
    Benign_names,
    n_malware,
    n_benign,
    precent_train,
):
    Malware_Labels = ["Malware"] * n_malware
    Benign_Labels = ["Benign"] * n_benign

    ## Combining the Malware and Benign
    graph_list = Malware_graphs + Benign_graphs
    file_names = Malware_names + Benign_names
    n_prog = n_malware + n_benign
    Labels = Malware_Labels + Benign_Labels

    ## Train Test divide of graphs
    n_train_mal = round(n_malware * precent_train)
    n_train_ben = round(n_benign * precent_train)

    n_train = n_train_mal + n_train_ben
    n_test = n_prog - n_train

    train_graphs = Malware_graphs[:n_train_mal] + Benign_graphs[:n_train_ben]
    test_graphs = Malware_graphs[n_train_mal:] + Benign_graphs[n_train_ben:]

    train_labels = Malware_Labels[:n_train_mal] + Benign_Labels[:n_train_ben]
    test_labels = Malware_Labels[n_train_mal:] + Benign_Labels[n_train_ben:]

    train_names = Malware_names[:n_train_mal] + Benign_names[:n_train_ben]
    test_names = Malware_names[n_train_mal:] + Benign_names[n_train_ben:]

    return (
        train_graphs,
        test_graphs,
        train_labels,
        test_labels,
        n_train,
        n_test,
        train_names,
        test_names,
    )


def train_G2V_model(
    train_graphs,
    train_labels,
    train_names,
    param,
    ndims_list,
    save_model=True,
    output_path="./",
):
    ## Create WL hash word documents
    print("Creating WL hash words for training set")
    train_documents = createWLhash(train_graphs, param)

    ## Shuffling of the data
    print("Shuffling data")
    train_corpus, train_labels, train_names = DocShuffle(
        train_documents, train_labels, train_names
    )

    # ndims_list = [8]
    for ndims in ndims_list:
        print("Dimensions = ", ndims)

        ## Parameters
        param = WL_parameters(dimensions=ndims)
        param._set_seed()

        ## Doc2Vec model training
        print("D2V training")
        d2v_model = trainD2Vmodel(train_corpus, param)

        if save_model:
            model_path = output_path + "d2v_models/"

            os.makedirs(model_path, exist_ok=True)

            model_name = (
                model_path + "/" + "d2v_model_" + str(param.dimensions) + ".model"
            )
            d2v_model.save(fname_or_handle=model_name)

    if save_model:
        return model_name
    else:
        return d2v_model


def ceildiv(a, b):
    return -(a // -b)


class WL_parameters(Estimator):
    def __init__(
        self,
        wl_iterations: int = 2,
        attributed: bool = False,
        dimensions: int = 128,
        workers: int = 4,
        down_sampling: float = 0.0001,
        epochs: int = 10,
        learning_rate: float = 0.025,
        min_count: int = 5,
        seed: int = 42,
        erase_base_features: bool = False,
    ):
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.dimensions = dimensions
        self.workers = workers
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self.erase_base_features = erase_base_features
