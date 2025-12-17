import argparse
import gc
import math
import os
from typing import List

import shap
from scipy.stats import loguniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from cluster_util import *
from g2v_util import *


def graph_embedding_training(
    cfg_path, output_path, ndims_list=[8], n_malware=3000, n_benign=3000
):
    malware_cfg_path = cfg_path + "Malware_CFG/"
    benign_cfg_path = cfg_path + "Benign_CFG/"

    n_precent_train = 0.2

    # Load .gpickle CFG files
    Malware_graphs, Malware_names = loadCFG(malware_cfg_path, n_malware)
    Benign_graphs, Benign_names = loadCFG(benign_cfg_path, n_benign)

    ## Train divide for vocabulary training graphs
    (
        vocab_train_graphs,
        train_graphs,
        vocab_train_labels,
        train_labels,
        n_vocab_train,
        n_train,
        vocab_train_names,
        train_names,
    ) = train_test_divide(
        Malware_graphs,
        Benign_graphs,
        Malware_names,
        Benign_names,
        n_malware,
        n_benign,
        n_precent_train,
    )

    # Save memory by removing unnecessary  variables
    if "Malware_graphs" in os.environ:
        del Malware_graphs

    if "Benign_graphs" in os.environ:
        del Benign_graphs

    if "Malware_names" in os.environ:
        del Malware_names

    if "Benign_names" in os.environ:
        del Benign_names

    gc.collect()

    ########## Training Graph2Vec* Model ##############

    ## Parameters
    param = WL_parameters()
    param._set_seed()

    # Train and save Graph2Vec model
    train_G2V_model(
        vocab_train_graphs,
        vocab_train_labels,
        vocab_train_names,
        param,
        ndims_list,
        save_model=True,
        output_path=output_path,
    )

    if "vocab_train_graphs" in os.environ:
        del vocab_train_graphs

    if "vocab_train_labels" in os.environ:
        del os.environ["vocab_train_labels"]

    if "vocab_train_names" in os.environ:
        del os.environ["vocab_train_names"]

    if "d2v_model" in os.environ:
        del os.environ["d2v_model"]

    ######### Inferencing the vector for  data ################

    ## Graph2Vec* inference
    print("Graph2Vec inference")
    # test_vector, test_vector_labels, test_vector_names = inferG2V(test_graphs, test_labels, test_names, param)

    ## Create WL hash word documents for testing set
    print("Creating WL hash words for remaining training set")
    train_documents = createWLhash(train_graphs, param)

    if "train_graphs" in os.environ:
        del train_graphs

    for ndim in ndims_list:
        print("Dimensions = ", ndim)

        ## Parameters
        param = WL_parameters(dimensions=ndim)
        param._set_seed()

        # model path
        model_path = output_path + "d2v_models/"
        model_name = model_path + "/" + "d2v_model_" + str(param.dimensions) + ".model"

        try:
            d2v_model = Doc2Vec.load(model_name)
        except Exception as e:
            print("ERROR!!!!!!! : %s" % e)

        ## Shuffling of the data
        print("Shuffling data")
        train_corpus, train_vector_labels, train_vector_names = DocShuffle(
            train_documents, train_labels, train_names
        )

        ## Doc2Vec inference
        print("Doc2Vec inference")
        train_vector = np.array([d2v_model.infer_vector(d.words) for d in train_corpus])

        vector_out_path = Path(
            model_path + "/" + "train_file_vectors_" + str(param.dimensions) + ".csv"
        )
        vector_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(train_vector).to_csv(vector_out_path, header=None, index=None)

        label_out_path = Path(
            model_path + "/" + "train_file_labels_" + str(param.dimensions) + ".csv"
        )
        label_out_path.parent.mkdir(parents=True, exist_ok=True)
        train_df = pd.DataFrame(
            {"name": train_vector_names, "Label": train_vector_labels}
        )
        train_df.to_csv(label_out_path)

        ## Visualizing
        print("Visualizations")
        twoD_tsne_vector, fig = TSNE_2D_plot(
            train_vector,
            train_vector_labels,
            n_train,
            param.dimensions,
            return_plot=True,
        )

        fig_name = (
            model_path + "/" + "train_vector_" + str(param.dimensions) + "-dims.png"
        )
        fig.savefig(fig_name)
        plt.clf()

    return param, model_path


def clustering_training(output_path, cluster_alg_name=["Kmeans"], ndims_list=[8]):
    save_model = True
    # ndims_list = [2, 4, 8, 16, 32, 64, 128, 256]
    # ndims_list = [2]

    # cluster_alg_name = ['Kmeans', 'spectral', 'Aggloromative', 'DBSCAN']

    # cluster_alg_name = ['DBSCAN']

    for ndim in ndims_list:
        print("Dimensions = ", ndim)

        # load data
        model_path = output_path + "d2v_models/"
        vector_path = model_path + "/" + "train_file_vectors_" + str(ndim) + ".csv"
        label_path = model_path + "/" + "train_file_labels_" + str(ndim) + ".csv"

        train_vector = pd.read_csv(vector_path, header=None).values

        vector = StandardScaler().fit_transform(train_vector)

        train_df = pd.read_csv(label_path)
        train_vector_labels = train_df["Label"].tolist()

        X_train, X_val, y_train, y_val = train_test_split(
            vector, train_vector_labels, test_size=0.4, random_state=0
        )

        ## Visualizing
        print("generating clustering and visualizations...")
        twoD_tsne_vector, fig = TSNE_2D_plot(
            X_train, y_train, len(y_train), ndim, return_plot=True
        )

        fig_name = model_path + "/" + "val_vector_" + str(ndim) + "-dims.png"
        fig.savefig(fig_name)
        plt.clf()

        if "fig" in os.environ:
            del os.environ["fig"]

        for cluster_alg in cluster_alg_name:
            if cluster_alg == "Kmeans":
                hyper_para_name = "n_clusters"
                random_state = 0
                hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
                hyper_para_list = np.arange(2, 31, step=1)
            elif cluster_alg == "spectral":
                hyper_para_name = "n_clusters"
                assign_labels = "discretize"
                hyper_para_list = np.arange(2, 31, step=1)
            elif cluster_alg == "Aggloromative":
                hyper_para_name = "n_clusters"
                linkage = "ward"
                hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
                hyper_para_list = np.arange(2, 31, step=1)
            elif cluster_alg == "DBSCAN":
                hyper_para_name = "eps"
                min_samples = 1
                hyper_para_list = np.arange(5, 150, step=5)

            # hyper_para_name, hyper_para_list = get_clf_hyper_para(cluster_alg)

            print(cluster_alg)
            cluster_valuation = pd.DataFrame()

            for hyper_para in hyper_para_list:
                print("hyper parameter = ", hyper_para)
                if cluster_alg == "Kmeans":
                    clustering_model, fig = generate_kmeans_clustering(
                        X_train,
                        y_train,
                        twoD_tsne_vector,
                        n_cluster=hyper_para,
                        random_state=random_state,
                    )
                    y_pred = clustering_model.predict(X_val)
                elif cluster_alg == "spectral":
                    clustering_model, fig = generate_spectral_clustering(
                        X_train,
                        y_train,
                        twoD_tsne_vector,
                        n_cluster=hyper_para,
                        assign_labels=assign_labels,
                    )
                    y_pred = clustering_model.fit_predict(X_val)
                elif cluster_alg == "Aggloromative":
                    clustering_model, fig = generate_AgglomerativeCluster(
                        X_train,
                        y_train,
                        twoD_tsne_vector,
                        n_cluster=hyper_para,
                        linkage=linkage,
                    )
                    y_pred = clustering_model.fit_predict(X_val)
                elif cluster_alg == "DBSCAN":
                    clustering_model, fig = generate_DBSCAN(
                        X_train,
                        y_train,
                        hyper_para / 100,
                        min_samples,
                        twoD_tsne_vector,
                    )
                    y_pred = clustering_model.fit_predict(X_val)

                # cluster_valuation.loc[0,len(cluster_valuation.index)] = cluster_evaluation(y_val, y_pred)
                eval, cntg = cluster_evaluation(y_val, y_pred)
                print(cntg)
                cluster_valuation = cluster_valuation._append(eval, ignore_index=True)
                cluster_valuation.reset_index()

                if save_model:
                    cluster_model_path = (
                        model_path + "/" + cluster_alg + "/cluster_models"
                    )

                    os.makedirs(cluster_model_path, exist_ok=True)

                    cluster_model_name = (
                        cluster_model_path
                        + "/"
                        + "clustering_model_"
                        + str(ndim)
                        + "-dims_"
                        + str(hyper_para)
                        + "-clusters.sav"
                    )

                    pickle.dump(clustering_model, open(cluster_model_name, "wb"))

                # fig_name = Path(model_path + '/'+ cluster_alg+'/validation/' + 'val_clustering_' + str(ndims) + '-dims_'+ str(n_cluster)+'-clusters.png')
                # fig_name.parent.mkdir(parents=True, exist_ok=True)
                # fig.savefig(fig_name)

            cluster_valuation.insert(0, hyper_para_name, hyper_para_list)

            valuation_name = Path(
                model_path
                + "/"
                + cluster_alg
                + "/validation/"
                + "val_clustering_evaluation_"
                + str(ndim)
                + "-dims.csv"
            )
            valuation_name.parent.mkdir(parents=True, exist_ok=True)
            cluster_valuation.to_csv(valuation_name)

    return 0


def cluster_prediction(
    cfg_path, output_path, param, cluster_alg_name=["Kmeans"], ndims_list=[8]
):
    ##### Testing ######################

    # load testing data
    print("Load test CFG data")
    test_graphs, test_file_names, test_Labels, n_test = loadTestCFG(
        cfg_path, n_test_malware, n_test_benign
    )

    ## Create WL hash word documents for testing set
    print("Creating WL hash words for testing set")
    test_documents = createWLhash(test_graphs, param)

    for ndims in ndims_list:
        print("Dimensions = ", ndims)

        ## Parameters
        param = WL_parameters(dimensions=ndims)
        param._set_seed()

        # model path
        model_path = output_path + "d2v_models"
        model_name = model_path + "/" + "d2v_model_" + str(param.dimensions) + ".model"

        try:
            d2v_model = Doc2Vec.load(model_name)
        except Exception as e:
            print("ERROR - d2v model not found!!!!!!! : %s" % e)

        ## Doc2Vec inference
        print("Doc2Vec inference")
        test_vector = np.array(
            [d2v_model.infer_vector(d.words) for d in test_documents]
        )

        vector_out_path = Path(
            model_path
            + "/Test/"
            + "test_file_vectors_"
            + str(param.dimensions)
            + ".csv"
        )
        vector_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_vector).to_csv(vector_out_path, header=None, index=None)

        label_out_path = Path(
            model_path + "/Test/" + "file_labels_" + str(param.dimensions) + ".csv"
        )
        label_out_path.parent.mkdir(parents=True, exist_ok=True)
        test_df = pd.DataFrame({"name": test_file_names, "Label": test_Labels})
        test_df.to_csv(label_out_path)

        ## Visualizing
        print("Visualizations")
        twoD_tsne_vector, fig = TSNE_2D_plot(
            test_vector, test_Labels, n_test, param.dimensions, return_plot=True
        )

        fig_name = (
            model_path + "/Test/" + "test_vector_" + str(param.dimensions) + "-dims.png"
        )
        fig.savefig(fig_name)
        plt.clf()

        for cluster_alg in cluster_alg_name:
            print(cluster_alg)
            cluster_valuation = pd.DataFrame()

            if cluster_alg == "Kmeans":
                hyper_para_name = "n_clusters"
                random_state = 0
                hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
            elif cluster_alg == "spectral":
                hyper_para_name = "n_clusters"
                assign_labels = "discretize"
                hyper_para_list = np.arange(2, 31, step=1)
            elif cluster_alg == "Aggloromative":
                hyper_para_name = "n_clusters"
                linkage = "ward"
                hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
            elif cluster_alg == "DBSCAN":
                hyper_para_name = "eps"
                min_samples = 2
                hyper_para_list = np.arange(5, 150, step=5)

            hyper_para_name, hyper_para_list = get_clf_hyper_para(cluster_alg)

            for hyper_para in hyper_para_list:
                print("hyper parameter = ", hyper_para)
                cluster_model_path = model_path + "/" + cluster_alg + "/cluster_models"
                load_model = True
                if load_model:
                    os.makedirs(cluster_model_path, exist_ok=True)

                    cluster_model_name = (
                        cluster_model_path
                        + "/"
                        + "clustering_model_"
                        + str(ndims)
                        + "-dims_"
                        + str(hyper_para)
                        + "-clusters.sav"
                    )

                    # load the model from disk
                    try:
                        clustering_model = pickle.load(open(cluster_model_name, "rb"))
                    except Exception as e:
                        print("ERROR - clustering model not found!!!!!!! : %s" % e)

                if cluster_alg == "Kmeans":
                    array_float = np.array(test_vector, dtype=np.float64)
                    y_pred = clustering_model.predict(array_float)
                elif cluster_alg == "spectral":
                    y_pred = clustering_model.fit_predict(test_vector)
                elif cluster_alg == "Aggloromative":
                    y_pred = clustering_model.fit_predict(test_vector)
                elif cluster_alg == "DBSCAN":
                    y_pred = clustering_model.fit_predict(test_vector)

                # cluster_valuation.loc[0,len(cluster_valuation.index)] = cluster_evaluation(y_val, y_pred)

                eval, cntg = cluster_evaluation(test_Labels, y_pred)
                print(cntg)
                print(eval)
                cluster_valuation = cluster_valuation._append(eval, ignore_index=True)
                cluster_valuation.reset_index()

                predict_out_path = Path(
                    model_path
                    + "/"
                    + cluster_alg
                    + "/Test/"
                    + "file_predictions_"
                    + str(ndims)
                    + "-dims_"
                    + str(hyper_para)
                    + "-clusters.csv"
                )
                predict_out_path.parent.mkdir(parents=True, exist_ok=True)
                test_df = pd.DataFrame(
                    {
                        "name": test_df["name"].tolist(),
                        "Label": test_Labels,
                        "Predict": y_pred,
                    }
                )
                test_df.to_csv(predict_out_path)

                fig = plot_clusters(
                    twoD_tsne_vector,
                    y_pred,
                    alg_name=cluster_alg,
                    hyper_para_name=hyper_para_name,
                    hyp_para=hyper_para,
                    ndims=ndims,
                )

                fig_name = Path(
                    model_path
                    + "/"
                    + cluster_alg
                    + "/Test/"
                    + "test_clustering_"
                    + str(ndims)
                    + "-dims_"
                    + str(hyper_para)
                    + "-clusters.png"
                )
                fig_name.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_name)
                plt.clf()

            cluster_valuation.insert(0, hyper_para_name, hyper_para_list)

            valuation_name = Path(
                model_path
                + "/"
                + cluster_alg
                + "/Test/"
                + "test_clustering_evaluation_"
                + str(ndims)
                + "-dims.csv"
            )
            valuation_name.parent.mkdir(parents=True, exist_ok=True)
            cluster_valuation.to_csv(valuation_name)


def get_clf_hyper_para(cluster_alg):
    if cluster_alg == "Kmeans":
        hyper_para_name = "n_clusters"
        random_state = 0
        hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    elif cluster_alg == "spectral":
        hyper_para_name = "n_clusters"
        assign_labels = "discretize"
        hyper_para_list = np.arange(2, 31, step=1)
    elif cluster_alg == "Aggloromative":
        hyper_para_name = "n_clusters"
        linkage = "ward"
        hyper_para_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    elif cluster_alg == "DBSCAN":
        hyper_para_name = "eps"
        min_samples = 2
        hyper_para_list = np.arange(5, 150, step=5)

    return hyper_para_name, hyper_para_list


def cv_train_model(
    X_train, y_train, alg_name: str, output_path: str, ndims: int, k_folds: int = 5
):
    """
    Wrapper for training a specific classification model using RandomizedSearchCV.
    Saves the best model and returns it along with its best CV score.
    """
    print(f"--- Training {alg_name} for {ndims} dimensions ---")

    if alg_name == "LR":
        model = LogisticRegression(max_iter=10000, random_state=42)
        param_grid = {
            "classifier__solver": ["saga", "lbfgs"],
            "classifier__penalty": ["l1", "l2", "elasticnet", None],
            "classifier__C": loguniform(0.001, 100),
        }
    elif alg_name == "MLP":
        model = MLPClassifier(
            early_stopping=True, n_iter_no_change=10, random_state=42, max_iter=1000
        )
        param_grid = {
            "classifier__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64, 32)],
            "classifier__activation": ["relu", "tanh", "logistic"],
            "classifier__alpha": [10, 1, 0.1, 0.001, 0.0001],
            "classifier__learning_rate_init": [0.1, 0.01, 0.001],
            "classifier__batch_size": [16, 32, 64, 128],
        }
    elif alg_name == "LSVC":
        model = LinearSVC(max_iter=10000, random_state=42)
        param_grid = {
            "classifier__penalty": ["l1", "l2"],
            "classifier__C": loguniform(0.001, 100),
        }
    elif alg_name == "RBF-SVC":
        model = SVC(kernel="rbf", random_state=42)
        param_grid = {
            "classifier__C": loguniform(0.001, 100),
            "classifier__gamma": loguniform(0.001, 10),
        }
    elif alg_name == "Sig-SVC":
        model = SVC(kernel="sigmoid", random_state=42)
        param_grid = {
            "classifier__C": loguniform(0.001, 100),
            "classifier__gamma": loguniform(0.001, 10),
        }
    elif alg_name == "Poly-SVC":
        model = SVC(kernel="poly", random_state=42)
        param_grid = {
            "classifier__degree": [2, 3, 4],
            "classifier__C": loguniform(0.001, 100),
            "classifier__gamma": loguniform(0.001, 10),
        }
    elif alg_name == "DT":
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [2, 4, 8, 16, 32, None],
            "classifier__max_features": ["sqrt", "log2", 0.5, 0.1, None],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 5, 10, 20],
        }
    elif alg_name == "RF":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "classifier__n_estimators": [2, 4, 8, 16, 32, 64, 128, 256, 512],
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [2, 4, 8, 16, 32, None],
            "classifier__max_features": ["sqrt", "log2", 0.5, 0.1, None],
        }
    elif alg_name == "HGBC":
        model = HistGradientBoostingClassifier(random_state=42, early_stopping=True)
        param_grid = {
            "classifier__learning_rate": loguniform(0.001, 10),
            "classifier__max_iter": [100, 200, 500, 1000],
            "classifier__max_depth": [2, 4, 8, 12, 16, 32, 64, 128, 256],
            "classifier__l2_regularization": [10.0, 5.0, 1.0, 0, 0.1, 0.01],
            "classifier__max_leaf_nodes": [7, 15, 31, 63],
        }
    elif alg_name == "Dummy":
        model = DummyClassifier(strategy="most_frequent")
        param_grid = {}
    else:
        print(f"Warning: Unknown model {alg_name}. Skipping.")
        return None, 0.0

    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", model)])

    n_cv_iter = 1 if alg_name == "Dummy" else 60
    if alg_name in ["RBF-SVC", "Poly-SVC"]:
        n_cv_iter = 60
    cv_folds = k_folds

    if alg_name == "Dummy":
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        best_cv_score = 0.5
    else:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=n_cv_iter,
            cv=cv_folds,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_cv_score = search.best_score_
        print(f"  Best CV ROC-AUC: {best_cv_score:.4f}")
        print(f"  Best Hyperparameters: {search.best_params_}")

    save_path_dir = Path(output_path) / alg_name
    save_path_dir.mkdir(parents=True, exist_ok=True)
    save_path_file = save_path_dir / f"{alg_name}_model_{ndims}dims.pkl"

    with open(save_path_file, "wb") as f:
        pickle.dump(best_model, f)
    print(f"  Saved best {alg_name} model to {save_path_file}")

    return best_model, best_cv_score


def evaluate_model(model, X_test, y_test):
    """Calculates all required metrics for a trained model on the test set."""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
        if y_scores.ndim > 1:
            y_scores = y_scores[:, 1]
        auc = roc_auc_score(y_test, y_scores)
    else:
        auc = 0.5

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": auc,
    }
    return metrics


def supervised_clf(
    output_path: str = "./results/trained_clf_models/",
    clf_alg_names: List = [],
    ndims_list: List = [128],
    train_cfg_path: str = "",
    n_train_malware: int = 0,
    n_train_benign: int = 0,
    test_cfg_path: str = "",
    n_test_malware: int = 0,
    n_test_benign: int = 0,
):
    print("Start supervised training")

    clf_save_path = Path(output_path)
    results_root_path = clf_save_path.parent
    d2v_model_path = results_root_path / "d2v_models"

    clf_save_path.mkdir(parents=True, exist_ok=True)

    k_folds = 100
    all_results = []
    all_model_summaries = []

    for ndims in ndims_list:
        print("\n======================================")
        print(f"Processing models for {ndims} dimensions")
        print("======================================")

        try:
            base_param = WL_parameters(dimensions=ndims)
        except NameError:
            print("ERROR: WL_parameters class is not defined. Please import it.")
            exit()

        print("--- Getting Training Data ---")
        X_train, y_train = load_and_generate_vectors(
            str(d2v_model_path),
            ndims,
            "Train",
            train_cfg_path,
            n_train_malware,
            n_train_benign,
            base_param,
        )
        print("--- Getting Testing Data ---")
        X_test, y_test = load_and_generate_vectors(
            str(d2v_model_path),
            ndims,
            "Test",
            test_cfg_path,
            n_test_malware,
            n_test_benign,
            base_param,
        )

        if X_train is None or y_train is None or X_test is None or y_test is None:
            print(f"Skipping {ndims} dimensions due to data generation failure.")
            continue

        all_clf_names = ["Dummy"] + clf_alg_names

        for alg_name in all_clf_names:
            (best_model, best_cv_score) = cv_train_model(
                X_train, y_train, alg_name, str(clf_save_path), ndims, k_folds=k_folds
            )

            if best_model is None:
                print(
                    f"Woa nelly, no model was returned from our training process for {alg_name}"
                )
                continue

            print(f"  Evaluating {alg_name} on test set...")
            test_metrics = evaluate_model(best_model, X_test, y_test)
            test_metrics["Model"] = alg_name
            test_metrics["Dimensions"] = ndims
            test_metrics["CV-ROC-AUC"] = best_cv_score

            model_summary_dict = get_model_summary_dict(alg_name, best_model, ndims)
            all_model_summaries.append(model_summary_dict)

            print(f"  Test Metrics: {test_metrics}")
            all_results.append(test_metrics)

        results_df = pd.DataFrame(all_results)
        if not results_df.empty:
            cols = [
                "Model",
                "Dimensions",
                "CV-ROC-AUC",
                "ROC-AUC",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
            ]
            results_df = results_df[cols]

        results_save_path = (
            clf_save_path / f"supervised_classification_results_{ndims}.csv"
        )
        results_df.to_csv(results_save_path, index=False)
        print(f"\nAll classification results saved to {results_save_path}")
        print(results_df.to_markdown(index=False, floatfmt=".4f"))

        models_df = pd.DataFrame(all_model_summaries)
        if not models_df.empty:
            cols_to_order = ["Model", "Dimensions"]
            new_cols = cols_to_order + [
                c for c in models_df.columns if c not in cols_to_order
            ]
            models_df = models_df[new_cols]

            models_save_path = clf_save_path / f"supervised_model_summaries_{ndims}.csv"
            models_df.to_csv(models_save_path, index=False)
            print(f"\nAll model summaries saved to {models_save_path}")
            print(f"\n--- Best Model Hyperparameter Summaries ({ndims})---")
            print(models_df.fillna("N/A").to_markdown(index=False))

    return results_df, models_df


def load_and_generate_vectors(
    d2v_model_path_str: str,
    ndims: int,
    data_split: str,
    cfg_data_path: str,
    n_malware: int,
    n_benign: int,
    base_param,
):
    """
    Tries to load vectors/labels from CSV. If not found, generates them
    from raw CFGs using the saved Doc2Vec model and saves them.
    """

    d2v_model_path = Path(d2v_model_path_str)

    n_samples = n_malware + n_benign
    vector_filename = (
        f"{data_split.lower()}_file_vectors_{ndims}_{n_samples}_samples.csv"
    )
    label_filename = f"file_labels_{ndims}_{n_samples}_samples.csv"

    vector_out_path = d2v_model_path / data_split / vector_filename
    label_out_path = d2v_model_path / data_split / label_filename

    # If the embeddings already exist, just load them
    try:
        X = pd.read_csv(vector_out_path, header=None).values
        y_df = pd.read_csv(label_out_path)
        y_str = y_df["Label"].values
        y = np.where(y_str == "Malware", 1, 0)
        print(f"Successfully loaded {data_split} data from CSV for {ndims} dims.")
        return X, y

    except FileNotFoundError:
        print(
            f"Vector/label CSVs not found for {data_split} ({ndims} dims). Generating..."
        )

    except Exception as e:
        print(f"Error loading CSVs: {e}. Will try to generate.")

    # If the embeddings don't exist, we load the Doc2Vec model, inference on the Wl hashes (words), and save the embeddings
    try:
        print(f"Load {data_split} CFG data from {cfg_data_path}")
        graphs, file_names, labels, n_samples = loadTestCFG(
            cfg_data_path, n_malware, n_benign
        )

        print(f"Creating WL hash words for {data_split} set")
        documents = createWLhash(graphs, base_param)

        model_name = d2v_model_path / f"d2v_model_{ndims}.model"
        d2v_model = Doc2Vec.load(str(model_name))

        print(f"Doc2Vec inference for {data_split} set")
        vectors = np.array([d2v_model.infer_vector(d.words) for d in documents])

        print(f"Saving generated {data_split} vectors to {vector_out_path}")
        vector_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(vectors).to_csv(vector_out_path, header=None, index=None)

        print(f"Saving generated {data_split} labels to {label_out_path}")
        label_out_path.parent.mkdir(parents=True, exist_ok=True)
        df_to_save = pd.DataFrame({"name": file_names, "Label": labels})
        df_to_save.to_csv(label_out_path, index=False)

        print("  Mapping generated labels: 'Malware'->1, 'Benign'->0")
        labels_numeric = np.where(np.array(labels) == "Malware", 1, 0)

        return vectors, labels_numeric

    except NameError as e:
        print(f"ERROR: Missing function: {e}. ")
        print(
            "Please ensure 'loadTestCFG', 'createWLhash', 'WL_parameters', and 'Doc2Vec' are imported."
        )
        return None, None

    except FileNotFoundError as e:
        print(f"ERROR: Model file not found: {e.filename}")
        print(
            "Please ensure STEP 1 (graph_embedding_training) has been run successfully."
        )
        return None, None

    except Exception as e:
        print(f"An unexpected error occurred during vector generation: {e}")
        return None, None


def get_model_summary_dict(alg_name: str, model_pipeline: Pipeline, ndims: int) -> dict:
    """
    Generates a dictionary of a model's key hyperparameters for DataFrame creation.
    """
    try:
        clf = model_pipeline.named_steps["classifier"]
        params = clf.get_params()

        summary_dict = {"Model": alg_name, "Dimensions": ndims}

        if alg_name == "LR":
            summary_dict.update(
                {
                    "Penalty": params.get("penalty"),
                    "C": params.get("C"),
                    "Solver": params.get("solver"),
                    "Max_Iter": params.get("max_iter"),
                }
            )

        elif alg_name == "MLP":
            summary_dict.update(
                {
                    "Hidden_Layers": str(params.get("hidden_layer_sizes")),
                    "Activation": params.get("activation"),
                    "Alpha (L2)": params.get("alpha"),
                    "Learning_Rate_Init": params.get("learning_rate_init"),
                    "Batch_Size": params.get("batch_size"),
                }
            )

        elif alg_name == "LSVC":
            summary_dict.update(
                {
                    "Penalty": params.get("penalty"),
                    "C": params.get("C"),
                    "Loss": params.get("loss"),
                    "Max_Iter": params.get("max_iter"),
                }
            )

        elif alg_name == "RBF-SVC":
            summary_dict.update(
                {
                    "Kernel": params.get("kernel"),
                    "C": params.get("C"),
                    "Gamma": params.get("gamma"),
                }
            )

        elif alg_name == "Sig-SVC":
            summary_dict.update(
                {
                    "Kernel": params.get("kernel"),
                    "C": params.get("C"),
                    "Gamma": params.get("gamma"),
                }
            )

        elif alg_name == "Poly-SVC":
            summary_dict.update(
                {
                    "Degree": params.get("degree"),
                    "C": params.get("C"),
                    "Gamma": params.get("gamma"),
                }
            )

        elif alg_name == "DT":
            summary_dict.update(
                {
                    "Criterion": params.get("criterion"),
                    "Max_Depth": params.get("max_depth"),
                    "Max_Features": params.get("max_features"),
                    "Min_Samples_Split": params.get("min_samples_split"),
                    "Min_Samples_Leaf": params.get("min_samples_leaf"),
                }
            )

        elif alg_name == "RF":
            summary_dict.update(
                {
                    "Num_Estimators": params.get("n_estimators"),
                    "Criterion": params.get("criterion"),
                    "Max_Depth": params.get("max_depth"),
                    "Max_Features": params.get("max_features"),
                }
            )

        elif alg_name == "HGBC":
            summary_dict.update(
                {
                    "Learning_Rate": params.get("learning_rate"),
                    "Max_Iter": params.get("max_iter"),
                    "Max_Depth": params.get("max_depth"),
                    "L2_Regularization": params.get("l2_regularization"),
                    "Max Leaf Nodes": params.get("max_leaf_nodes"),
                }
            )

        elif alg_name == "Dummy":
            summary_dict.update({"Strategy": params.get("strategy")})

        return summary_dict

    except Exception as e:
        return {"Model": alg_name, "Dimensions": ndims, "Error": str(e)}


def explain_models(
    model_root_dir: str,
    clf_alg_names: List[str],
    ndims: int = 16,
    train_cfg_path: str = "",
    n_train_malware: int = 0,
    n_train_benign: int = 0,
    test_cfg_path: str = "",
    n_test_malware: int = 0,
    n_test_benign: int = 0,
    compute_shap: bool = False,
):
    clf_save_path = Path("./results/trained_clf_models")
    results_root_path = clf_save_path.parent
    d2v_model_path = results_root_path / "d2v_models"
    clf_save_path.mkdir(parents=True, exist_ok=True)

    shap_plot_dir = Path("./results/shap_plots")
    shap_plot_dir.mkdir(parents=True, exist_ok=True)
    clf_save_path.mkdir(parents=True, exist_ok=True)

    num_plots = 9
    cols = 3
    rows = math.ceil(num_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    try:
        base_param = WL_parameters(dimensions=ndims)
    except NameError:
        print("ERROR: WL_parameters class is not defined. Please import it.")
        exit()

    print("--- Getting Training Data ---")
    X_train, y_train = load_and_generate_vectors(
        str(d2v_model_path),
        ndims,
        "Train",
        train_cfg_path,
        n_train_malware,
        n_train_benign,
        base_param,
    )

    print("--- Getting Test Data ---")
    X_test, y_test = load_and_generate_vectors(
        str(d2v_model_path),
        ndims,
        "Test",
        test_cfg_path,
        n_test_malware,
        n_test_benign,
        base_param,
    )

    X_train_background = shap.utils.sample(X_train, 100)
    X_test_sample = shap.utils.sample(X_test, 50)

    feature_names = [f"Feature {i + 1}" for i in range(ndims)]

    valid_plot_count = 0

    for idx, alg_name in enumerate(clf_alg_names):
        ax = axes[idx]
        try:
            model_path = (
                Path(model_root_dir) / alg_name / f"{alg_name}_model_{ndims}dims.pkl"
            )
            with open(model_path, "rb") as f:
                model_pipeline = pickle.load(f)
                model = model_pipeline.named_steps[
                    "classifier"
                ]  # this is trained on scaled data
                scaler = model_pipeline.named_steps["scaler"]
        except FileNotFoundError:
            print(f"Model file not found for {alg_name} at {model_path}")
            continue
        full_alg_name = alg_name
        mapping = {
            "DT": "Simple Decision Tree",
            "RF": "Random Forest",
            "HGBC": "Hist Grad Boosting",  # Shortened for plot titles
            "LR": "Logistic Regression",
            "MLP": "Multi-layer Perceptron",
            "LSVC": "Linear SVC",
            "RBF-SVC": "RBF SVC",
            "Poly-SVC": "Polynomial SVC",
            "Sig-SVC": "Sigmoid SVC",
        }
        if alg_name in mapping:
            full_alg_name = mapping[alg_name]

        print(f"\n==========Processing best model for {full_alg_name}==========")

        X_train_background_scaled = scaler.transform(X_train_background)
        X_test_sample_scaled = scaler.transform(X_test_sample)

        print("-----Quick Model Summary-----")
        if alg_name in ["LR", "LSVC"]:
            print(f"\n--- {alg_name} Coefficients (Magnitude on Scaled Features): ---")
            if hasattr(model, "coef_"):
                coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                top_indices = np.argsort(np.abs(coefs))[-10:][::-1]
                print(f"{'Feature':<15} | {'Coefficient':<12} | {'Direction'}")
                for i in top_indices:
                    direction = "Positive (+)" if coefs[i] > 0 else "Negative (-)"
                    print(f"{feature_names[i]:<15}: {coefs[i]:.5f} | {direction}")
            else:
                print("Model is linear but no coef_ attribute found.")

        elif alg_name in ["DT", "RF", "HGBC"]:
            print(f"\n--- {alg_name} Feature Importance (Gini/Entropy) ---")
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[-10:][::-1]
                for i in top_indices:
                    print(f"{feature_names[i]:<15}: {importances[i]:.5f}")

        if False:
            print("\n--- Permutation Importance ---")
            perm_results = permutation_importance(
                model_pipeline,
                X_test,
                y_test,
                n_repeats=50,
                random_state=42,
                n_jobs=-1,
                scoring="roc_auc",
            )
            perm_means = perm_results["importances_mean"]
            perm_std = perm_results["importances_std"]
            top_perm_indices = np.argsort(perm_means)[-10:]

            print(f"{'Feature':<15} | {'Mean Drop in Accuracy':<22} | {'Std Dev'}")
            for i in top_perm_indices[::-1]:
                print(f"{feature_names[i]:<15} | {perm_means[i]:.5f}")
            ax.boxplot(
                perm_results["importances"][top_perm_indices].T,
                vert=False,
                tick_labels=[feature_names[i] for i in top_perm_indices],
            )
            ax.set_title(f"Permutation Importance (ROC-AUC) - {alg_name}", fontsize=10)
            ax.set_xlabel("Decrease in ROC-AUC Score")

        if False:
            plt.figure(figsize=(10, 6))
            plt.boxplot(
                perm_results["importances"][top_perm_indices].T,
                vert=False,
                tick_labels=[feature_names[i] for i in top_perm_indices],
            )
            plt.title(f"Permutation Importance (ROC-AUC) - {alg_name}")
            plt.xlabel("Decrease in ROC-AUC Score")
            plt.tight_layout()
            plt.savefig(f"xai_results/{alg_name}_perm_importance.png")
            plt.close()

        if not compute_shap:
            print("Skipping Shapley value computations...")
            continue

        if alg_name not in [
            "DT",
            "RF",
            "LSVC",
            "LR",
            "Poly-SVC",
            "Sig-SVC",
            "RBF-SVC",
            "MLP",
            "HGBC",
        ]:
            print(
                f"Sorry but {alg_name} is too complex to feasibly calculate Shaply values for"
            )
            continue

        print("\n--- SHAP Value Computation ---")
        if alg_name in ["DT", "RF"]:
            explainer = shap.TreeExplainer(
                model,
                feature_perturbation="interventional",
                model_output="probability",
                data=X_train_background_scaled,
            )
            shap_values_raw = explainer.shap_values(X_test_sample_scaled)
            shap_values = shap_values_raw[:, :, 1]
            expected_value = explainer.expected_value[1]
        elif alg_name in ["RBF-SVC", "Poly-SVC", "Sig-SVC", "MLP", "HGBC"]:
            explainer = shap.KernelExplainer(model.predict, X_train_background_scaled)
            shap_values = explainer.shap_values(X_test_sample_scaled)
            expected_value = explainer.expected_value
        else:
            explainer = shap.LinearExplainer(
                model, X_train_background_scaled, feature_perturbation="interventional"
            )
            shap_values = explainer.shap_values(X_test_sample_scaled)
            expected_value = explainer.expected_value

        print(f"SHAP Expected Value (Base Rate): {expected_value:.5f}")

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_test_sample_scaled,
            feature_names=feature_names,
            show=False,
            max_display=16,
        )

        plot_filename = shap_plot_dir / f"SHAP_Summary_Plot_{alg_name}_{ndims}dims.png"
        plt.title(f"SHAP Summary Plot for {alg_name} (Dimensions: {ndims})")
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        print(f"===SHAP Summary Plot saved to: {plot_filename}===")
    if False:
        for i in range(idx + 1, len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        summary_save_path = shap_plot_dir / "permutation_importance_summary.png"
        print(f"\nSaving summary plot to: {summary_save_path}")
        plt.savefig(summary_save_path, dpi=300)  # Higher DPI for papers
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Supervised Learning on CFG Embeddings"
    )
    parser.add_argument(
        "dimensions",
        type=int,
        nargs="+",
        help="Dimensions of Embeddings (e.g., 64 128 256)",
    )
    parser.add_argument(
        "-e", "--embed", action="store_true", help="Compute Graph Embeddings"
    )
    parser.add_argument(
        "-t",
        "--train_evaluate",
        action="store_true",
        help="Train and evaluate new models",
    )
    parser.add_argument(
        "-s", "--explain", action="store_true", help="Conduct XAI Summary"
    )
    args = parser.parse_args()

    print("Binary file analysis using Graph2Vec and classification")

    # prog_path = "./Binary_data/" # binary files not available
    cfg_path = "./data/CFG_dataset/"
    output_path = "./results/"
    # info_path = "./data/CFG_dataset/class_labels_no_dupes.csv"

    train_cfg_path = cfg_path + "Train_CFG/"
    n_train_malware = 3000  # Maximum 3000,  for quick run use 300
    n_train_benign = 3000  # Maximum 3000,  for quick run use 300

    ## Testing binary files
    test_cfg_path = cfg_path + "Test_CFG/"
    n_test_malware = 1000  # Maximum 1000, for quick run use 100
    n_test_benign = 1000  # Maximum 1000, for quick run use 100

    clf_alg_names = [
        "RF",
        "LR",
        "MLP",
        "LSVC",
        "RBF-SVC",
        "Poly-SVC",
        "Sig-SVC",
        "DT",
        "HGBC",
    ]
    clf_alg_names = ["MLP", "HGBC"]
    clf_model_output_path = output_path + "trained_clf_models/"
    ndims_list = args.dimensions

    if args.embed:
        print(
            "******************* STEP: 1 Create Vector Representation for CFGs using Graph2Vec *******************"
        )
        param, model_path = graph_embedding_training(
            train_cfg_path, output_path, ndims_list
        )
        print(model_path)
    else:
        print(
            "--- SKIPPING STEP 1 (Assuming D2V models already exist in ./results/d2v_models/) ---"
        )

    if args.train_evaluate:
        print(
            "******************* STEP: 2 Supervised classification training with hold-out validation. *******************"
        )
        results_df, model_summaries_df = supervised_clf(
            clf_model_output_path,
            clf_alg_names,
            ndims_list,
            train_cfg_path=train_cfg_path,
            n_train_malware=n_train_malware,
            n_train_benign=n_train_benign,
            test_cfg_path=test_cfg_path,
            n_test_malware=n_test_malware,
            n_test_benign=n_test_benign,
        )

    if args.explain:
        print("******************* STEP: 3 XAI *******************")
        for ndims in ndims_list:
            explain_models(
                clf_model_output_path,
                clf_alg_names,
                ndims,
                train_cfg_path,
                n_train_malware,
                n_train_benign,
                test_cfg_path,
                n_test_malware,
                n_test_benign,
                compute_shap=True,
            )

    print("******************* Process Finished *******************")


if __name__ == "__main__":
    main()
