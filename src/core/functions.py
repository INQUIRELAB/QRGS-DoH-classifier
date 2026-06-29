import logging
import os
import numpy as np
import os
import json
from sklearn.decomposition import PCA
from preprocessing import doh
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def load_dataset(
        data_dir=None,
        args_dict=None,
        run_results_dir=None):

    logging.info(f" Loading {args_dict['dataset']} data...")

    processed_filename = f"{args_dict['dataset']}"

    processed_file_path = os.path.join(
        data_dir,
        processed_filename + ".csv"
    )
    metadata_file_path = os.path.join(
        data_dir,
        processed_filename + ".json"
    )

    if args_dict["dataset"] == "doh":
        df = doh.prepare_data(data_dir, args_dict)
    else:
        raise Exception("Dataset not specified")


    # change all zeros to -1 if qnn-reup which uses EstimatorQNN
    if args_dict['algorithm'] == 'qnn-reup':
        df['Class'] = df['Class'].apply(lambda x: -1 if x == 0 else x)

    # Calculate the percent positive class
    percent_positive_class = round(df[df["Class"] == 1]["Class"].sum()/len(df["Class"]), 2)
    logging.info(f"* Percent Positive Class: {percent_positive_class}")

    # ALWAYS MAKE SURE WHEN ADDING DATASETS THAT LABEL IS IN LAST COLUMN
    # Map features and labels to numpy arrays
    X = df.iloc[:, :-1].to_numpy()
    Y = df.iloc[:, -1:].to_numpy().astype("int").flatten()

    # Optional feature selection (global, prior to padding).
    selected_feature_indices = None
    selected_feature_scores = None
    k = int(args_dict.get("feature_select_number") or 0)
    if args_dict.get("feature_select") and k > 0:
        if k >= X.shape[1]:
            logging.info(f"* feature_select_number ({k}) >= num_features ({X.shape[1]}), skipping selection")
        else:
            logging.info(f"* Selecting top {k} features (mutual information)")
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X = selector.fit_transform(X, Y)
            selected_feature_indices = selector.get_support(indices=True).tolist()
            # scores_ can contain NaNs if MI can't be estimated for a feature; convert safely
            scores = getattr(selector, "scores_", None)
            if scores is not None:
                selected_feature_scores = [None if (s is None or np.isnan(s)) else float(s) for s in scores.tolist()]

    # Optional PCA (dimensionality reduction), applied after feature selection and before padding.
    pca_components = args_dict.get("pca_components")
    pca_applied = False
    pca_explained_variance_ratio = None
    if pca_components is not None:
        try:
            pca_components = int(pca_components)
        except Exception:
            pca_components = 0
    if pca_components and pca_components > 0:
        if pca_components >= X.shape[1]:
            logging.info(
                f"* pca_components ({pca_components}) >= num_features ({X.shape[1]}), skipping PCA"
            )
        else:
            logging.info(f"* Applying PCA with n_components={pca_components}")
            pca = PCA(n_components=pca_components, random_state=args_dict.get("random_seed"))
            X = pca.fit_transform(X)
            pca_applied = True
            if getattr(pca, "explained_variance_ratio_", None) is not None:
                pca_explained_variance_ratio = pca.explained_variance_ratio_.tolist()

    # Add zero padding
    if args_dict['algorithm'] == 'qnn-reup':
        # Calculate the number of blocks and padding needed
        if (args_dict['q_reup_method'] == "asymmetrical"):
            if (X.shape[1] < args_dict['q_num_qubit']):
                raise Exception("Number of Input Features MUST be greater than, or equal to number of qubits")
            block_size = 3 * args_dict['q_num_qubit'] # A block 
            remainder = X.shape[1] % block_size
            if remainder != 0:
                padding_needed = block_size - remainder
            else:
                padding_needed = 0
            X = np.pad(X, ((0, 0), (0, padding_needed)), 'constant')
        else:
            block_size = 3
            num_blocks = int(np.ceil(X.shape[1] / block_size))
            total_features_with_padding = num_blocks * block_size
            padding_needed = total_features_with_padding - X.shape[1] 

            # Add zero padding to each sample
            X = np.pad(X, ((0, 0), (0, padding_needed)), 'constant')

    # write csv
    data = np.concatenate((X, Y[:, np.newaxis].astype(int)), axis=1)
    np.savetxt(processed_file_path, data, delimiter=",")

    # Log dataset metadata
    metadata = {
        "records": len(Y),
        "percent_positive_class_prior_to_downsampling": percent_positive_class,
        "x_shape": [X.shape[0], X.shape[1]],
        "x_min_max":[np.min(X), np.max(X)],
        "y_shape": Y.shape,
        "y_unique_values": np.unique(Y).tolist(),
        "filename": processed_filename + ".csv",
        "feature_selection": {
            "enabled": bool(args_dict.get("feature_select")),
            "k": k,
            "selected_feature_indices": selected_feature_indices,
            "all_feature_scores": selected_feature_scores,
        },
        "pca": {
            "enabled": bool(pca_applied),
            "n_components": int(pca_components) if pca_components else 0,
            "explained_variance_ratio": pca_explained_variance_ratio,
        },
    }

    with open(metadata_file_path, "w") as outfile:
        json.dump(metadata, outfile)

    return (processed_filename, processed_file_path, metadata_file_path)
    
