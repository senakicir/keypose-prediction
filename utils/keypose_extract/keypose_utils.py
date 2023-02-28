
def keypose_directory(dataset, threshold=250, kp_suffix=""): 
    if kp_suffix == "":
        keypose_dir = "/" + dataset +"_keypose_data_"+  str(threshold)
    else:
        keypose_dir = "/" + dataset +"_keypose_data_"+  str(threshold) + "_" + kp_suffix
    return keypose_dir

#rewrites existing data from data_loaded into data, if it exists
def set_data(data, data_loaded) :
    # keyposes and loc do not need to be overwritten, so keep them
    data["keyposes"] = data_loaded["keyposes"]
    data["loc"] = data_loaded["loc"]
    
    if "naive_labels" in data_loaded:
        data["naive_labels"] = data_loaded["naive_labels"]
    else:
        data["naive_labels"] = {}

    if "naive_distances" in data_loaded:
        data["naive_distances"] = data_loaded["naive_distances"]
    else:
        data["naive_distances"] = {}

    if "naive_inds" in data_loaded:
        data["naive_inds"] = data_loaded["naive_inds"]
    else:
        data["naive_inds"] = {}

    return data