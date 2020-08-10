import yaml
import os
import os.path as osp
import numpy as np

def create_create_yaml_files(output_dir, cfg_yaml_filename="../cfg_files/fit_smplx.yaml"):

    depth_weights = np.array([1,1,1,1,1])
    depth_weight_scales = np.logspace(-5,3, 9)

    if not os.path.exists( output_dir ):
        os.makedirs(       output_dir )

    for i in range( len(depth_weight_scales) ):
        curr_depth_weights = depth_weights * depth_weight_scales[i]

        with open(cfg_yaml_filename, "r") as ymlfile: 
            cfg = yaml.load(ymlfile) 

        cfg["depth_weights"] = curr_depth_weights.tolist()

        filename = "smplify-x_experiment_" + str(i).zfill(4) + ".yaml"

        with open(osp.join(output_dir, filename), 'w') as conf_file:
            yaml.dump(cfg, conf_file)