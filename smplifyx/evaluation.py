
import sys
import os

import os.path as osp

import numpy as np
import pickle
import yaml
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

import trimesh
import configargparse
import smplx
import torch
from human_body_prior.tools.model_loader import load_vposer

import utils
from data_parser import create_dataset

##### this code assumes that there is only one person detected in each image

########################################
##### TODO PROCRUSTES ALIGNMENT ########
########################################


def evaluate_results(**args):

    ground_truth_meshes_folder = args.pop('ground_truth_meshes_folder')
    ground_truth_meshes_folder = osp.expandvars(ground_truth_meshes_folder)

    SMPLifyX_output_folder = args.pop('SMPLifyX_output_folder')
    SMPLifyX_output_folder = osp.expandvars(SMPLifyX_output_folder)

    with open(os.path.join( SMPLifyX_output_folder, "conf.yaml" ), 'r') as ymlfile: 
        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    vposer_ckpt = cfg.pop('vposer_ckpt')
    vp, ps = load_vposer(vposer_ckpt)

    ground_truth_meshes = [os.path.join(ground_truth_meshes_folder, name) for name in os.listdir(ground_truth_meshes_folder) if os.path.isfile(os.path.join(ground_truth_meshes_folder, name))]
    result_meshes       = [os.path.join(os.path.join( SMPLifyX_output_folder, "meshes" ), name) for name in os.listdir(os.path.join( SMPLifyX_output_folder, "meshes" )) if os.path.isdir(os.path.join(os.path.join( SMPLifyX_output_folder, "meshes" ), name))]
    result_results      = [os.path.join(os.path.join( SMPLifyX_output_folder, "results" ), name) for name in os.listdir(os.path.join( SMPLifyX_output_folder, "results" )) if os.path.isdir(os.path.join(os.path.join( SMPLifyX_output_folder, "results" ), name))]

    ground_truth_meshes.sort()
    result_meshes.sort()
    result_results.sort()
    # check identical number of files
    assert( len(ground_truth_meshes) == len(result_meshes)), "Number of files must be identical in each of the folders."

    dataset_obj = create_dataset(**cfg)
    joint_mapper = utils.JointMapper(dataset_obj.get_model2data())

    errors = dict()
    errors["v2v_error"] = np.zeros( len(ground_truth_meshes) )
    errors["joint_error"] = np.zeros( len(ground_truth_meshes) )
    for i, data in enumerate(dataset_obj):
        errors["v2v_error"][i] = v2v_error( ground_truth_meshes[i], os.path.join( result_meshes[i], "000.obj" ) )
        errors["joint_error"][i] = joint_error( data, os.path.join( result_results[i], "000.pkl" ), vp, joint_mapper, **cfg )

    with open( os.path.join( SMPLifyX_output_folder, "errors.pkl" ), 'wb' ) as handle:
        pickle.dump( errors, handle )

    print( "Mean v2v_error:", errors["v2v_error"].mean() )
    print( "Mean joint_error:", errors["joint_error"].mean())


def v2v_error( mesh_1, mesh_2 ):
    mesh_1_vertices = trimesh.load_mesh(mesh_1).vertices
    mesh_2_vertices = trimesh.load_mesh(mesh_2).vertices

    norm1 = np.linalg.norm(mesh_1_vertices)

    mesh_1_vertices, mesh_2_vertices, _ = procrustes( mesh_1_vertices, mesh_2_vertices )

    v2v_error = np.linalg.norm( mesh_1_vertices-mesh_2_vertices, axis=1 ).mean()

    return v2v_error

def joint_error( joints_1, joints_2, vp, joint_mapper, **args ):

    model_path = args.pop("model_folder")
    model = smplx.create(model_path=model_path, joint_mapper=joint_mapper, **args)

    results = np.load( joints_2 , allow_pickle=True )

    # according to https://github.com/nghorbani/human_body_prior/blob/master/notebooks/vposer_decoder.ipynb
    # body_pose is decoded to the 63 parameters like this:
    body_pose = vp.decode(torch.from_numpy(results["body_pose"]), output_type='aa').view(-1, 63)

    x = model(  betas=torch.from_numpy(results["betas"]), 
                global_orient=torch.from_numpy(results["global_orient"]),
                body_pose=body_pose, 
                jaw_pose=torch.from_numpy(results["jaw_pose"]), 
                leye_pose=torch.from_numpy(results["leye_pose"]), 
                reye_pose=torch.from_numpy(results["reye_pose"]), 
                expression=torch.from_numpy(results["expression"]), 
                transl=torch.from_numpy(results["camera_translation"]),
                left_hand_pose=torch.from_numpy(results["left_hand_pose"]), 
                right_hand_pose=torch.from_numpy(results["right_hand_pose"]) )
    results_joints = x.joints

    joints_1 = joints_1['keypoints'][0]
    joints_2 = results_joints.cpu().detach().numpy()[0]

    norm1 = np.linalg.norm(joints_1)

    joints_1, joints_2, _ = procrustes( joints_1, joints_2 )

    joint_error = np.linalg.norm( joints_1 - joints_2, axis=1 ).mean()

    return joint_error

def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter

    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'quantitative evaluation of SMPLifyX results'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='SMPLifyX')

    parser.add_argument('--ground_truth_meshes_folder',
                        required=True,
                        help='The directory that contains the pseudo ground truth meshes.')
    parser.add_argument('--SMPLifyX_output_folder',
                        required=True,
                        help='The directory that contains the SMPLify-X output that is to be evaluated.')
    parser.add_argument('--vposer_ckpt', type=str, default='',
                        help='The path to the V-Poser checkpoint')
    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


if __name__ == "__main__":
    args = parse_config()
    evaluate_results(**args)


