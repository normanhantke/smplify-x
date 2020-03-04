import os
import os.path as osp

import numpy as np
import pickle
import yaml
from scipy.spatial import procrustes
from scipy.spatial import transform

import trimesh
import configargparse
import smplx
import torch
from human_body_prior.tools.model_loader import load_vposer

import utils
from data_parser import create_dataset


##### this code assumes that there is only one person detected in each image

def evaluate_results(**args):

    ground_truth_meshes_folder = args.pop('ground_truth_meshes_folder')
    ground_truth_meshes_folder = osp.expandvars(ground_truth_meshes_folder)

    SMPLifyX_output_folder = args.pop('SMPLifyX_output_folder')
    SMPLifyX_output_folder = osp.expandvars(SMPLifyX_output_folder)

    joint_regressor_folder = args.pop('joint_regressor_folder')
    joint_regressor_folder = osp.expandvars(joint_regressor_folder)

    ground_truth_meshes = [os.path.join(ground_truth_meshes_folder, name) for name in os.listdir(ground_truth_meshes_folder) if os.path.isfile(os.path.join(ground_truth_meshes_folder, name))]
    result_meshes       = [os.path.join(os.path.join( SMPLifyX_output_folder, "meshes" ), name) for name in os.listdir(os.path.join( SMPLifyX_output_folder, "meshes" )) if os.path.isdir(os.path.join(os.path.join( SMPLifyX_output_folder, "meshes" ), name))]

    ground_truth_meshes.sort()
    result_meshes.sort()
    # check identical number of files
    assert( len(ground_truth_meshes) == len(result_meshes)), "Number of files must be identical in each of the folders."

    errors = dict()
    errors["v2v_error"] = np.zeros( len(ground_truth_meshes) )
    errors["joint_error"] = np.zeros( len(ground_truth_meshes) )
    for i in range( len(ground_truth_meshes) ):
        errors["v2v_error"][i] = v2v_error( ground_truth_meshes[i], os.path.join( result_meshes[i], "000.obj" ) )
        errors["joint_error"][i] = joint_error( ground_truth_meshes[i], os.path.join( result_meshes[i], "000.obj"), joint_regressor_folder )

    with open( os.path.join( SMPLifyX_output_folder, "errors.pkl" ), 'wb' ) as handle:
        pickle.dump( errors, handle )

    print( "Mean v2v_error:", errors["v2v_error"].mean() )
    print( "Mean joint_error:", errors["joint_error"].mean())


def v2v_error( mesh_1, mesh_2 ):

    mesh_1_vertices = trimesh.load_mesh(mesh_1, process=False).vertices
    mesh_2_vertices = trimesh.load_mesh(mesh_2, process=False).vertices

    norm1 = np.linalg.norm(mesh_1_vertices - np.mean(mesh_1_vertices, 0) )
    
    mesh_1_vertices, mesh_2_vertices, _ = procrustes( mesh_1_vertices, mesh_2_vertices )

    v2v_error = np.linalg.norm( mesh_1_vertices-mesh_2_vertices, axis=1 ).mean()
    
    # multiply error by norm, to have it in meter scale
    return v2v_error * norm1

def joint_error( mesh_1, mesh_2, joint_regressor_folder ):

    mesh_1_vertices = trimesh.load_mesh(mesh_1, process=False).vertices
    mesh_2_vertices = trimesh.load_mesh(mesh_2, process=False).vertices

    with open( os.path.join( joint_regressor_folder, "SMPLX_to_J14.pkl" ), "rb" ) as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        joint_regressor = u.load()
    
    joints_1 = joint_regressor.dot(mesh_1_vertices)
    joints_2 = joint_regressor.dot(mesh_2_vertices)

    norm1 = np.linalg.norm(joints_1 - np.mean(joints_1, 0) )
    
    joints_1, joints_2, _ = procrustes( joints_1, joints_2 )

    joint_error = np.linalg.norm( joints_1-joints_2, axis=1 ).mean()

    # multiply error by norm, to have it in meter scale
    return joint_error * norm1

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
    parser.add_argument('--joint_regressor_folder',
                        required=True,
                        help='The directory that contains the joint regressor pkl file.')
    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


if __name__ == "__main__":
    args = parse_config()
    evaluate_results(**args)


