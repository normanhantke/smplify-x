
import sys
import os
import os.path as osp

import numpy as np
import pickle
import yaml
from scipy.spatial import procrustes

import trimesh
import configargparse
import smplx
import torch
from human_body_prior.tools.model_loader import load_vposer

import utils
from data_parser import create_dataset
from camera import create_camera


# ground_truth camera parameters for the EHF dataset:
EHF_GT_CAMERA_FOCAL_LENGTH = np.array([1498.22426237, 1498.22426237], dtype=np.float32)
EHF_GT_CAMERA_CENTER = np.array([[790.263706, 578.90334 ]], dtype=np.float32)
EHF_GT_CAMERA_ROTATION = np.array([[[-2.98747896,0,0], [0, 0.01172457, 0], [0,0, -0.05704687]]], dtype=np.float32)
EHF_GT_CAMERA_TRANSLATION = np.array([[-0.03609917, 0.43416458, 2.37101226]], dtype=np.float32)



##### this code assumes that there is only one person detected in each image

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
        errors["v2v_error"][i], model_scale_to_meter_scale_factor = v2v_error( ground_truth_meshes[i], os.path.join( result_meshes[i], "000.obj" ) )
        errors["joint_error"][i] = joint_error( data, os.path.join( result_results[i], "000.pkl" ), model_scale_to_meter_scale_factor, vp, joint_mapper, **cfg )

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
    return v2v_error * norm1, norm1

def joint_error( joints_1, joints_2, model_scale_to_meter_scale_factor, vp, joint_mapper, **args ):

    # smplify-x outputs the results as float32, so we have to do everything in float32 here...
    dtype=np.float32
    torch_dtype=torch.float32

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
                right_hand_pose=torch.from_numpy(results["right_hand_pose"]),
                dtype=torch_dtype )
    results_joints = x.joints

    # drop depth dimension: "We do not provide 3D keypoints (the 3D fields in the JSON file are due to legacy reasons). OpenPose needs 
    # more than 1 view to lift 2D keypoints to 3D using multi-view information and triangulation. No multi-view information is available 
    # here, only single-view." (https://github.com/vchoutas/smplify-x/issues/56#issuecomment-561791619) 
    
    ### project model model joints into pixel space
    focal_length = args.get('focal_length')
    camera = create_camera( focal_length_x=focal_length,
                            focal_length_y=focal_length,
                            center=torch.from_numpy( EHF_GT_CAMERA_CENTER),
                            rotation=torch.from_numpy(results["camera_rotation"]),
                            translation=torch.from_numpy(results["camera_translation"]),
                            dtype=torch_dtype )
    joints_2 = camera(results_joints).cpu().detach().numpy()[0]

    # drop the third dimension
    joints_1 = joints_1['keypoints'][0][:,:2]

    norm2 = np.linalg.norm(joints_2 - np.mean(joints_2, 0) )

    joints_1, joints_2, _ = procrustes( joints_1, joints_2 )

    joint_error = np.linalg.norm( joints_1 - joints_2, axis=1 ).mean()

    # scale the error into the model's original scale
    joint_error *= norm2

    # and now into meter scale
    joint_error *= model_scale_to_meter_scale_factor

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


