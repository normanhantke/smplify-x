import configargparse
import utils
import os
import os.path as osp
try:
    import cPickle as pickle
except ImportError:
    import pickle

def generate_depthmaps(**args):

    ground_truth_meshes_folder = args.pop('ground_truth_meshes_folder')
    ground_truth_meshes_folder = osp.expandvars(ground_truth_meshes_folder)

    output_folder = args.pop('output_folder')
    output_folder = osp.expandvars(output_folder)

    image_size = args.pop("image_size")

    orig_size = args.pop("orig_image_size")

    if not os.path.exists( output_folder ):
        os.makedirs(       output_folder )

    ground_truth_meshes = [os.path.join(ground_truth_meshes_folder, name) for name in os.listdir(ground_truth_meshes_folder) if os.path.isfile(os.path.join(ground_truth_meshes_folder, name))]
    ground_truth_meshes.sort()

    for i in range( len(ground_truth_meshes) ):
        print("Calculating depthmap #", i)
        depthmap = utils.render_mesh_to_depthmap( ground_truth_meshes[i], image_size=image_size, orig_size=orig_size )
        filename = osp.join(output_folder, str(i+1).zfill(2) + "_img_depthmap.pkl" )
        with open( filename, 'wb') as f:
            pickle.dump( depthmap, f )



def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter

    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'calculates depthmaps from meshes'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='SMPLifyX')

    parser.add_argument('--ground_truth_meshes_folder',
                        required=True,
                        help='The directory that contains the pseudo ground truth meshes.')
    parser.add_argument('--output_folder',
                        required=True,
                        help='The output directory where the depthmaps are stored, for example the dataset folder that also contains the images and keypoints folders.')
    parser.add_argument('--image_size',
                        default=[512,424],
                        type=int, nargs=2,
                        help='Image size of the depthmaps.')
    parser.add_argument('--orig_image_size',
                        default=[1600,1200],
                        type=int, nargs=2,
                        help='The size of the RGB images')

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


if __name__ == "__main__":
    args = parse_config()
    generate_depthmaps(**args)