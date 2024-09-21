import argparse
from monoloco.test_module import MonoLocoPredictor
from openpifpaf import decoder, network, visualizer, show, logger

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
decoder.cli(parser)
logger.cli(parser)
network.Factory.cli(parser)
show.cli(parser)
visualizer.cli(parser)
args = parser.parse_args()
# Example usage of the class
# args = {
#     'images': ['docs/002282.png'],
#     'mode': 'mono',  # 'stereo' or 'mono'
#     'output_directory': 'output_directory',
#     'output_types': 'multi',
#     'batch_size': 1,
#     'glob': None,
#     'path_gt': None,
#     'instance_threshold': None,
#     'decoder': None,
#     'profile_decoder': None,
#     'cif_th': 0.5,
#     'seed_threshold': 0.5,
#     'caf_th': 0.3,
#     'decoder_workers': 1,
#     'tr_single_pose_threshold': None,
#     'tr_multi_pose_threshold': None,
#     'tr_multi_pose_n': 3,
#     'tr_minimum_threshold': 0.1,
#     'dense_connections': 0.0,
#     'trackingpose_track_recovery': False,
#     'posesimilarity_distance': 'euclidean',
#     'keypoint_threshold': 0.0,
# }
print(args.mode)
#args = argparse.Namespace(**args)
predictor = MonoLocoPredictor(args)
output = predictor.predict()
print(output)