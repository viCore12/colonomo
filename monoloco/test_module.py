import os
import glob
import json
import logging
import time
import numpy as np
import torch
import PIL
import openpifpaf
from collections import defaultdict

from monoloco.network import Loco, factory_for_gt, load_calibration, preprocess_pifpaf
from openpifpaf import datasets, decoder, network, visualizer, show, Predictor
from openpifpaf.predict import out_name

OPENPIFPAF_MODEL = 'https://drive.google.com/uc?id=1b408ockhh29OLAED8Tysd2yGZOo0N_SQ'
MONOLOCO_MODEL_KI = 'https://drive.google.com/uc?id=1krkB8J9JhgQp4xppmDu-YBRUxZvOs96r'
MONOLOCO_MODEL_NU = 'https://drive.google.com/uc?id=1BKZWJ1rmkg5AF9rmBEfxF1r8s8APwcyC'
MONSTEREO_MODEL = 'https://drive.google.com/uc?id=1xztN07dmp2e_nHI6Lcn103SAzt-Ntg49'

LOG = logging.getLogger(__name__)

class MonoLocoPredictor:
    def __init__(self, args):
        self.args = args
        self.dic_models = self.download_checkpoints()
        self.args.checkpoint = self.dic_models['keypoints']
        self.configure_devices()
        self.configure_visualization()

        self.predictor = Predictor(checkpoint=self.args.checkpoint)

    def get_torch_checkpoints_dir(self):
        if hasattr(torch, 'hub') and hasattr(torch.hub, 'get_dir'):
            base_dir = torch.hub.get_dir()
        elif os.getenv('TORCH_HOME'):
            base_dir = os.getenv('TORCH_HOME')
        elif os.getenv('XDG_CACHE_HOME'):
            base_dir = os.path.join(os.getenv('XDG_CACHE_HOME'), 'torch')
        else:
            base_dir = os.path.expanduser(os.path.join('~', '.cache', 'torch'))
        return os.path.join(base_dir, 'checkpoints')

    def download_checkpoints(self):
        torch_dir = self.get_torch_checkpoints_dir()
        os.makedirs(torch_dir, exist_ok=True)
        pifpaf_model = os.path.join(torch_dir, 'shufflenetv2k30-201104-224654-cocokp-d75ed641.pkl')
        dic_models = {'keypoints': pifpaf_model}

        if not os.path.exists(pifpaf_model):
            LOG.info('Downloading OpenPifPaf model in %s', torch_dir)
            # Download the model (you can use gdown or another method)
            # DOWNLOAD(OPENPIFPAF_MODEL, pifpaf_model, quiet=False)

        # Load stereo or mono models
        if self.args.mode == 'stereo':
            path = MONSTEREO_MODEL
            name = 'monstereo-201202-1212.pkl'
        else:
            path = MONOLOCO_MODEL_NU
            name = 'monoloco_pp-201207-1350.pkl'

        model = os.path.join(torch_dir, name)
        dic_models[self.args.mode] = model

        if not os.path.exists(model):
            LOG.info('Downloading model in %s', torch_dir)
            # DOWNLOAD(path, model, quiet=False)

        return dic_models


    def configure_devices(self):
        self.args.device = torch.device('cpu')
        self.args.pin_memory = False
        if torch.cuda.is_available():
            self.args.device = torch.device('cuda')
            self.args.pin_memory = True
        LOG.debug('neural network device: %s', self.args.device)

    def configure_visualization(self):
        if not self.args.output_types and self.args.mode != 'keypoints':
            self.args.output_types = ['multi']
        self.args.figure_width = 10
        self.args.dpi_factor = 1.0

    def factory_from_args(self):
        # Data
        if self.args.glob:
            self.args.images += glob.glob(self.args.glob)
        if not self.args.images:
            raise Exception("no image files given")

        if self.args.path_gt is None:
            self.args.show_all = True

        # Models
        dic_models = self.download_checkpoints()
        self.args.checkpoint = dic_models['keypoints']

        # Devices
        self.args.device = torch.device('cpu')
        self.args.pin_memory = False
        if torch.cuda.is_available():
            self.args.device = torch.device('cuda')
            self.args.pin_memory = True
        LOG.debug('neural network device: %s', self.args.device)

        # Add visualization defaults
        if not self.args.output_types and self.args.mode != 'keypoints':
            self.args.output_types = ['multi']
        self.args.figure_width = 10
        self.args.dpi_factor = 1.0

        if self.args.mode == 'stereo':
            self.args.batch_size = 2
            self.args.images = sorted(args.images)
        else:
            self.args.batch_size = 1

        # Patch for stereo images with batch_size = 2
        if self.args.batch_size == 2 and not self.args.long_edge:
            self.args.long_edge = 1238
            LOG.info("Long-edge set to %i", self.args.long_edge)

        # Make default pifpaf argument
        self.args.force_complete_pose = True
        LOG.info("Force complete pose is active")

        if self.args.mode != 'keypoints':
            assert any((xx in self.args.output_types for xx in ['front', 'bird', 'multi', 'json'])), \
                "No output type specified, please select one among front, bird, multi, json, or choose mode=keypoints"

        # Configure
        decoder.configure(self.args)
        network.Factory.configure(self.args)
        Predictor.configure(self.args)
        show.configure(self.args)
        visualizer.configure(self.args)

        return self.args, dic_models

    def predict(self):
        cnt = 0
        assert self.args.mode in ('keypoints', 'mono', 'stereo')
        args, dic_models = self.factory_from_args()
        # Load Models
        if args.mode in ('mono', 'stereo'):
            net = Loco(
                model=dic_models[args.mode],
                mode=args.mode,
                device=args.device,
                n_dropout=args.n_dropout,
                p_dropout=args.dropout)

        # for openpifpaf predictions
        predictor = Predictor(checkpoint=args.checkpoint)

        # data
        data = datasets.ImageList(args.images, preprocess=predictor.preprocess_factory)
        
        if args.mode == 'stereo':
            assert len(data.image_paths) % 2 == 0, "Odd number of images in a stereo setting"

        pifpaf_outs = {}
        start = time.time()
        timing = []
        for idx, (pred, _, meta) in enumerate(predictor.images(args.images, batch_size=args.batch_size)):
            
            if idx % args.batch_size != 0:  # Only for MonStereo
                pifpaf_outs['right'] = [ann.json_data() for ann in pred]
            else:
                if args.json_output is not None:
                    json_out_name = out_name(args.json_output, meta['file_name'], '.predictions.json')
                    LOG.debug('json output = %s', json_out_name)
                    with open(json_out_name, 'w') as f:
                        json.dump([ann.json_data() for ann in pred], f)

                pifpaf_outs['pred'] = pred
                pifpaf_outs['left'] = [ann.json_data() for ann in pred]
                pifpaf_outs['file_name'] = meta['file_name']
                pifpaf_outs['width_height'] = meta['width_height']
                # Set output image name
                if args.output_directory is None:
                    splits = os.path.split(meta['file_name'])
                    output_path = os.path.join(splits[0], 'out_' + splits[1])
                else:
                    file_name = os.path.basename(meta['file_name'])
                    output_path = os.path.join(
                        args.output_directory, 'out_' + file_name)

                im_name = os.path.basename(meta['file_name'])
                print(f'{idx} image {im_name} saved as {output_path}')

            if (args.mode == 'mono') or (args.mode == 'stereo' and idx % args.batch_size != 0):
                # 3D Predictions
                if args.mode == 'keypoints':
                    dic_out = defaultdict(list)
                    kk = None
                else:
                    im_size = (float(pifpaf_outs['width_height'][0]), float(pifpaf_outs['width_height'][1]))

                    if args.path_gt is not None:
                        dic_gt, kk = factory_for_gt(args.path_gt, im_name)
                    else:
                        kk = load_calibration(args.calibration, im_size, focal_length=args.focal_length)
                        dic_gt = None
                    # Preprocess pifpaf outputs and run monoloco
                    boxes, keypoints = preprocess_pifpaf(##############################################
                        pifpaf_outs['left'], im_size, enlarge_boxes=False)#############################

                    if args.mode == 'mono':
                        LOG.info("Prediction with MonoLoco++")
                        dic_out = net.forward(keypoints, kk)
                        fwd_time = (time.time()-start)*1000
                        timing.append(fwd_time)  # Skip Reordering and saving images
                        print(f"Forward time: {fwd_time:.0f} ms")
                        dic_out = net.post_process(
                            dic_out, boxes, keypoints, kk, dic_gt)
                        if 'social_distance' in args.activities:
                            dic_out = net.social_distance(dic_out, args)
                        if 'raise_hand' in args.activities:
                            dic_out = net.raising_hand(dic_out, keypoints)

                    else:
                        LOG.info("Prediction with MonStereo")
                        _, keypoints_r = preprocess_pifpaf(pifpaf_outs['right'], im_size)
                        dic_out = net.forward(keypoints, kk, keypoints_r=keypoints_r)
                        fwd_time = (time.time()-start)*1000
                        timing.append(fwd_time)
                        dic_out = net.post_process(
                            dic_out, boxes, keypoints, kk, dic_gt)

                # Output
                factory_outputs(args, pifpaf_outs, dic_out, output_path, kk=kk)
                print(f'Image {cnt}\n' + '-' * 120)
                cnt += 1
                start = time.time()
        timing = np.array(timing)
        avg_time = int(np.mean(timing))
        std_time = int(np.std(timing))
        print(f'Processed {idx * args.batch_size} images with an average time of {avg_time} ms and a std of {std_time} ms')
        return dic_out

    def run_monoloco(self, pifpaf_outs):
        # Process predictions using MonoLoco
        im_size = (float(pifpaf_outs['width_height'][0]), float(pifpaf_outs['width_height'][1]))
        boxes, keypoints = preprocess_pifpaf(pifpaf_outs['left'], im_size, enlarge_boxes=False)

        # Run the MonoLoco model
        net = Loco(
            model=self.dic_models[self.args.mode],
            mode=self.args.mode,
            device=self.args.device,
            n_dropout=self.args.n_dropout,
            p_dropout=self.args.dropout)
        dic_out = net.forward(keypoints, None)
        return dic_out

    def save_results(self, dic_out, pifpaf_outs, output_path):
        with open(pifpaf_outs['file_name'], 'rb') as f:
            cpu_image = PIL.Image.open(f).convert('RGB')
        annotation_painter = openpifpaf.show.AnnotationPainter()

        with openpifpaf.show.image_canvas(cpu_image, output_path) as ax:
            annotation_painter.annotations(ax, pifpaf_outs['pred'])

    def get_output_path(self, file_name):
        if self.args.output_directory is None:
            splits = os.path.split(file_name)
            return os.path.join(splits[0], 'out_' + splits[1])
        else:
            return os.path.join(self.args.output_directory, 'out_' + os.path.basename(file_name))
