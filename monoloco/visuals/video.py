import os
import time
import logging
import numpy as np

import torch
import matplotlib.pyplot as plt
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None

import openpifpaf
from openpifpaf import decoder, network, visualizer, show, logger
from openpifpaf import datasets

from ..visuals import Printer
from ..network import Loco, preprocess_pifpaf, load_calibration
from ..predict import download_checkpoints

LOG = logging.getLogger(__name__)

def factory_from_args(args):

    # Model
    dic_models = download_checkpoints(args)
    args.checkpoint = dic_models['keypoints']

    logger.configure(args, LOG)  # logger first

    assert len(args.output_types) == 1 and 'json' not in args.output_types

    # Devices
    args.device = torch.device('cpu')
    args.pin_memory = False
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    # Add visualization defaults
    if not args.output_types:
        args.output_types = ['multi']

    args.figure_width = 10
    args.dpi_factor = 1.0

    args.z_max = 10
    args.show_all = True
    args.no_save = True
    args.batch_size = 1

    if args.long_edge is None:
        args.long_edge = 144
    # Make default pifpaf argument
    args.force_complete_pose = True
    LOG.info("Force complete pose is active")

    # Configure
    decoder.configure(args)
    network.Factory.configure(args)
    show.configure(args)
    visualizer.configure(args)

    return args, dic_models

def process_video(args):
    assert args.mode in 'mono'
    assert cv2

    args, dic_models = factory_from_args(args)

    # Load Models
    net = Loco(model=dic_models[args.mode], mode=args.mode, device=args.device,
               n_dropout=args.n_dropout, p_dropout=args.dropout)

    # for openpifpaf predictions
    predictor = openpifpaf.Predictor(checkpoint=args.checkpoint)

    # Open video file
    video_input = cv2.VideoCapture(args.video_path_in)
    if not video_input.isOpened():
        LOG.error("Error opening video file: %s", args.video_path_in)
        return

    # Get video properties
    frame_width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_input.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object for output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.video_path_out, fourcc, fps, (frame_width, frame_height))

    visualizer_mono = None
    frame_count = 0
    total_processing_time = 0
    while True:
        start = time.time()
        ret, frame = video_input.read()
        if not ret:
            LOG.info("End of video stream")
            break

        # Resize frame
        scale = (args.long_edge) / frame.shape[0]
        image = cv2.resize(frame, None, fx=scale, fy=scale)
        height, width, _ = image.shape
        LOG.debug('resized image size: {}'.format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        # Preprocessing and DataLoader setup
        data = datasets.PilImageList([pil_image], preprocess=predictor.preprocess_factory())
        data_loader = torch.utils.data.DataLoader(
            data, batch_size=1, shuffle=False, pin_memory=False, collate_fn=datasets.collate_images_anns_meta
        )

        # Process each frame with the model
        for (_, _, _) in data_loader:
            for idx, (preds, _, _) in enumerate(predictor.dataset(data)):
                if idx == 0:
                    pifpaf_outs = {
                        'pred': preds,
                        'left': [ann.json_data() for ann in preds],
                        'image': image
                    }

        # Calibration and Prediction
        kk = load_calibration(args.calibration, pil_image.size, focal_length=args.focal_length)
        boxes, keypoints = preprocess_pifpaf(pifpaf_outs['left'], (width, height))
        dic_out = net.forward(keypoints, kk)
        dic_out = net.post_process(dic_out, boxes, keypoints, kk)

        if 'social_distance' in args.activities:
            dic_out = net.social_distance(dic_out, args)
        if 'raise_hand' in args.activities:
            dic_out = net.raising_hand(dic_out, keypoints)
        if visualizer_mono is None:
            visualizer_mono = Visualizer(kk, args)

        # Get the processed image from the visualizer
        processed_image = visualizer_mono(pil_image, dic_out, pifpaf_outs)
        processed_image = np.array(processed_image)
        processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        processed_image_bgr_resized = cv2.resize(processed_image_bgr, (frame_width, frame_height))

        # Calculate FPS and display it
        end = time.time()
        frame_processing_time = end - start
        total_processing_time += frame_processing_time
        fps_display = 1 / frame_processing_time

        # Overlay FPS on the video
        cv2.putText(processed_image_bgr_resized, f"FPS: {fps_display:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        out.write(processed_image_bgr_resized)
        frame_count += 1

        LOG.info("run-time: {:.2f} ms".format((end-start)*1000))

    # Release video input and output
    video_input.release()
    out.release()

    cv2.destroyAllWindows()

    if frame_count > 0:
        avg_fps = frame_count / total_processing_time
        LOG.info(f"Average FPS: {avg_fps:.2f}")
        return avg_fps
    else:
        return 0


class Visualizer:
    def __init__(self, kk, args):
        self.kk = kk
        self.args = args

    def __call__(self, pil_image, dic_out, pifpaf_outs):
        # Convert PIL image to numpy array
        image = np.array(pil_image)

        # Process results and draw on the image
        printer = Printer(pil_image, output_path="output_directory/updating", kk=self.kk, args=self.args)
        figures, axes = printer.factory_axes(None)
        
        # Clear previous annotations
        for patch in axes[0].patches:
            patch.remove()
        for line in axes[0].lines:
            line.remove()
        for text in axes[0].texts:
            text.remove()

        if len(axes) > 1:
            for patch in axes[1].patches:
                patch.remove()
            for line in axes[1].lines:
                line.remove()
            for text in axes[1].texts:
                text.remove()

        # Process and draw new results
        printer._process_results(dic_out)
        figures = printer.draw(figures, axes, image, dic_out, pifpaf_outs['left'])
        
        fig = figures[0]
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return img_array


def mypause(interval):
    manager = plt._pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()
        canvas.start_event_loop(interval)
    else:
        time.sleep(interval)
