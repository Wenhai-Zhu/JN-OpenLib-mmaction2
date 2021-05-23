import argparse
import time
from collections import deque
from operator import itemgetter
from threading import Thread
import mmcv
import cv2
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
from mmaction.utils import import_module_error_func
from mmaction.models import build_detector
import random
import seaborn as sns
try:
    from mmdet.apis import inference_detector, init_detector, show_result_pyplot
except (ImportError, ModuleNotFoundError):
    @import_module_error_func('mmdet')
    def inference_detector(*args, **kwargs):
        pass

    @import_module_error_func('mmdet')
    def init_detector(*args, **kwargs):
        pass

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode', 'FrameSelector'
]

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 webcam demo')

    parser.add_argument(
        '--config',
        default=('configs/detection/ava/'
                 'slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb.py'),
        help='spatio temporal detection config file path')
    parser.add_argument(
        '--checkpoint',
        default=('../../Checkpoints/mmdetection/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15'),
        help='spatio temporal detection checkpoint file/url')

    parser.add_argument(
        '--det-config',
        default='../../Checkpoints/mmdetection/my_gfl_r50_fpn_mstrain_2x_person_gn.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('../../Checkpoints/mmdetection/my_gfl_r50_fpn_mstrain_2x_person_gn.pth'),
        help='human detection checkpoint file/url')

    parser.add_argument(
        '--label-map', default='./mywork/label_map_ava.txt', help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id',  default=0, help='camera device id')
    parser.add_argument(
        '--img-scale', type=float, default=0.25, help='out img reszie scale')

    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.5,
        help='the threshold of human detection score')
    parser.add_argument(
        '--act-score-thr',
        type=float,
        default=0.5,
        help='the threshold of human action score')

    parser.add_argument(
        '--drawing-stepsize',
        type=int,
        default=3,
        help='drawing  give out a drawing per n frames')
    args = parser.parse_args()
    assert args.drawing_stepsize >= 0 , \
        'upper bound FPS value of drawing and inference should be set as ' \
        'positive number, or zero for no limit'
    return args


def load_label_map(file_path):
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}

class DetModel:
    def __init__(self, config, checkpoint, score_thr=0.5, device='cuda:0'):
        self.score_thr = score_thr
        self.device = device
        self.model = init_detector(config, checkpoint ,self.device)
        assert self.model.CLASSES[0] == 'person', ('We require you to use a detector '
                                                  'trained on COCO')
    def __call__(self, imgs):
        result = inference_detector(self.model , imgs)
        bboxes, labels = restore_result(result)
        thr_ind = bboxes[:,4]>=self.score_thr
        bboxes, labels = bboxes[thr_ind], labels[thr_ind]
        class_ind = labels==0
        bboxes, labels = bboxes[class_ind], labels[class_ind]
        return bboxes, labels

class ActModel():
    def __init__(self, config_path, checkpoint, score_thr=0.5, label_dict=None, device='cuda:0'):
        self.score_thr = score_thr
        self.label_dict =label_dict
        self.device = device
        config = mmcv.Config.fromfile(config_path)
        self.img_norm_cfg = config['img_norm_cfg']
        if 'to_rgb' not in self.img_norm_cfg and 'to_bgr' in self.img_norm_cfg:
            to_bgr = self.img_norm_cfg.pop('to_bgr')
            self.img_norm_cfg['to_rgb'] = to_bgr
        self.img_norm_cfg['mean'] = np.array(self.img_norm_cfg['mean'])
        self.img_norm_cfg['std'] = np.array(self.img_norm_cfg['std'])
        # Get clip_len, frame_interval and calculate center index of each clip
        val_pipeline = config['val_pipeline']
        sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]

        self.clip_len, self.frame_interval = sampler['clip_len'], sampler['frame_interval']
        self.window_size = self.clip_len * self.frame_interval
        assert self.clip_len % 2 == 0, 'We would like to have an even clip_len'

        config.model.backbone.pretrained = None
        self.model = build_detector(config.model, test_cfg=config.get('test_cfg'))
        load_checkpoint(self.model, checkpoint, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()


    def post_proce(self, result, proposal):
        result = result[0]
        act_pred = []
        # N proposals
        for i in range(proposal.shape[0]):
            act_pred.append([])
        # Perform action score thr
        for i in range(len(result)):
            if i + 1 not in self.label_dict:
                continue
            for j in range(proposal.shape[0]):
                if result[i][j, 4] > self.score_thr:
                    act_pred[j].append((self.label_dict[i + 1], result[i][j, 4]))
        return act_pred

    def __call__(self, frames, proposals):
        frame_w, frame_h = frames[0].shape[1], frames[0].shape[0]
        new_w, new_h = mmcv.rescale_size((frame_w, frame_h), (256, np.Inf))
        w_ratio, h_ratio = new_w / frame_w, new_h / frame_h
        frames = [mmcv.imresize(img, (new_w, new_h)) for img in frames]
        _ = [mmcv.imnormalize_(frame, **self.img_norm_cfg) for frame in frames]
        # THWC -> CTHW -> 1CTHW
        input_array = np.stack(frames).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(self.device)

        proposal = proposals[len(proposals) // 2]
        proposal = torch.from_numpy(proposal[:, :4]).to(self.device)
        if proposal.shape[0] == 0:
            return None

        proposal[:, 0:4:2] *= w_ratio
        proposal[:, 1:4:2] *= h_ratio
        with torch.no_grad():
            result = self.model(
                return_loss=False,
                img=[input_tensor],
                img_metas=[[dict(img_shape=(new_h, new_w))]],
                proposals=[[proposal]])
        return self.post_proce(result, proposal)

def random_color(seed):
    """Random a color according to the input seed."""
    random.seed(seed)
    colors = sns.color_palette()
    color = random.choice(colors)
    return color

def visualize_bbox_act(img, bboxes,labels, act_preds,
              classes=None,thickness=1,
              font_scale=0.4,show=False,
              wait_time=0,out_file=None):
    """Show the tracks with opencv."""
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 5
    if isinstance(img, str):
        img = mmcv.imread(img)

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    text_width, text_height = 8, 15
    for i, (bbox, label) in enumerate(zip(bboxes, labels), 0):
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        score = float(bbox[-1])

        # bbox
        bbox_color = random_color(label)
        bbox_color = [int(255 * _c) for _c in bbox_color][::-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        # score
        text = '{:.02f}'.format(score)
        width = len(text) * text_width
        img[y1 - text_height:y1, x1:x1 + width, :] = bbox_color
        cv2.putText(
            img,
            text, (x1, y1 - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=(0, 0, 0))


        classes_color = random_color(label + 1)
        text = classes[label]
        width = len(text) * text_width
        img[y1:y1 + text_height, x1:x1 + width, :] = bbox_color
        cv2.putText(img,text,
                    (x1, y1 + text_height - 2),
                    cv2.FONT_HERSHEY_COMPLEX,
                    font_scale,color=classes_color)

        #background_color = random_color(label + 5)
        background_color = [255, 204, 153]
        if (act_preds is not None) and (len(bboxes)==len(labels)==len(act_preds)):
            for j, act_pred in enumerate(act_preds[i]):
                text = '{}: {:.02f}'.format(act_pred[0], act_pred[1])
                width = len(text) * (text_width)
                img[y1+text_height*(j+2) :y1 + text_height*(j+3), x1:x1 + width, :] = background_color
                cv2.putText(img, text,
                            (x1, y1 + text_height*(j+3) - 2),
                            cv2.FONT_HERSHEY_COMPLEX,
                            font_scale, color=classes_color)

    if show:
        mmcv.imshow(img, wait_time=wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img

def restore_result(result, return_ids=False):
    """Restore the results (list of results of each category) into the results
    of the model forward.

    Args:
        result (list[ndarray]): shape (n, 5) or (n, 6)
        return_ids (bool, optional): Whether the input has tracking
            result. Default to False.

    Returns:
        tuple: tracking results of each class.
    """
    labels = []
    for i, bbox in enumerate(result):
        labels.extend([i] * bbox.shape[0])
    bboxes = np.concatenate(result, axis=0).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    if return_ids:
        ids = bboxes[:, 0].astype(np.int64)
        bboxes = bboxes[:, 1:]
        return bboxes, labels, ids
    else:
        return bboxes, labels

def det_inference(args):
    global start_frame_ind, frame_queue_length
    total_frame_id = 0
    activate_frame_id = 0
    is_already_act_show = True
    act_fid = None
    act_preds =None
    border_length = act_model.window_size//2
    while True:
        ret, frame = camera.read()
        total_frame_id += 1
        if (not ret) or (total_frame_id%args.drawing_stepsize!=0) :
            continue
        activate_frame_id +=1
        shape = (int(frame.shape[1] * args.img_scale), int(frame.shape[0] * args.img_scale))
        img = cv2.resize(frame, shape)
        bboxes, labels = det_model(img)
        frame_queue.append([activate_frame_id,np.array(img),bboxes, labels])
        show_frame_id, old_frame, bboxes, labels= frame_queue[0]
        if is_already_act_show and len(result_queue)!=0:
            act_fid, new_preds = result_queue.popleft()
            is_already_act_show = False

        visualize_bbox_act(old_frame, bboxes, labels, act_preds,
                  classes =['person'],show=True,wait_time=1)

        if (act_fid is not None):
            #print(start_frame_ind, show_frame_id - act_fid, len(result_queue))
            if  abs(show_frame_id - act_fid) > border_length:
                start_frame_ind += (show_frame_id - act_fid)//5
                start_frame_ind = max(min(start_frame_ind, frame_queue_length-border_length), border_length)
                #print(frame_queue_length-border_length, start_frame_ind, border_length)
            if  (show_frame_id >= act_fid) :
                act_preds = new_preds
                is_already_act_show = True


def act_inference(args):
    while len(frame_queue) == 0:
        time.sleep(0.2)
    while True:
        frame_inds = start_frame_ind + np.arange(0, act_model.window_size, act_model.frame_interval)
        if (len(result_queue) <=3) and (len(frame_queue) > frame_inds[-1]):
            cur_windows_fids, cur_windows_imgs, = [], []
            cur_windows_dets, cur_windows_labels = [], []
            for ind in frame_inds:
                cur_windows_fids.append(frame_queue[ind][0])
                act_img = frame_queue[ind][1].astype(np.float32)
                cur_windows_imgs.append(act_img)
                det = frame_queue[ind][2].astype(np.float32)
                cur_windows_dets.append(det)
                cur_windows_labels.append(frame_queue[ind][3])

            fid = cur_windows_fids[len(cur_windows_fids) // 2]
            act_pred = act_model(cur_windows_imgs, cur_windows_dets)
            result_queue.append([fid, act_pred])
        else:
            time.sleep(0.005)


def main():
    global frame_queue, frame_queue_length, result_queue
    global    det_model
    global    act_model, window_size, start_frame_ind
    global camera, camera_fps, frame_w, frame_h

    args = parse_args()

    camera = cv2.VideoCapture(args.camera_id)
    #camera = cv2.VideoCapture('rtsp://admin:XFchipeak@192.168.1.48:554')
    #camera = cv2.VideoCapture('demo/ava_demo.mp4')
    camera_fps = camera.get(cv2.CAP_PROP_FPS)  # 帧率

    det_model = DetModel(args.det_config, args.det_checkpoint, score_thr=0.5, device=args.device)
    label_dict = load_label_map(args.label_map)
    act_model = ActModel(args.config, args.checkpoint, score_thr=0.5, label_dict=label_dict, device=args.device)

    start_frame_ind = act_model.window_size
    frame_queue_length = act_model.window_size*3
    assert act_model.window_size > 0

    try:
        frame_queue = deque(maxlen=frame_queue_length)
        result_queue = deque(maxlen=5)
        pw = Thread(target=det_inference, args=(args, ), daemon=True)
        pr = Thread(target=act_inference, args=(args,  ), daemon=True)
        pw.start()
        pr.start()
        pw.join()
        pr.join()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
