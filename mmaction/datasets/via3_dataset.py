import copy
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime
import glob
import mmcv
import numpy as np
from mmcv.utils import print_log
from tqdm import tqdm
from ..core.evaluation.via3_utils import via3_eval, read_labelmap, results2csv
from ..utils import get_root_logger
from .base import BaseDataset
from .registry import DATASETS
import json
from .via3_tool import Via3Json
from colorama import Fore

@DATASETS.register_module()
class VIA3Dataset(BaseDataset):
    """AVA dataset for spatial temporal detection.

    Based on official AVA annotation files, the dataset loads raw frames,
    bounding boxes, proposals and applies specified transformations to return
    a dict containing the frame tensors and other information.

    This datasets can load information from the following files:

    .. code-block:: txt

        ann_file -> ava_{train, val}_{v2.1, v2.2}.csv
        exclude_file -> ava_{train, val}_excluded_timestamps_{v2.1, v2.2}.csv
        label_file -> ava_action_list_{v2.1, v2.2}.pbtxt /
                      ava_action_list_{v2.1, v2.2}_for_activitynet_2019.pbtxt
        proposal_file -> ava_dense_proposals_{train, val}.FAIR.recall_93.9.pkl

    Particularly, the proposal_file is a pickle file which contains
    ``img_key`` (in format of ``{video_id},{timestamp}``). Example of a pickle
    file:

    .. code-block:: JSON

        {
            ...
            '0f39OWEqJ24,0902':
                array([[0.011   , 0.157   , 0.655   , 0.983   , 0.998163]]),
            '0f39OWEqJ24,0912':
                array([[0.054   , 0.088   , 0.91    , 0.998   , 0.068273],
                       [0.016   , 0.161   , 0.519   , 0.974   , 0.984025],
                       [0.493   , 0.283   , 0.981   , 0.984   , 0.983621]]),
            ...
        }

    Args:
        ann_file (str): Path to the annotation file like
            ``ava_{train, val}_{v2.1, v2.2}.csv``.
        exclude_file (str): Path to the excluded timestamp file like
            ``ava_{train, val}_excluded_timestamps_{v2.1, v2.2}.csv``.
        pipeline (list[dict | callable]): A sequence of data transforms.
        label_file (str): Path to the label file like
            ``ava_action_list_{v2.1, v2.2}.pbtxt`` or
            ``ava_action_list_{v2.1, v2.2}_for_activitynet_2019.pbtxt``.
            Default: None.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        proposal_file (str): Path to the proposal file like
            ``ava_dense_proposals_{train, val}.FAIR.recall_93.9.pkl``.
            Default: None.
        person_det_score_thr (float): The threshold of person detection scores,
            bboxes with scores above the threshold will be used. Default: 0.9.
            Note that 0 <= person_det_score_thr <= 1. If no proposal has
            detection score larger than the threshold, the one with the largest
            detection score will be used.
        num_classes (int): The number of classes of the dataset. Default: 81.
            (AVA has 80 action classes, another 1-dim is added for potential
            usage)
        custom_classes (list[int]): A subset of class ids from origin dataset.
            Please note that 0 should NOT be selected, and ``num_classes``
            should be equal to ``len(custom_classes) + 1``
        data_prefix (str): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
                        Default: 'RGB'.
        num_max_proposals (int): Max proposals number to store. Default: 1000.
        timestamp_start (int): The start point of included timestamps. The
            default value is referred from the official website. Default: 902.
        timestamp_end (int): The end point of included timestamps. The
            default value is referred from the official website. Default: 1798.
    """

    _FPS = 30

    def __init__(self,
                 ann_file,
                 proposal_file,
                 pipeline,
                 filename_tmpl='_{:05}.jpg',
                 attribute='person',
                 custom_classes=None,
                 person_det_score_thr=0.9,
                 data_prefix=None,
                 test_mode=False,
                 modality='RGB',
                 num_max_proposals=1000,
                 timestamp_start = 0,
                 timestamp_end = 1800,):
        # since it inherits from `BaseDataset`, some arguments
        # should be assigned before performing `load_annotations()`
        self.attribute = attribute


        self.proposals = {}
        self.via3_proposal = Via3Json(proposal_file)
        self.att_ids_proposal = self.via3_proposal.loadIdsFromAttsname(self.attribute)


        self.via3_gt = Via3Json(ann_file)
        self.att_ids_gt = self.via3_gt.loadIdsFromAttsname(self.attribute)
        assert len(self.att_ids_gt) > 0
        self.att_ids2att_labels = {att_id : i for i, att_id in enumerate(self.att_ids_gt)}
        self.att_ids2att_infos = {att_id : self.via3_gt.loadAttFromId(att_id) for  att_id in self.att_ids_gt}
        self.fids_gt = self.via3_gt.loadFilesFid()
        self.filesinfo_gt = self.via3_gt.loadFilesInfoFromAll()

        self.opt_ids2opt_names = self.via3_gt.attributes[self.att_ids_gt[0]]['options']
        self.att_ids2att_infos = self.via3_gt.loadAttsFromAll()

        self.all_classes = [self.opt_ids2opt_names[pt_id] for pt_id in self.opt_ids2opt_names]
        self.custom_classes = [custom_classes, ] if isinstance(custom_classes, str) else custom_classes
        if custom_classes is None:
            self.classes = self.all_classes
        else:
            assert isinstance(custom_classes, (list, tuple))
            assert set(self.custom_classes).issubset(self.all_classes)
            self.classes = self.custom_classes

        self.opt_ids = self.via3_gt.loadOptidsFromAtt(self.att_ids2att_infos[self.att_ids_gt[0]], self.classes)

        self.opt_ids2opt_labels = {opt_id : label for  label, opt_id in enumerate(self.opt_ids)}
        self.opt_labels2opt_names = {self.opt_ids2opt_labels[opt_id] :
                                     self.opt_ids2opt_names[opt_id]
                                     for  opt_id in self.opt_ids}

        arrs_opts = {self.att_ids_gt[0]:self.opt_ids}
        self.metadatasinfo_gt = self.via3_gt.loadMetadatasInfoFromAll(arrs_opts)
        self.metadatasinfo_proposal = self.via3_proposal.loadMetadatasInfoFromAll(arrs_opts)

        assert 0 <= person_det_score_thr <= 1, (
            'The value of '
            'person_det_score_thr should in [0, 1]. ')


        self.person_det_score_thr = person_det_score_thr
        self.filename_tmpl = filename_tmpl
        self.num_max_proposals = num_max_proposals
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.logger = get_root_logger()
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            modality=modality,
            num_classes=len(self.opt_ids2opt_labels))

    def parse_metadatainfo(self, metadatainfo):
        att_id = self.att_ids_gt[0]
        bboxes, labels = [], []
        #while len(img_records) > 0:
        for  a_metadatainfo in  metadatainfo:
            vid, flg= a_metadatainfo['vid'], a_metadatainfo['flg']
            z, xy, av = a_metadatainfo['z'], a_metadatainfo['xy'], a_metadatainfo['av']
            if (av.get(att_id, None)==None) or av[att_id]=='':
                continue
            opt_id_list = av[att_id].split(',')
            valid_labels = np.array([self.opt_ids2opt_labels[opt_id] for opt_id in opt_id_list])
            #print(valid_labels)
            label = np.zeros(self.num_classes, dtype=np.float32)
            label[valid_labels] = 1
            labels.append(label)
            bboxes.append(list(map(float, xy[1:])))

        if (bboxes==[]) or  (labels==[]):
            bboxes  = np.zeros((0,4))
            labels = [0,self.num_classes]
        else:
            bboxes = np.array(bboxes)
            labels = np.array(labels)
        return bboxes, labels

    def load_annotations(self):
        video_infos = []
        # self.fids = self.via3_gt.loadFilesFid()
        # self.filesinfo = self.via3_gt.loadFilesInfoFromAll()
        # self.metadatasinfo = self.via3_gt.loadMetadatasInfoFromAll()
        timestamps_dict = defaultdict(list)
        print('loading data : {}'.format(self.ann_file))
        for fid in tqdm(self.fids_gt, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET)):
            metadatainfo_proposal = self.metadatasinfo_proposal[fid]
            bboxes_xywh_proposal, labels_proposal = self.parse_metadatainfo(metadatainfo_proposal)
            bboxes_xyxy_proposal = bboxes_xywh_proposal.copy()
            bboxes_xyxy_proposal[:, [2, 3]] = bboxes_xywh_proposal[:, [0, 1]] + bboxes_xywh_proposal[:, [2, 3]]
            bboxes_proposal = bboxes_xyxy_proposal

            metadatainfo_gt = self.metadatasinfo_gt[fid]
            bboxes_xywh_gt, labels_gt = self.parse_metadatainfo(metadatainfo_gt)
            bboxes_xyxy_gt = bboxes_xywh_gt.copy()
            bboxes_xyxy_gt[:, [2, 3]] = bboxes_xyxy_gt[:, [0, 1]] + bboxes_xyxy_gt[:, [2, 3]]
            bboxes_gt = bboxes_xyxy_gt

            if bboxes_gt.size <= 0:
                continue

            fileinfo = self.filesinfo_gt[fid]
            img_name = fileinfo['fname']
            img_name_prefix , extension = osp.splitext(img_name)
            video_id, timestamp = img_name_prefix.split('_')
            frame_dir = video_id
            timestamp = int(timestamp)
            timestamps_dict[video_id].append(timestamp)
            if len(timestamps_dict[video_id]) >= 2:
                assert timestamps_dict[video_id][-1] > timestamps_dict[video_id][-2]

            if self.data_prefix is not None:
                frame_dir = osp.join(self.data_prefix, frame_dir)
            img_key = '{},{:05}'.format(video_id, timestamp)

            filepath = osp.join(frame_dir, video_id + self.filename_tmpl.format(int(timestamp)))
            h, w = mmcv.imread(filepath).shape[:2]
            scale_factor = np.array([w,h,w,h])

            self.proposals[img_key] = bboxes_proposal/scale_factor
            ann = dict(gt_bboxes=bboxes_gt/scale_factor, gt_labels=labels_gt)

            video_info = dict(
                frame_dir=frame_dir,
                video_id=video_id,
                timestamp=int(timestamp),
                img_key=img_key,
                fps=self._FPS,
                ann=ann)
            video_infos.append(video_info)
        for video_info in video_infos:
            timestamps_list = timestamps_dict[video_info['video_id']]
            video_info.update(timestamps_list=timestamps_list)
        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        img_key = results['img_key']
        results['filename_tmpl'] = results['video_id'] + self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = 0
        results['timestamp_start'] = self.timestamp_start
        results['timestamp_end'] = results['timestamps_list'][-1]

        assert self.proposals
        if img_key not in self.proposals:
            results['proposals'] = np.array([[0, 0, 1, 1]])
            results['scores'] = np.array([1])
        else:
            proposals = self.proposals[img_key]
            assert proposals.shape[-1] in [4, 5]
            if proposals.shape[-1] == 5:
                thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                positive_inds = (proposals[:, 4] >= thr)
                proposals = proposals[positive_inds]
                proposals = proposals[:self.num_max_proposals]
                results['proposals'] = proposals[:, :4]
                results['scores'] = proposals[:, 4]
            else:
                proposals = proposals[:self.num_max_proposals]
                results['proposals'] = proposals


        ann = results.pop('ann')
        results['gt_bboxes'] = ann['gt_bboxes']
        results['gt_labels'] = ann['gt_labels']
        #results['entity_ids'] = ann['entity_ids']
        results = self.pipeline(results)
        return results

    def prepare_test_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        img_key = results['img_key']
        results['filename_tmpl'] = results['video_id'] + self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = 0
        results['timestamp_start'] = self.timestamp_start
        results['timestamp_end'] = results['timestamps_list'][-1]

        if self.proposals is not None:
            if img_key not in self.proposals:
                results['proposals'] = np.array([[0, 0, 1, 1]])
                results['scores'] = np.array([1])
            else:
                proposals = self.proposals[img_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = (proposals[:, 4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals[:, :4]
                    results['scores'] = proposals[:, 4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals

        ann = results.pop('ann')
        results['gt_bboxes'] = ann['gt_bboxes']
        results['gt_labels'] = ann['gt_labels']
        #results['entity_ids'] = ann['entity_ids']
        return self.pipeline(results)

    def dump_results(self, results, out):
        assert out.endswith('csv')
        results2csv(self, results, out, self.custom_classes)

    def evaluate(self,
                 results,
                 metrics=('mAP', ),
                 metric_options=None,
                 logger=None):

        # need to create a temp result file
        assert len(metrics) == 1 and metrics[0] == 'mAP', (
            'For evaluation on AVADataset, you need to use metrics "mAP" '
            'See https://github.com/open-mmlab/mmaction2/pull/567 '
            'for more info.')
        #time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        #temp_file = f'VIA3.0_{time_now}_result.csv'
        #results2csv(self, results, temp_file, self.custom_classes)
        ret = {}
        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            eval_result = via3_eval(results,self.video_infos,
                                    self.opt_ids,
                                    self.opt_ids2opt_labels,
                                    self.opt_ids2opt_names,metric)

            log_msg = []
            for k, v in eval_result.items():
                log_msg.append(f'\n{k}\t{v: .4f}')
            log_msg = ''.join(log_msg)
            print_log(log_msg, logger=logger)
            ret.update(eval_result)
        return ret

