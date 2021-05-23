import csv
import heapq
import logging
import time
from collections import defaultdict

import numpy as np

from .ava_evaluation import object_detection_evaluation as det_eval
from .ava_evaluation import standard_fields
from .recall import eval_recalls


def det2csv(video_infos, results):
    csv_results = []
    for idx in range(len(video_infos)):
        video_id = video_infos[idx]['video_id']
        timestamp = video_infos[idx]['timestamp']
        result = results[idx]
        for label, _ in enumerate(result):
            for bbox in result[label]:
                bbox_ = tuple(bbox.tolist())
                csv_results.append((
                    video_id,
                    timestamp,
                ) + bbox_[:4] + (label, ) + bbox_[4:])
    return csv_results


# results is organized by class
def results2csv(dataset, results, out_file, custom_classes=None):
    if isinstance(results[0], list):
        csv_results = det2csv(dataset, results, custom_classes)

    # save space for float
    def tostr(item):
        if isinstance(item, float):
            return f'{item:.3f}'
        return str(item)

    with open(out_file, 'w') as f:
        for csv_result in csv_results:
            f.write(','.join(map(lambda x: tostr(x), csv_result)))
            f.write('\n')


def print_time(message, start):
    print('==> %g seconds to %s' % (time.time() - start, message))


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return f'{video_id},{int(timestamp):04d}'


def read_csv(csv_file, class_whitelist=None, capacity=0):
    """Loads boxes and class labels from a CSV file in the AVA format.

    CSV file format described at https://research.google.com/ava/download.html.

    Args:
        csv_file: A file object.
        class_whitelist: If provided, boxes corresponding to (integer) class
        labels not in this set are skipped.
        capacity: Maximum number of labeled boxes allowed for each example.
        Default is 0 where there is no limit.

    Returns:
        boxes: A dictionary mapping each unique image key (string) to a list of
        boxes, given as coordinates [y1, x1, y2, x2].
        labels: A dictionary mapping each unique image key (string) to a list
        of integer class lables, matching the corresponding box in `boxes`.
        scores: A dictionary mapping each unique image key (string) to a list
        of score values lables, matching the corresponding label in `labels`.
        If scores are not provided in the csv, then they will default to 1.0.
    """
    start = time.time()
    entries = defaultdict(list)
    boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)
    reader = csv.reader(csv_file)
    for row in reader:
        assert len(row) in [7, 8], 'Wrong number of columns: ' + row
        image_key = make_image_key(row[0], row[1])
        x1, y1, x2, y2 = [float(n) for n in row[2:6]]
        action_id = int(row[6])
        if class_whitelist and action_id not in class_whitelist:
            continue

        score = 1.0
        if len(row) == 8:
            score = float(row[7])
        if capacity < 1 or len(entries[image_key]) < capacity:
            heapq.heappush(entries[image_key],
                           (score, action_id, y1, x1, y2, x2))
        elif score > entries[image_key][0][0]:
            heapq.heapreplace(entries[image_key],
                              (score, action_id, y1, x1, y2, x2))
    for image_key in entries:
        # Evaluation API assumes boxes with descending scores
        entry = sorted(entries[image_key], key=lambda tup: -tup[0])
        for item in entry:
            score, action_id, y1, x1, y2, x2 = item
            boxes[image_key].append([y1, x1, y2, x2])
            labels[image_key].append(action_id)
            scores[image_key].append(score)
    print_time('read file ' + csv_file.name, start)
    return boxes, labels, scores


def read_exclusions(exclusions_file):
    """Reads a CSV file of excluded timestamps.

    Args:
        exclusions_file: A file object containing a csv of video-id,timestamp.

    Returns:
        A set of strings containing excluded image keys, e.g.
        "aaaaaaaaaaa,0904",
        or an empty set if exclusions file is None.
    """
    excluded = set()
    if exclusions_file:
        reader = csv.reader(exclusions_file)
    for row in reader:
        assert len(row) == 2, 'Expected only 2 columns, got: ' + row
        excluded.add(make_image_key(row[0], row[1]))
    return excluded


def read_labelmap(labelmap_file):
    """Reads a labelmap without the dependency on protocol buffers.

    Args:
        labelmap_file: A file object containing a label map protocol buffer.

    Returns:
        labelmap: The label map in the form used by the
        object_detection_evaluation
        module - a list of {"id": integer, "name": classname } dicts.
        class_ids: A set containing all of the valid class id integers.
    """
    labelmap = []
    class_ids = set()
    name = ''
    class_id = ''
    for line in labelmap_file:
        if line.startswith('  name:'):
            name = line.split('"')[1]
        elif line.startswith('  id:') or line.startswith('  label_id:'):
            class_id = int(line.strip().split(' ')[-1])
            labelmap.append({'id': class_id, 'name': name})
            class_ids.add(class_id)
    return labelmap, class_ids


# Seems there is at most 100 detections for each image
def via3_eval(results,video_infos, opt_ids,
              opt_ids2opt_labels, opt_ids2opt_names,
              metric, max_dets=(100, ), verbose=True):

    assert metric in ['mAP']
    start = time.time()
    gt_boxes_all, gt_labels_all =defaultdict(list), defaultdict(list)
    for video_info in  video_infos:
        for gt_box,gt_labels in zip(video_info['ann']['gt_bboxes'], video_info['ann']['gt_labels']):
            gt_labels = np.argwhere(gt_labels==1)
            for gt_label in  gt_labels:
                gt_boxes_all[video_info['img_key']].append(gt_box)
                gt_labels_all[video_info['img_key']].append(gt_label[0])
        #print(gt_labels_all[video_info['img_key']])
    if verbose:
        print_time('process groundtruth results', start)

    categories = [{'id': opt_ids2opt_labels[opt_id], 'name': opt_ids2opt_names[opt_id] } for opt_id in opt_ids[1:]]


    start = time.time()
    pred_boxes_all, pred_labels_all, pred_scores_all = defaultdict(list), defaultdict(list), defaultdict(list)
    for result, video_info in zip(results, video_infos):
        img_key = video_info['img_key']
        for label, bboxes in enumerate(result):
            for bbox in bboxes:
                box, score = bbox[:4], bbox[4]
                if score < 0.3:
                    continue
                pred_boxes_all[img_key].append(box)
                pred_labels_all[img_key].append(label)
                pred_scores_all[img_key].append(score)

    if verbose:
        print_time('process prediction results', start)


    if metric == 'mAP':
        pascal_evaluator = det_eval.PascalDetectionEvaluator(categories, label_id_offset=0, num_class_offset=1)
        start = time.time()
        for image_key in gt_boxes_all:
            pascal_evaluator.add_single_ground_truth_image_info(
                image_key, {
                    standard_fields.InputDataFields.groundtruth_boxes:
                    np.array(gt_boxes_all[image_key], dtype=float),
                    standard_fields.InputDataFields.groundtruth_classes:
                    np.array(gt_labels_all[image_key], dtype=int),
                    standard_fields.InputDataFields.groundtruth_difficult:
                    np.zeros(len(gt_boxes_all[image_key]), dtype=bool)
                })
        if verbose:
            print_time('Convert groundtruth', start)

        start = time.time()
        for image_key in pred_boxes_all:
            pascal_evaluator.add_single_detected_image_info(
                image_key, {
                    standard_fields.DetectionResultFields.detection_boxes:
                    np.array(pred_boxes_all[image_key], dtype=float),
                    standard_fields.DetectionResultFields.detection_classes:
                    np.array(pred_labels_all[image_key], dtype=int),
                    standard_fields.DetectionResultFields.detection_scores:
                    np.array(pred_scores_all[image_key], dtype=float)
                })
        if verbose:
            print_time('convert prediction', start)

        start = time.time()
        metrics = pascal_evaluator.evaluate()
        if verbose:
            print_time('run_evaluator', start)
        for display_name in metrics:
            print(f'{display_name}=\t{metrics[display_name]}')
        return {
            display_name: metrics[display_name]
            for display_name in metrics if 'ByCategory' not in display_name
        }
