import numpy as np
import torch
from mmaction.core.bbox import bbox2result
from mmaction.utils import import_module_error_class

try:
    from mmdet.core.bbox import bbox2roi
    from mmdet.models import HEADS as MMDET_HEADS
    from mmdet.models.roi_heads import StandardRoIHead
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:

    @MMDET_HEADS.register_module()
    class AVARoIHead(StandardRoIHead):
        def _bbox_forward(self, x, rois):
            rois = rois.float()
            bbox_feat = self.bbox_roi_extractor(x, rois)
            if self.with_shared_head:
                bbox_feat = self.shared_head(bbox_feat)
            cls_score, bbox_pred = self.bbox_head(bbox_feat)
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feat)
            return bbox_results

        def simple_test(self,
                        x,
                        proposal_list,
                        img_metas,
                        proposals=None,
                        rescale=False):
            assert self.with_bbox, 'Bbox head must be implemented.'

            if isinstance(x, tuple):
                x_shape = x[0].shape
            else:
                x_shape = x.shape

            assert x_shape[0] == 1, 'only accept 1 sample at test mode'
            assert x_shape[0] == len(img_metas) == len(proposal_list)

            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
            bbox_results = bbox2result(
                det_bboxes,
                det_labels,
                self.bbox_head.num_classes,
                thr=self.test_cfg.action_thr)
            return [bbox_results]

        def simple_test_bboxes(self,
                               x,
                               img_metas,
                               proposals,
                               rcnn_test_cfg,
                               rescale=False):
            """Test only det bboxes without augmentation."""
            rois = bbox2roi(proposals)
            bbox_results = self._bbox_forward(x, rois)
            cls_score = bbox_results['cls_score']

            img_shape = img_metas[0]['img_shape']
            crop_quadruple = np.array([0, 0, 1, 1])
            flip = False

            if 'crop_quadruple' in img_metas[0]:
                crop_quadruple = img_metas[0]['crop_quadruple']

            if 'flip' in img_metas[0]:
                flip = img_metas[0]['flip']

            det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                img_shape,
                flip=flip,
                crop_quadruple=crop_quadruple,
                cfg=rcnn_test_cfg)

            return det_bboxes, det_labels
else:
    # Just define an empty class, so that __init__ can import it.
    @import_module_error_class('mmdet')
    class AVARoIHead:
        pass



import numpy as np
import torch
from mmaction.core.bbox import bbox2result
from mmaction.utils import import_module_error_class

try:
    from mmdet.core.bbox import bbox2roi
    from mmdet.models import HEADS as MMDET_HEADS
    from mmdet.models.roi_heads import StandardRoIHead
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:

    @MMDET_HEADS.register_module()
    class Via3RoIHead(StandardRoIHead):
        def _bbox_forward(self, x, rois):
            rois = rois.float()
            bbox_feat = self.bbox_roi_extractor(x, rois)
            if self.with_shared_head:
                bbox_feat = self.shared_head(bbox_feat)
            cls_score, bbox_pred = self.bbox_head(bbox_feat)
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feat)
            return bbox_results

        def bbox2result(self, bboxes, labels, num_classes, thr=0.01):
            """Convert detection results to a list of numpy arrays.

            Args:
                bboxes (Tensor): shape (n, 4)
                labels (Tensor): shape (n, #num_classes)
                num_classes (int): class number, including background class
                thr (float): The score threshold used when converting predictions to
                    detection results
            Returns:
                list(ndarray): bbox results of each class
            """
            if bboxes.shape[0] == 0:
                return list(np.zeros((num_classes - 1, 0, 5), dtype=np.float32))
            else:
                bboxes = bboxes.cpu().numpy()
                labels = labels.cpu().numpy()

                # We only handle multilabel now
                assert labels.shape[-1] > 1

                scores = labels  # rename for clarification
                thr = (thr,) * num_classes if isinstance(thr, float) else thr
                assert scores.shape[1] == num_classes
                assert len(thr) == num_classes

                result = []
                for i in range(num_classes):
                    where = scores[:, i] > thr[i]
                    result.append(
                        np.concatenate((bboxes[where, :4], scores[where, i:i +1]),
                                       axis=1))
                return result

        def simple_test(self,
                        x,
                        proposal_list,
                        img_metas,
                        proposals=None,
                        rescale=False):
            assert self.with_bbox, 'Bbox head must be implemented.'

            if isinstance(x, tuple):
                x_shape = x[0].shape
            else:
                x_shape = x.shape

            assert x_shape[0] == 1, 'only accept 1 sample at test mode'
            assert x_shape[0] == len(img_metas) == len(proposal_list)

            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
            bbox_results = self.bbox2result(
                det_bboxes,
                det_labels,
                self.bbox_head.num_classes,
                thr=self.test_cfg.action_thr)
            return [bbox_results]

        def simple_test_bboxes(self,
                               x,
                               img_metas,
                               proposals,
                               rcnn_test_cfg,
                               rescale=False):
            """Test only det bboxes without augmentation."""
            rois = bbox2roi(proposals)
            bbox_results = self._bbox_forward(x, rois)
            cls_score = bbox_results['cls_score']

            img_shape = img_metas[0]['img_shape']
            crop_quadruple = np.array([0, 0, 1, 1])
            flip = False

            if 'crop_quadruple' in img_metas[0]:
                crop_quadruple = img_metas[0]['crop_quadruple']

            if 'flip' in img_metas[0]:
                flip = img_metas[0]['flip']

            det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                img_shape,
                flip=flip,
                crop_quadruple=crop_quadruple,
                cfg=rcnn_test_cfg)

            return det_bboxes, det_labels
else:
    # Just define an empty class, so that __init__ can import it.
    @import_module_error_class('mmdet')
    class Via3RoIHead:
        pass
