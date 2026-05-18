
import torch
from mmseg.models import SEGMENTORS
from mmseg.models import build_backbone

from .base_segmentor import CustomBaseSegmentor

@SEGMENTORS.register_module()
class BEVSegmentor(CustomBaseSegmentor):

    def __init__(
        self,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        backbone_fp16=False,
        history_no_grad=False,
        # use_post_fusion=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # self.fp16_enabled = False
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices
        self.backbone_fp16 = backbone_fp16
        # Optimization: when True, during training the backbone+encoder forward
        # for historical frames (all frames except the last/current one) runs
        # under torch.no_grad(), so they do not store activations nor receive
        # gradients. Only the current frame contributes gradients to backbone +
        # encoder. The temporal_encoder still sees all F frames concatenated.
        self.history_no_grad = history_no_grad
        # self.use_post_fusion = use_post_fusion

        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if extra_img_backbone is not None:
            self.extra_img_backbone = build_backbone(extra_img_backbone)

    def _run_img_backbone_flat(self, imgs_flat):
        """Run backbone+FPN on a flat (M, C, H, W) image batch.
        Returns the multi-scale FPN feature list, each (M, C', H', W').
        """
        with torch.cuda.amp.autocast(enabled=self.backbone_fp16, dtype=torch.bfloat16):
            img_feats_backbone = self.img_backbone(imgs_flat)
            if isinstance(img_feats_backbone, dict):
                img_feats_backbone = list(img_feats_backbone.values())
            img_feats = [img_feats_backbone[idx] for idx in self.img_backbone_out_indices]
            img_feats = self.img_neck(img_feats)
        if self.backbone_fp16:
            img_feats = [f.float() for f in img_feats]
        return img_feats

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images.

        If ``self.history_no_grad`` is True and we are in training mode with
        F > 1, the history frames go through the backbone under no_grad and
        the current (last) frame goes through normally with autograd. The
        outputs are then concatenated back to the original (B*F, N, C, H, W)
        layout so the rest of the pipeline is unchanged.
        """
        B, F, N, C, H, W = imgs.size()

        if self.training and self.history_no_grad and F > 1:
            imgs_hist = imgs[:, :-1].reshape(B * (F - 1) * N, C, H, W)
            imgs_curr = imgs[:, -1:].reshape(B * 1 * N, C, H, W)

            with torch.no_grad():
                feats_hist = self._run_img_backbone_flat(imgs_hist)
            feats_curr = self._run_img_backbone_flat(imgs_curr)

            img_feats_reshaped = []
            for fh, fc in zip(feats_hist, feats_curr):
                Cf, Hf, Wf = fh.shape[1:]
                fh = fh.view(B, F - 1, N, Cf, Hf, Wf)
                fc = fc.view(B, 1, N, Cf, Hf, Wf)
                merged = torch.cat([fh, fc], dim=1).reshape(B * F, N, Cf, Hf, Wf)
                img_feats_reshaped.append(merged)
            return {'ms_img_feats': img_feats_reshaped}

        # Original path: process all B*F*N images together.
        imgs_flat = imgs.reshape(B * F * N, C, H, W)
        img_feats = self._run_img_backbone_flat(imgs_flat)
        img_feats_reshaped = [
            f.view(B * F, N, *f.shape[1:]) for f in img_feats
        ]
        return {'ms_img_feats': img_feats_reshaped}

    def forward_extra_img_backbone(self, imgs, **kwargs):
        """Extract features of images."""
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.extra_img_backbone(imgs)

        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())

        img_feats_backbone_reshaped = []
        for img_feat_backbone in img_feats_backbone:
            BN, C, H, W = img_feat_backbone.size()
            img_feats_backbone_reshaped.append(
                img_feat_backbone.view(B, int(BN / B), C, H, W))
        return img_feats_backbone_reshaped

    # ---------------------------------------------------------------------
    # Helpers for history-no-grad split encoder forward
    # ---------------------------------------------------------------------
    @staticmethod
    def _slice_metas_along_frames(metas, B, F, frame_slice):
        """Return a shallow copy of ``metas`` with ``projection_mat`` and
        ``image_wh`` sliced along the frame dimension.

        Shape conventions (after dataloader collation):
          - ``projection_mat``: (B, F*N, 4, 4)
          - ``image_wh``:       (B, F, N, 2)

        ``frame_slice`` is a python slice object selecting frames.
        """
        out = dict(metas)
        pm = metas.get('projection_mat', None)
        if pm is not None and isinstance(pm, torch.Tensor):
            # (B, F*N, 4, 4) -> (B, F, N, 4, 4) -> slice -> flatten back
            N = pm.shape[1] // F
            pm_r = pm.view(B, F, N, 4, 4)[:, frame_slice]
            Fs = pm_r.shape[1]
            out['projection_mat'] = pm_r.reshape(B, Fs * N, 4, 4).contiguous()

        iw = metas.get('image_wh', None)
        if iw is not None and isinstance(iw, torch.Tensor) and iw.dim() >= 3:
            # (B, F, N, 2) -> slice -> keep
            out['image_wh'] = iw[:, frame_slice].contiguous()
        return out

    def _encoder_forward_split(self, results):
        """Run encoder twice: history frames under no_grad, current frame
        with autograd. The merged ``anchors`` / ``instance_feature`` keep the
        original (B*F, ...) layout so downstream temporal_encoder is unchanged.
        """
        imgs = results['imgs']
        B, F, N = imgs.size(0), imgs.size(1), imgs.size(2)
        metas = results['metas']

        # --- split anchor (lifter output): (B*F, num_anchor, C) ----------
        anchor = results['representation']
        num_anchor, Ca = anchor.shape[1], anchor.shape[2]
        anchor_v = anchor.view(B, F, num_anchor, Ca)
        anchor_hist = anchor_v[:, :-1].reshape(B * (F - 1), num_anchor, Ca)
        anchor_curr = anchor_v[:, -1:].reshape(B * 1, num_anchor, Ca)

        # --- split ms_img_feats: each is (B*F, N, C, H, W) ---------------
        feats_hist, feats_curr = [], []
        for f in results['ms_img_feats']:
            Cf, Hf, Wf = f.shape[2], f.shape[3], f.shape[4]
            fv = f.view(B, F, N, Cf, Hf, Wf)
            feats_hist.append(fv[:, :-1].reshape(B * (F - 1), N, Cf, Hf, Wf))
            feats_curr.append(fv[:, -1:].reshape(B * 1, N, Cf, Hf, Wf))

        # --- slice metas along frame dim ---------------------------------
        metas_hist = self._slice_metas_along_frames(metas, B, F, slice(None, -1))
        metas_curr = self._slice_metas_along_frames(metas, B, F, slice(-1, None))

        # --- forward ------------------------------------------------------
        with torch.no_grad():
            out_hist = self.encoder(
                representation=anchor_hist,
                ms_img_feats=feats_hist,
                metas=metas_hist,
            )
        out_curr = self.encoder(
            representation=anchor_curr,
            ms_img_feats=feats_curr,
            metas=metas_curr,
        )

        # --- merge anchors / instance_feature back to (B*F, ...) ---------
        def _merge(hist, curr):
            tail = hist.shape[1:]
            h = hist.view(B, F - 1, *tail)
            c = curr.view(B, 1, *tail)
            return torch.cat([h, c], dim=1).reshape(B * F, *tail)

        anchors_merged = _merge(out_hist['anchors'], out_curr['anchors'])
        inst_feat_merged = _merge(
            out_hist['instance_feature'], out_curr['instance_feature']
        )

        # ``representation`` from the encoder (last-layer gaussian dict) is
        # not consumed downstream (head uses ``representation_temp`` from the
        # temporal encoder). Return the current-frame one for consistency.
        return {
            'representation': out_curr['representation'],
            'instance_feature': inst_feat_merged,
            'anchors': anchors_merged,
        }

    def forward(self,
                imgs=None,
                metas=None,
                points=None,
                extra_backbone=False,
                occ_only=False,
                rep_only=False,
                **kwargs,
        ):
        """Forward training function.
        """
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=imgs)

        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points,
            'gt_boxes': metas['gt_boxes'],
        }
        results.update(kwargs)
        outs = self.extract_img_feat(**results)
        results.update(outs)
        outs = self.lifter(**results)
        results.update(outs)

        # Encoder: optionally run history frames under no_grad to save
        # memory and backward time. Activated only in training when
        # history_no_grad is enabled and we actually have multiple frames.
        if (self.training
                and self.history_no_grad
                and imgs is not None
                and imgs.dim() >= 2
                and imgs.size(1) > 1):
            outs = self._encoder_forward_split(results)
        else:
            outs = self.encoder(**results)
        results.update(outs)
        if hasattr(self, 'temporal_encoder'):
            outs = self.temporal_encoder(**results)
            results.update(outs)

        outs = self.decoder(results)
        results.update(outs)
        if hasattr(self, 'map_decoder'):
            # TODO 需要直接传入gt数据，4个
            outs = self.map_decoder(results)
            results.update(outs)

        outs = self.planner_head(results)
        results.update(outs)

        outs = self.head(**results)
        results.update(outs)

        return results
