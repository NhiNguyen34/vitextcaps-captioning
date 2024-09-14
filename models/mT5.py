from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from transformers import AutoTokenizer, AutoConfig
import torch
from torch import nn
import torch.nn.functional as F
from utils.logging_utils import setup_logger
from builders.model_builder import META_ARCHITECTURE
logger = setup_logger()


@META_ARCHITECTURE.register()
class MT5_MODEL(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.pretrained_config = AutoConfig.from_pretrained("google/mt5-base")
        self.build()

    def build(self):
        # split model building into several components
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_encoder_decoder()
        # self._build_decoder()

    def _build_obj_encoding(self):
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.OBJECT_EMBEDDING.D_FEATURE, self.config.D_MODEL
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, self.config.D_MODEL)

        self.obj_feat_layer_norm = nn.LayerNorm(self.config.D_MODEL)
        self.obj_bbox_layer_norm = nn.LayerNorm(self.config.D_MODEL)
        self.obj_drop = nn.Dropout(self.config.OBJECT_EMBEDDING.DROPOUT)

    def _build_ocr_encoding(self):
        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.OCR_EMBEDDING.D_FEATURE, self.config.D_MODEL
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, self.config.D_MODEL)

        # OCR word embedding features
        # self.ocr_word_embedding = build_word_embedding(self.config.OCR_TEXT_EMBEDDING)

        self.ocr_feat_layer_norm = nn.LayerNorm(self.config.D_MODEL)
        self.ocr_bbox_layer_norm = nn.LayerNorm(self.config.D_MODEL)
        self.ocr_text_layer_norm = nn.LayerNorm(self.config.D_MODEL)
        self.ocr_drop = nn.Dropout(self.config.OCR_EMBEDDING.DROPOUT)

    # def _build_encoder(self):
    #     self.pretrained_config.update({
    #         "is_decoder": False,
    #         "add_cross_attention": False
    #     })
    #     self.encoder = MT5EncoderModel(self.pretrained_config)
    #     self.encoder.from_pretrained("google/mt5-base")

    def _build_encoder_decoder(self):
        # self.pretrained_config.update({
        #     "is_decoder": True,
        #     "add_cross_attention": True
        # })
        mt5_model = MT5ForConditionalGeneration(self.pretrained_config)
        mt5_model.from_pretrained("google/mt5-base")

        self.encoder = mt5_model.encoder
        self.decoder = mt5_model.decoder
        self.lm_head = mt5_model.lm_head

        del mt5_model

    def forward(self, items):
        fwd_results = {}
        self._forward_obj_encoding(items, fwd_results)
        self._forward_ocr_encoding(items, fwd_results)
        self._forward_encoder(items, fwd_results)
        self._forward_decoder(items, fwd_results)
        self._forward_result(fwd_results)
        results = {"scores": fwd_results["scores"]}
        return results

    def _forward_result(self, fwd_results):
        fwd_results['scores'] = self.lm_head(fwd_results['decoder_outputs'])

    def _forward_obj_encoding(self, items, fwd_results):
        # object appearance feature
        obj_feat = items.region_features
        obj_bbox = items.region_boxes.squeeze()
        obj_mmt_in = self.obj_feat_layer_norm(
            self.linear_obj_feat_to_mmt_in(obj_feat)
        ) + self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(obj_bbox))
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results["obj_mmt_in"] = obj_mmt_in
        obj_nums = torch.tensor([obj_feat.shape[1]]*obj_mmt_in.shape[0])
        fwd_results["obj_mask"] = _get_mask(obj_nums, obj_mmt_in.size(1))

    def _forward_ocr_encoding(self, items, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_fasttext = items.ocr_fasttext_features
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR rec feature (256-dim), replace the OCR PHOC features, extracted from swintextspotter
        ocr_phoc = items.ocr_rec_features
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 256

        # OCR appearance feature, extracted from swintextspotter
        ocr_fc = items.ocr_det_features
        ocr_fc = F.normalize(ocr_fc, dim=-1)
        max_len = max(ocr_fasttext.shape[1], ocr_phoc.shape[1], ocr_fc.shape[1])

        if ocr_fasttext.shape[1] < max_len:
            ocr_fasttext = torch.nn.functional.pad(ocr_fasttext,
                                                    (0, 0, 0, max_len - ocr_fasttext.shape[1], 0, 0))
        elif ocr_phoc.shape[1] < max_len:
            ocr_phoc = torch.nn.functional.pad(ocr_phoc,
                                               (0, 0, 0, max_len - ocr_phoc.shape[1], 0, 0))
        elif ocr_fc.shape[1] < max_len:
            ocr_fc = torch.nn.functional.pad(ocr_fc,
                                             (0, 0, 0, max_len - ocr_fc.shape[1], 0, 0))

        ocr_feat = torch.cat(
            [ocr_fasttext, ocr_phoc, ocr_fc], dim=-1
        )
        ocr_bbox = items.ocr_boxes.squeeze()
        ocr_mmt_in = self.ocr_feat_layer_norm(
            self.linear_ocr_feat_to_mmt_in(ocr_feat)
        ) + self.ocr_bbox_layer_norm(self.linear_ocr_bbox_to_mmt_in(ocr_bbox))
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)

        fwd_results["ocr_mmt_in"] = ocr_mmt_in
        mask = ocr_fc.sum(dim=1) != 0

        max_feat = []
        for i in range(items.ocr_fasttext_features.shape[0]):
            item_ = items.ocr_fasttext_features[i]
            mask = item_.sum(dim=1) != 0
            max_feat.append(item_[mask].shape[0])
        max_feat = torch.tensor(max_feat)
        fwd_results["ocr_mask"] = _get_mask(max_feat, ocr_mmt_in.size(1)).squeeze()


    def _forward_encoder(self, items, fwd_results):
        input_embs = torch.cat([fwd_results['obj_mmt_in'], fwd_results['ocr_mmt_in']], dim=1)
        input_mask = torch.cat([fwd_results['obj_mask'], fwd_results['ocr_mask']], dim=1)

        encoder_outputs = self.encoder(
            inputs_embeds=input_embs,
            attention_mask=input_mask
        ).hidden_states

        fwd_results['encoder_outputs'] = encoder_outputs
        fwd_results['input_embs'] = input_embs
        fwd_results['input_mask'] = input_mask


    def _forward_decoder(self, items, fwd_results):
        decoder_outputs = self.decoder(
            input_ids=items.answer_tokens.squeeze(),
            attention_mask=items.answer_masks.squeeze(),
            encoder_hidden_states=fwd_results['encoder_outputs'],
            encoder_attention_mask=fwd_results['input_mask'],
        )

        fwd_results['decoder_outputs'] = decoder_outputs[0]

def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask
