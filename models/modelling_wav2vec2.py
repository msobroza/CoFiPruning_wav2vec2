# allows removing layers of heads and mlps
import pdb

import transformers

from utils.cofi_utils import load_pruned_model

__version__ = transformers.__version__

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForPreTrainingOutput

from transformers import AutoConfig, Wav2Vec2Config
from transformers.modeling_outputs import CausalLMOutput, Wav2Vec2BaseModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (Wav2Vec2ForCTC,
                                                            Wav2Vec2Model,
                                                            _HIDDEN_STATES_START_POSITION,
                                                            Wav2Vec2FeatureEncoder,
                                                            Wav2Vec2FeatureProjection,
                                                            
                                                           )


import torch
from torch import nn
from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
import logging
from models.modeling_bert import CoFiLayerNorm 

logger = logging.getLogger(__name__)

BertLayerNorm = CoFiLayerNorm

class CoFiWav2Vec2FeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = CoFiLayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states, mlp_z, hidden_z=None, inference=False):
        # non-projected hidden states are needed for quantization
        if not inference and hidden_states.sum().eq(0).item():
           norm_hidden_states = hidden_states
        else:
            if hidden_z is not None:
                hidden_states = hidden_states.mult(hidden_z)
            norm_hidden_states = self.layer_norm(hidden_states, hidden_z)
            if hidden_z is not None:
                norm_hidden_states = norm_hidden_states.mul(norm_hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


class CoFiWav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = CoFiWav2Vec2FeatureProjection(config)
        if config.do_stable_layer_norm:
            self.encoder = CoFiWav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = CoFiWav2Vec2Encoder(config)
        self.adapter = CoFiWav2Vec2Adapter(config) if config.add_adapter else None
        self.post_init()
    
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class CoFiWav2Vec2ForCTC(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = CoFiWav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        self.do_layer_distill = getattr(config, "do_layer_distill", False)

        if self.do_layer_distill:
            self.layer_transformation = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.layer_transformation = None

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        if os.path.exists(pretrained_model_name_or_path):
            weights = torch.load(os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
        else:
            archive_file = hf_bucket_url(pretrained_model_name_or_path, filename="pytorch_model.bin") 
            resolved_archive_file = cached_path(archive_file)
            weights = torch.load(resolved_archive_file, map_location="cpu")

        
        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in weights.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            weights[new_key] = weights.pop(old_key)

        if "config" not in kwargs:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            config.do_layer_distill = False
        else:
            config = kwargs["config"]
        
        model = cls(config)

        load_pruned_model(model, weights)
        return model
