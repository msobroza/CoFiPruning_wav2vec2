import logging
import os
import sys
from copy import deepcopy
import re
import torch
import transformers
from datasets import load_dataset, load_metric
from transformers import (AutoConfig, AutoTokenizer, EvalPrediction,
                          HfArgumentParser, PreTrainedTokenizerFast)
from transformers import TrainingArguments, default_data_collator, set_seed
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from args import AdditionalArguments
from models.l0_module import L0Module
from models.model_args import ModelArguments
from models.modelling_wav2vec2 import CoFiWav2Vec2ForCTC
from trainer.trainer_wav2vec2_ctc import CoFIWav2Vec2ForCTCTrainer
from utils.cofi_utils import *
from utils.utils import *

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def main():
    parser = HfArgumentParser(
        (ModelArguments, TrainingArguments, AdditionalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, additional_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [
            -1, 0] else logging.WARN,
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info("Training/evaluation.py parameters %s", training_args)

    # save args
    torch.save(data_args, os.path.join(
        training_args.output_dir, "data_args.bin"))
    torch.save(model_args, os.path.join(
        training_args.output_dir, "model_args.bin"))
    torch.save(additional_args, os.path.join(
        training_args.output_dir, "additional_args.bin"))

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # print all arguments
    log_all_parameters(logger, model_args, data_args,
                       training_args, additional_args)

    datasets = load_dataset("librispeech_asr")
    datasets = datasets.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # set up configuration for distillation
    if additional_args.do_distil:
        config.output_attentions = True
        config.output_hidden_states = True

    Model = CoFiWav2Vec2ForCTC

    teacher_model = None
    if additional_args.do_distill:
        teacher_model = Model.from_pretrained(
           "facebook/wav2vec2-base",
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
            config=deepcopy(config)
        )
        teacher_model.eval()

    model = Model.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # initialize the layer transformation matrix to be an identity matrix
    if additional_args.do_layer_distill:
        initialize_layer_transformation(model)

    logger.info(model)
    logger.info(f"Model size: {calculate_parameters(model)}")

    zs = None
    if additional_args.pretrained_pruned_model is not None and not additional_args.pretrained_pruned_model == "None":
        zs = load_zs(additional_args.pretrained_pruned_model)

        model = load_model(additional_args.pretrained_pruned_model, Model, zs)
        prune_model_with_z(zs, model)
        print(
            f"Model Size after pruning and padding: {calculate_parameters(model)}")

    l0_module = None
    if additional_args.str_pruning_method == "l0_reg":
        l0_module = L0Module(config=config,
                             droprate_init=additional_args.droprate_init,
                             temperature=additional_args.temperature,
                             target_sparsity=additional_args.target_sparsity,
                             pruning_type=additional_args.pruning_type)

    # Tokenizer check: this script requires a fast tokenizcser.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.py.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

    def preprocessing(batch):
        batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower() + " "
        all_text = " ".join(batch["text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            preprocessing,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

   
    if training_args.do_eval:
        # validation_dataset = load_and_cache_examples(training_args, data_args, model_args, tokenizer, evaluate=True, output_examples=False)
        validation_dataset = datasets["validation"].map(
            preprocessing,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    
    metric = load_metric("wer")

    def compute_metrics(pred: EvalPrediction):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # Initialize our Trainer
    trainer = CoFIWav2Vec2ForCTCTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        additional_args=additional_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
        logger=logger,
        l0_module=l0_module,
        teacher_model=teacher_model
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
