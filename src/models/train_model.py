# train_model.py contains the code to train the model
# Note that it mostly contains code from `4_0_t5_model.ipynb` notebook

# Necessary inputs
import warnings

import transformers
from datasets import DatasetDict, Dataset, load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import numpy as np

warnings.filterwarnings("ignore")


def preprocess_function(df):
    inputs = [prefix + ex for ex in df["reference"]]
    targets = [ex for ex in df["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# simple postprocessing for text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


# compute metrics function to pass to trainer
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


if __name__ == "__main__":
    # selecting model checkpoint
    model_checkpoint = "t5-small"

    # setting random seed for transformers library
    transformers.set_seed(42)

    # Load the BLUE metric
    metric = load_metric("sacrebleu")

    # load raw dataset
    raw_datasets = load_dataset(
        "csv",
        data_files={
            "train": "../../data/interim/train.csv",
            "validation": "../../data/interim/val.csv",
        },
    )

    # we will use autotokenizer for this purpose
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # prefix for model input
    prefix = "paraphrase toxic sentences:"

    max_input_length = 128
    max_target_length = 128

    # split train dataset into train and test
    train_dataset = raw_datasets["train"].train_test_split(test_size=0.1)
    val_dataset = raw_datasets["validation"]

    # train_dataset, val_dataset
    tokenized_train = (
        train_dataset["train"]
        .map(preprocess_function, batched=True)
        .select(range(5000))
    )
    tokenized_test = (
        train_dataset["test"].map(preprocess_function, batched=True).select(range(500))
    )
    tokenized_val = val_dataset.map(preprocess_function, batched=True).select(
        range(500)
    )

    # create a model for the pretrained model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # defining the parameters for training
    batch_size = 32
    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-toxicity",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=True,
    )

    # instead of writing collate_fn function we will use DataCollatorForSeq2Seq
    # simliarly it implements the batch creation for training
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # instead of writing train loop we will use Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # saving model
    trainer.save_model("best")
