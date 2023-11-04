# predict_model.py is a script which loads last checkpoint
# and provides a simple interface for user to input text and get transformation

# Necessary inputs
import warnings
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

warnings.filterwarnings("ignore")


def translate(model, inference_request, tokenizer):
    """
    translate is helper function which allows easier inference of model

    Args:
        model (transformers.modeling_utils.PreTrainedModel): model to use for inference
        inference_request (str): input string to transform
        tokenizer (transformers.tokenization_utils.PreTrainedTokenizer): tokenizer to use for inference
    """
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0)


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Detoxicate text")

    # Add argument for input string
    parser.add_argument(
        "--input", type=str, help="Input sentence to transform", required=True
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Access the input string from args and process it
    input_string = args.input
    print("Parsed input:", input_string)

    # load model
    model = AutoModelForSeq2SeqLM.from_pretrained("./models/best")

    # get tokenizer from model
    tokenizer = AutoTokenizer.from_pretrained("./models/best")

    # translate
    result = translate(model.to("cpu"), input_string, tokenizer)
    print("Result:", result)
