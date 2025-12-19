from datasets import load_dataset

dataset_name = "ayoubkirouane/llava-instruct-small"

# Load Dataset
dataset = load_dataset(dataset_name)

# import os
# import zipfile
# import io
# # from datasets import DatasetDict
# from huggingface_hub import hf_hub_download, list_repo_files
# from PIL import Image

# dataset_train_split = "test"

# def format_data(samples: dict[str, any]) -> dict[str, list]:
#     formatted_samples = {"messages": []}
#     for cont in range(len(samples["question"])):
#         images = []
#         for img_path in samples["input_image_path"][cont]:
#             try:
#                 with open(img_path, "rb") as f:
#                     img_bytes = f.read()
#                 image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#                 images.append({"type": "image", "image": image})
#             except Exception as e:
#                 print(f"Error processing image {img_path}: {e}")
#                 continue

#         formatted_samples["messages"].append(
#             [
#                 {"role": "system", "content": [{"type": "text", "text": samples["context"][cont]}]},
#                 {"role": "user", "content": images + [{"type": "text", "text": samples["question"][cont]}]},
#                 {"role": "assistant", "content": [{"type": "text", "text": samples["output"][cont]}]},
#             ]
#         )
#     return formatted_samples

# For multi-image example
# def prepare_dataset(dataset: DatasetDict, dataset_name: str, dataset_train_split: str) -> DatasetDict:
#     all_files = list_repo_files(dataset_name, repo_type="dataset")
#     zip_files = [f for f in all_files if f.endswith(".zip")]

#     for zip_filename in zip_files:
#         zip_path = hf_hub_download(repo_id=dataset_name, filename=zip_filename, repo_type="dataset")
#         extract_folder = zip_filename.replace(".zip", "")
#         os.makedirs(extract_folder, exist_ok=True)

#         with zipfile.ZipFile(zip_path, "r") as zip_ref:
#             zip_ref.extractall(extract_folder)

#     dataset = dataset.map(format_data, batched=True, batch_size=4, num_proc=16)
#     return dataset

# dataset = prepare_dataset(dataset, dataset_name, dataset_train_split)
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

# Load model and tokenizer
model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    attn_implementation="eager", # Important (Ref: https://github.com/huggingface/transformers/blob/c15a7adb283fa984a40558c7fe7bed30ae975cdd/src/transformers/models/gemma3/modeling_gemma3.py#L934)
    quantization_config=bnb_config
)
processor = AutoProcessor.from_pretrained(model_id,use_fast=True)
processor.tokenizer.padding_side = "right"

from peft import LoraConfig, get_peft_model

# Configure QLoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

from trl import SFTConfig

training_args = SFTConfig(
    output_dir="smolvlm-trl-sft-test",     # Directory to save the model and push to the Hub. Use a specific repository id (e.g., gemma-3-4b-it-trl-sft-MMIU-Benchmark for multi-image datasets).
    num_train_epochs=1,                                             # Set the number of epochs to train the model.
    per_device_train_batch_size=2,                                  # Batch size for each device (e.g., GPU) during training. multi-image -> per_device_train_batch_size=1
    gradient_accumulation_steps=32,                                  # Number of steps before performing a backward/update pass to accumulate gradients. multi-image -> gradient_accumulation_steps=1
    gradient_checkpointing=True,                                    # Enable gradient checkpointing to reduce memory usage during training.
    optim="adamw_torch_fused",                                      # Use the fused AdamW optimizer for better performance.
    save_strategy="epoch",                                          # Save checkpoints at the end of each epoch.
    learning_rate=2e-05,                                            # Learning rate for training.
    bf16=True,                                                      # Enable bfloat16 precision for training to save memory and speed up computations.
    push_to_hub=False,                                               # Automatically push the fine-tuned model to Hugging Face Hub after training.
    report_to="tensorboard",                                        # Automatically report metrics to tensorboard.
    gradient_checkpointing_kwargs={"use_reentrant": False},         # Set gradient checkpointing to non-reentrant to avoid issues.
    dataset_kwargs={"skip_prepare_dataset": True},                  # Skip dataset preparation to handle preprocessing manually.
    remove_unused_columns=False,                                    # Ensure unused columns are not removed in the collator (important for batch processing).
)
from PIL import Image

# For multi-image cases
def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                if image is not None:
                    image = Image.open(io.BytesIO(image["bytes"]))
                    image_inputs.append(image.convert("RGB"))
    return image_inputs

def collate_fn(examples):
    texts = [processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False).strip() for example in examples]
    if "images" in examples[0]:  # single-image
        images = [
            [img.convert("RGB") for img in example["images"]]
            for example in examples
        ]
    else:  # multi-image
        images = [process_vision_info(example["messages"]) for example in examples]

    # Tokenize the texts and process the images
    batch = processor(
        images=images, text=texts, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    # Mask image tokens
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    # labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch  # Return the prepared batch

    # Training
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=dataset["train"], # multi-image -> train_dataset=dataset["test"],
    processing_class=processor,
    peft_config=peft_config,
)

trainer.train()

# Save the final model
trainer.save_model()