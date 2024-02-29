from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration
from transformers import AutoTokenizer
from datasets import load_dataset
from utils.utils import tokenize_for_inference



def get_data():
    dataset = load_dataset("multi_news")
    return dataset


model_nm = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_nm)


def tokenize_data(x):
  model_inputs = tokenizer(
      x['document'],
      max_length=512,
      padding=True,
      truncation=True
  )
  labels = tokenizer(
      x['summary'],
      max_length=512,
      padding=True,
      truncation=True
  )
  model_inputs['labels'] = labels['input_ids']
  return model_inputs


def preprocess():
    dataset = get_data()
    tok_ds = dataset.map(tokenize_data, batched=True)
    return tok_ds


def train_model(tok_ds, num_train_epochs, batch_size):
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=2,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["validation"],
        # data_collator=data_collator,
        compute_metrics=lambda p: compute_rouge_scores(
            tokenizer.batch_decode(p.predictions, skip_special_tokens=True),
            tokenizer.batch_decode(p.label_ids, skip_special_tokens=True),
        ),
    )
    trainer.train()
    return trainer


def evaluate_model(trainer):
    eval_metrics = trainer.evaluate()


def infer_model(trainer):
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    text = input("Enter the text you want to summarize: ")
    tokenized = tokenize_for_inference(text)
    generated = trainer.model.generate(tokenized, max_length=256)

    # Convert the generated output back to text
    summary = tokenizer.decode(generated.squeeze(), skip_special_tokens=True)
    print(summary)
    return summary


def training_pipeline(num_train_epochs, batch_size):
    tok_ds = preprocess()
    # data_collator = DataCollatorForSeq2Seq(tokenizer,model=model,return_tensors='pt')
    trainer = train_model(tok_ds, num_train_epochs, batch_size)
    trained_model = trainer.model
    eval_metric = evaluate_model(trainer)
    infer_model(trainer)
