'''train bert based model for emotion analysis'''
from transformers import pipeline,AutoTokenizer,AutoModelForSequenceClassification
from transformers import TrainingArguments,Trainer
from datasets import load_dataset
import numpy as np
import evaluate
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# load dataset
dataset  = load_dataset("dair-ai/emotion", "split")

# eda
df = dataset['train'].to_pandas()
df.head()

df['text_length'] = df['text'].apply(len)


plt.figure(figsize=(8, 4))
plt.hist(df['text_length'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Text Lengths', fontsize=16)
plt.xlabel('Text Length', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

label_mapping = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

df['label_text'] = df['label'].map(label_mapping)

label_counts = df['label_text'].value_counts()

plt.figure(figsize=(8, 4))
label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Labels', fontsize=16)
plt.xlabel('Labels', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# tokenize dataset
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

def tokenize_function(examples):
    '''tokenize function'''
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# finetuning
#specify labels
model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest", 
    num_labels=6, ignore_mismatched_sizes=True)

# train hyperparameters
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# evaluate
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    '''evaluation function'''
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
)

trainer.train()

# save the model
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

# test the model
classifier = pipeline("text-classification", model="./saved_model", tokenizer=tokenizer, device=0)

texts = tokenized_datasets['test']['text']
results = classifier(texts)
print(results)

# Convert predictions to numeric labels
label_map = {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2,
    "LABEL_3": 3,
    "LABEL_4": 4,
    "LABEL_5": 5
}
predicted_labels = [label_map[res['label']] for res in results]

true_labels = tokenized_datasets['test']['label']

accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average='weighted')
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
