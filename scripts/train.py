# scripts/train.py

from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
from datasets import Dataset
from PIL import Image
import pandas as pd
import torch
import os

# Завантаження csv, генеруємо з make_pairs.py
df = pd.read_csv("dataset.csv")

# Ініціалізація препроцесора та моделі
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

# ВАЖЛИВО: задаємо стартовий токен декодера
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

# Препроцесинг однієї пари "картинка-текст"
def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]
    labels = processor.tokenizer(example["text"], padding="max_length", max_length=128, truncation=True).input_ids
    labels = [l if l != processor.tokenizer.pad_token_id else -100 for l in labels]  # маскуємо пади
    return {"pixel_values": pixel_values, "labels": labels}

# Обгортка у Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.map(preprocess)

# Аргументи тренування
training_args = Seq2SeqTrainingArguments(
    output_dir="./model",                  # Куди зберігати модель
    per_device_train_batch_size=1,         # Розмір пакета
    num_train_epochs=3,                    # Скільки разів пройтись по всьому датасету
    logging_dir="./logs",                  # Куди писати TensorBoard-логи
    logging_steps=10,                      # Логувати раз на 10 кроків
    save_strategy="epoch",                 # Зберігати після кожної епохи
    save_total_limit=2,                    # Тримати тільки останні 2 збереження
    report_to="tensorboard",               # Активувати логування у TensorBoard
    predict_with_generate=True,            # Генерація на валідації
    fp16=torch.cuda.is_available()         # Використовувати FP16 якщо GPU є
)

# Тренер
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.tokenizer,
    data_collator=default_data_collator  # додаємо, щоб не було помилки з input_ids
)

# Старт тренування
trainer.train()

# Збереження моделі
model.save_pretrained("./model")
processor.save_pretrained("./model")
