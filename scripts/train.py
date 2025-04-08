# scripts/train.py

from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator, PreTrainedTokenizerFast
from torch.utils.data import Dataset as TorchBaseDataset
from sklearn.model_selection import train_test_split
from datasets import Dataset
from PIL import Image
import pandas as pd
import shutil
import torch
import os

# Ініціалізація токенізатора з кастомного JSON
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/tokenizer.json")
tokenizer.pad_token = "[PAD]"
tokenizer.unk_token = "[UNK]"
tokenizer.cls_token = "[CLS]"
tokenizer.sep_token = "[SEP]"
tokenizer.mask_token = "[MASK]"

# Завантажуємо TrOCRProcessor і беремо тільки витягувач зображень
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
feature_extractor = processor.feature_extractor

# Завантажуємо модель TrOCR та підключаємо свій токенізатор
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
model.config.decoder_start_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Оновлюємо vocab_size УСЮДИ, куди тільки можна (щоб TrOCR не генерував токени поза словником)
new_vocab_size = len(tokenizer)
model.config.vocab_size = new_vocab_size
model.decoder.config.vocab_size = new_vocab_size
model.config.decoder.vocab_size = new_vocab_size
model.decoder.resize_token_embeddings(new_vocab_size)

# Завантаження CSV
df = pd.read_csv("dataset.csv")

# Розбиваємо дані на тренувальні та валідаційні (наприклад 80%/20%)
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

# 💡 Дивимось найчастіші тексти
# print(train_data["text"].value_counts().head(10))

# Створюємо HuggingFace-датасети
train_dataset = Dataset.from_pandas(train_data)
eval_dataset = Dataset.from_pandas(val_data)

# Препроцесинг
def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values[0]

    labels = tokenizer(
        example["text"],
        padding="max_length",
        max_length=128,
        truncation=True
    ).input_ids
    labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

    return {
        "pixel_values": pixel_values,
        "labels": torch.tensor(labels, dtype=torch.long)
    }

# задаєм формат — і він конвертує ці списки в тензори:
# 1. Збираємо нові дані
processed_train = [preprocess(example) for example in train_dataset]
processed_eval = [preprocess(example) for example in eval_dataset]

class TorchDataset(TorchBaseDataset):
    def __init__(self, data):
        self._data = data  # список словників {"pixel_values": ..., "labels": ...}

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

train_dataset = TorchDataset(processed_train)
eval_dataset = TorchDataset(processed_eval)

print(type(train_dataset[0]["pixel_values"]))
print(train_dataset[0]["pixel_values"].shape)

# Якщо модель вже є — видаляємо
if os.path.exists("./model"):
    shutil.rmtree("./model")

# Аргументи тренування
training_args = Seq2SeqTrainingArguments(
    output_dir="./model",                         # Каталог для збереження моделей
    per_device_train_batch_size=1,                # Розмір батчу на кожен GPU/CPU
    num_train_epochs=10,                          # Кількість епох (повних проходів по датасету)
    learning_rate=5e-5,                           # Початкова швидкість навчання
    warmup_steps=10,                              # Кількість кроків розігріву (без повного градієнта)
    weight_decay=0.01,                            # L2-регуляризація (для уникнення переобучення)
    logging_dir="./logs",                         # Каталог для TensorBoard-логів
    logging_steps=10,                             # Як часто логувати (у кроках)
    logging_first_step=True,                      # Логувати вже на першому кроці
    save_strategy="epoch",                        # Як часто зберігати модель — після кожної епохи
    save_total_limit=2,                           # Максимум збережених чекпойнтів (старі видаляються)
    report_to="tensorboard",                      # Активація логування в TensorBoard
    predict_with_generate=True,                   # Генерувати послідовності при оцінці
    generation_max_length=128,                    # Максимальна довжина згенерованого тексту
    evaluation_strategy="epoch",                  # Коли проводити оцінку — після кожної епохи
    fp16=torch.cuda.is_available()                # Увімкнути FP16, якщо є підтримка GPU
)

# Тренер — обгортка, яка запускає навчання, оцінку і збереження моделі
trainer = Seq2SeqTrainer(
    model=model,                                  # Наша модель TrOCR
    args=training_args,                           # Аргументи навчання (гіперпараметри)
    train_dataset=train_dataset,                  # Навчальний датасет
    eval_dataset=eval_dataset,                    # Валідаційний датасет — для оцінки після кожної епохи
    tokenizer=tokenizer,                          # Кастомний токенізатор (ВАЖЛИВО!)
    data_collator=default_data_collator           # Коллатор — щоб зібрати батчі (додає паддінг і т.д.)
)

# Старт тренування
trainer.train()


# ✨ Генеруємо приклади передбачення після тренування
print("\n=== ПРИКЛАД ПЕРЕДБАЧЕННЯ ===")

# Беремо один приклад із eval
sample = eval_dataset[0]
input_tensor = sample["pixel_values"].unsqueeze(0)  # [1, 3, 384, 384]

# Генеруємо передбачення
with torch.no_grad():
    generated_ids = model.generate(input_tensor.to(model.device), max_length=128)

# Декодуємо
decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Оригінальний текст
original_text = tokenizer.decode([id for id in sample["labels"] if id != -100], skip_special_tokens=True)

print(f"🖼️ Original : {original_text}")
print(f"🤖 Predicted: {decoded_text}")

# Оновлюємо vocab_size на розмір кастомного токенізатора
model.config.vocab_size = tokenizer.vocab_size

print("Token count:", len(tokenizer))
print("Model vocab size:", model.config.vocab_size)

# Збереження моделі
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")