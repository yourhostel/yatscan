# scripts/train_tokenizer.py

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

# Ініціалізуємо порожній токенізатор з моделлю BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Використовуємо пробіли як токенізатор на рівні слів
tokenizer.pre_tokenizer = Whitespace()

# Шукаємо усі .txt файли для тренування токенізатора
files = list(Path("data").rglob("*.txt"))

# Налаштовуємо параметри тренування
trainer = BpeTrainer(
    vocab_size=512,  # достатньо для нашої абетки
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Запускаємо тренування токенізатора
tokenizer.train([str(f) for f in files], trainer)

# Зберігаємо токенізатор у директорію tokenizer
Path("tokenizer").mkdir(exist_ok=True)
tokenizer.save("tokenizer/tokenizer.json")

print("✅ Токенізатор збережено у ./tokenizer/tokenizer.json")