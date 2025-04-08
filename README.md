# yatscan
### Yatscan — OCR-процесор, натренований для розпізнавання стародруків із дореформеною кирилицею (ѣ, ъ, і, ѳ тощо).
#### Скрипт реалізує цикл обробки: навчання моделі оптичного розпізнавання тексту (OCR) на кастомному датасеті, сформованому зі скріншотів сторінок книжок із дореформеною російською орфографією (стародруків), а також подальше використання цієї моделі для розпізнавання тексту на аналогічних зображеннях.
```text
yatscan/
├── .venv/ => .gitignore
├── data/
│   ├── english_philosophy/
│   │   ├── images/
│   │   │   ├── english_philosophy-001.png
│   │   │   ├── ...
│   │   │   └── english_philosophy-<n>.png
│   │   └── labels/
│   │       ├── english_philosophy-001.txt
│   │       ├── ...
│   │       └── english_philosophy-<n>.txt
│   └── source/
│       └── test_image.jpg
├── logs/ => .gitignore
├── model/ => .gitignore
├── scripts/
│   ├── make_pairs.py
│   ├── recognize.py
│   ├── train.py
│   └── train_tokenizer.py
├── tokenizer/  .gitignore # генерується train_tokenizer.py
│   └── tokenizer.json
├── .gitignore
├── dataset.csv => .gitignore # генерується make_pairs.py
├── LICENSE
├── README.md
└── requirements.txt
```

```bash
# Створює нове ізольоване середовище Python у папці .venv
python3 -m venv .venv

# Активує середовище (всі пакети тепер будуть встановлюватися лише всередину .venv)
source .venv/bin/activate

# Встановлює всі залежності з файлу requirements.txt в активоване середовище
pip install -r requirements.txt

# Деактивує середовище (повертає до системного Python)
deactivate

# Створюємо dataset.csv
python scripts/make_pairs.py

# Вчимо модель на наших данних поверх моделі microsoft/trocr-base-stage1
python scripts/train.py

# Запускаємо TensorBoard
tensorboard --logdir=./logs

# Розпізнаємо нове зображення за допомогою натренованої моделі
python scripts/recognize.py data/source/test_image.jpg
```

- Вебінтерфейс TensorBoard `http://localhost:6006/` 

![Screenshot from 2025-04-08 06-20-04.png](screenshots/Screenshot%20from%202025-04-08%2006-20-04.png)