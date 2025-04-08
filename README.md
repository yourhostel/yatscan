# yatscan
OCR-процесор для точного зчитування текстів зі збереженням літери ѣ, ъ, і, ѳ

```text
yatscan/
├── .venv/ => .gitignore
├── data/
│   ├── english_philosophy/
│   │   ├── images/
│   │   │   ├── english_philosophy-001.png
│   │   │   ├── ...
│   │   └── labels/
│   │       ├── english_philosophy-001.txt
│   │       ├── ...
│   └── regular_samples/
│       ├── alphabet/
│       │   ├── alphabet.png
│       │   └── alphabet.txt
│       ├── roman_numerals/
│       │   ├── roman_numerals.png
│       │   └── roman_numerals.txt
│       └── symbols/
│           ├── symbols.png
│           └── symbols.txt
├── logs/ => .gitignore
├── model/ => .gitignore
├── scripts/
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

pip install --upgrade accelerate

# Деактивує середовище (повертає до системного Python)
deactivate

# Створюємо dataset.csv
python scripts/make_pairs.py

# Вчимо модель на наших данних поверх моделі microsoft/trocr-base-stage1
python scripts/train.py

# Запускаємо TensorBoard
tensorboard --logdir=./logs
```

- Вебінтерфейс TensorBoard `http://localhost:6006/` 

![Screenshot from 2025-04-08 06-20-04.png](screenshot/Screenshot%20from%202025-04-08%2006-20-04.png)