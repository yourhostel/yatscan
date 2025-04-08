# scripts/make_pairs.py

import os
import csv

# Конфігурація: словник з наборами даних
DATASETS = {
    "english_philosophy": "data/english_philosophy",
    "alphabet": "data/regular_samples/alphabet",
    "roman_numerals": "data/regular_samples/roman_numerals",
    "symbols": "data/regular_samples/symbols",
}

# Підготовка вихідного CSV
with open("dataset.csv", mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["image", "text"])
    writer.writeheader()

    for name, base_path in DATASETS.items():
        images_dir = os.path.join(base_path, "images") if os.path.isdir(os.path.join(base_path, "images")) else base_path
        labels_dir = os.path.join(base_path, "labels") if os.path.isdir(os.path.join(base_path, "labels")) else base_path

        files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        for img_file in files:
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + ".txt"

            img_path = os.path.join(images_dir, img_file)
            txt_path = os.path.join(labels_dir, label_file)

            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    writer.writerow({"image": img_path, "text": text})
            else:
                print(f"[WARN] Text not found for {img_file}")

print("dataset.csv згенеровано успішно")