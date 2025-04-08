# scripts/recognize.py

from transformers import VisionEncoderDecoderModel, PreTrainedTokenizerFast
from transformers import TrOCRProcessor
from PIL import Image
import sys

# Завантажуємо модель і токенізатор з ./model
model_path = "./model"
model = VisionEncoderDecoderModel.from_pretrained(model_path)
tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{model_path}/tokenizer.json")

# ОНОВЛЮЄМО vocab_size (як під час тренування)
new_vocab_size = len(tokenizer)
model.config.vocab_size = new_vocab_size
model.decoder.config.vocab_size = new_vocab_size
model.config.decoder.vocab_size = new_vocab_size
model.decoder.resize_token_embeddings(new_vocab_size)

# Ініціалізуємо processor тільки для витягування зображення
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
feature_extractor = processor.feature_extractor

# Вхідна картинка
image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")

# Підготовка до інференсу
inputs = feature_extractor(images=image, return_tensors="pt")
generated_ids = model.generate(
    pixel_values=inputs.pixel_values,
    max_new_tokens=128,                     # дозволяємо до 128 нових токенів
    do_sample=False,                        # вимикаємо випадковість (можна лишити True, якщо треба креативу)
    num_beams=4                             # beam search для кращої якості
)

# Декодуємо результат через СВІЙ токенізатор
predicted_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n--- Результат ---\n")
print(predicted_text)
