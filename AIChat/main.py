from huggingface_hub import snapshot_download, login
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM  
from transformers import TextStreamer
from dotenv import load_dotenv
import torch
import os
# Укажите путь, куда сохранить модель
model_path = "./models/gemma-2-2b-it"
#Логин на Hugging Face
load_dotenv()
login(token=os.getenv("HF_TOKEN"))
# Скачиваем модель (только если её ещё нет)
snapshot_download(
    repo_id="google/gemma-2-2b-it",
    local_dir=model_path,
)

#Штука для автоматического формирования токенов
tokenizer = AutoTokenizer.from_pretrained(model_path)#, trust_remote_code=True


#Сама моделька загрузка+где запускается (GPU/CPU - менять переменную device_map="cpu")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto") # для общих нейросеток 


#Сообщение для модельки \\ Данный вариаент позволяет общаться с моделькой более привычно через сообщения \\  tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True,)
messages = [
    {"role": "user", "content": "Ты весёлый AI ассистент, тебя считают другом и помошником. Отвечай на вопросы с неболишим количеством юмора и приколов. Отвечай на языке пользователя."},
    {"role": "assistant", "content": "Привет! Как я могу помочь?"},
]

def add_to_history(role, content):
    """Добавляет сообщение в историю"""
    messages.append({"role": role, "content": content})
def generate_response(new_message, max_new_tokens=256):
    add_to_history("user", new_message)
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt",return_dict=True, add_generation_prompt=True).to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs = model.generate(
        **inputs,
        streamer = streamer,
        max_new_tokens=max_new_tokens,
        temperature=0.6,      # Больше случайности (0.1–1.0)
        do_sample=True,       # Включает стохастическую генерацию
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Декодируем полный ответ
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Извлекаем только последний ответ ассистента
    assistant_response = full_response.split("assistant:")[-1].strip()
    # Добавляем ответ ассистента в историю
    add_to_history("assistant", assistant_response)
    
    
print("Чат с AI начат. Введите 'exit' для выхода.")
while True:
    user_input = input("Your message: ")
    if user_input.lower() in ['exit', 'quit', 'выход']:
        break
    print(f"AI anwser: ")
    generate_response(user_input)
 
    # Ограничиваем длину истории, чтобы не переполнить память
    if len(messages) > 10:  # Оставляем первые системные сообщения и последние 4 пары вопрос-ответ
        messages = [messages[0]] + [messages[1]] + messages[-8:]
