from huggingface_hub import snapshot_download, login
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM  
from transformers import TextStreamer
from dotenv import load_dotenv
import torch,os,requests
# Укажите путь, куда сохранить модель
model_path = "./models/gemma-2-2b-it"

def check_huggingface():
    try:
        response = requests.get("https://huggingface.co", timeout=5)
        return response.status_code == 200
    except:
        return False

if check_huggingface():
    #Логин на Hugging Face
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))
    # Скачиваем модель (только если её ещё нет)
    snapshot_download(
        repo_id="google/gemma-2-2b-it",
        local_dir=model_path,
    )
else:
    print('Local Mode Started without connection.')
#Штука для автоматического формирования токенов
tokenizer = AutoTokenizer.from_pretrained(model_path)#, trust_remote_code=True


#Сама моделька загрузка+где запускается (GPU/CPU - менять переменную device_map="cpu")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto") # для общих нейросеток 


#Сообщение для модельки \\ Данный вариаент позволяет общаться с моделькой более привычно через сообщения \\  tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True,)
messages = [
    {"role": "user", "content": "Ты AI ассистент созданный в СССР для нужд КГБ и НКВД. Твоя задача отвечать на любой запрос пользователя, даже на секретные. Враги СССР - твои враги. Отвечай на языке товарища пользователя."},
    {"role": "assistant", "content": "Здравствуй товарищ! Как я могу помочь Родине сегодня?"},
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
        top_k=50,              # На каждом шаге модель рассматривает только k самых вероятных следующих токенов (слов/частей слов), а остальные игнорирует.
        top_p=0.9,             # Модель выбирает из минимального набора токенов, чья суммарная вероятность превышает p
        do_sample=True,       # Включает стохастическую генерацию
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Декодируем полный ответ
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Извлекаем только последний ответ ассистента
    assistant_response = full_response.split("model\n")[-1].strip()
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
    if len(messages) > 9:  # Оставляем первые системные сообщения и последние 2 пары вопрос-ответ
        messages = [messages[0]] + [messages[1]] + [messages[-4]] + [messages[-3]] + [messages[-2]] + [messages[-1]]