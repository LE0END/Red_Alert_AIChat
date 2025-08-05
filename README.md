# Simple_AIChat
# Инструкция по запуску

## Подготовка окружения

0. В .env указать переменную HF_TOKEN.

1. Создайте виртуальную среду Python:
```bash
python3 -m venv .venv
 ```
Или если не работает:
```bash
python -m venv .venv
 ```
*Примечание*: В некоторых системах может потребоваться использовать *python* вместо *python3*

2. Активировать виртуальную среду: 
```bash
.venv\Scripts\activate
```

## Зависимости и запуск

3. Установить зависимости: 	
```bash
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install accelerate huggingface-hub transformers dotenv
```
4. Выполнить:
```bash
python main.py  
```
*Примечание*: В первый раз будет установлена модель, в последующие разы запуск быстрее.

## Молиться, чтобы заработало

# Описание
Этот чат предназначен для весёлого дружеского общения, может отвечать криво. # Ы