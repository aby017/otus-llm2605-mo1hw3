import os
import pandas as pd

def load_legal_docs(path='RuLegalNER/data/'):
    """Helper to load sample documents from CSV files using pandas."""
    docs = []
    if os.path.exists(path):
        for file in os.listdir(path):
            if file.endswith('.csv'):
                file_path = os.path.join(path, file)
                # Read CSV and take text from the first column
                df = pd.read_csv(file_path)
                if not df.empty:
                    # Extract text from the first column
                    texts = df.iloc[:, 0].dropna().astype(str).tolist()
                    for i, text in enumerate(texts):
                        docs.append({'filename': f"{file}_row_{i}", 'text': text})
    return docs

def chunk_text(text, max_chars=4000):
    """Splits text into chunks of maximum size."""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def prepare_documents(docs, max_chars=4000):
    """Prepares documents by checking length and chunking if necessary."""
    prepared_docs = []
    for doc in docs:
        text = doc['text']
        if len(text) > max_chars:
            chunks = chunk_text(text, max_chars)
            for i, chunk in enumerate(chunks):
                prepared_docs.append({
                    'filename': f"{doc['filename']}_part_{i}",
                    'text': chunk,
                    'original_length': len(text)
                })
        else:
            prepared_docs.append({
                'filename': doc['filename'],
                'text': text,
                'original_length': len(text)
            })
    return prepared_docs

from yandex_gpt import YandexGPT, YandexGPTConfigManagerForAPIKey
from config.variables import YC_API_KEY, YC_CATALOG_ID

CONFIG_MANAGER = YandexGPTConfigManagerForAPIKey(
    model_type="yandexgpt-lite",
    catalog_id=YC_CATALOG_ID, 
    api_key=YC_API_KEY
)

GPT = YandexGPT(config_manager=CONFIG_MANAGER)

SYSTEM_PROMPT = """
    Ты **точный** и **строгий** помощник юриста. Извлекай факты из текста без искажений. Если в тексте нет требуемых сведений, то верни NULL.
"""

USER_PROMPT_TEMPLATE = """
Извлеки из текста следующее:
    - companies:
        - наименования юридических лиц
        - ИНН, КПП юридических лиц
    - dates:
        - дата подписания
        - сроки действия
    - amounts:
        - суммы с указанием валюты, например "150 млн долларов"
    - commitment_period:
        - сроки обязательств в днях/месяцах или конкретных датах

Дай ответ **строго** и **только** в формате JSON-объекта, без markdown-разметки и без сопроводительного текста.

Пример формата ответа:
{{
"companies": [""ПАО Хорошая компания" ИНН 7727563778 КПП 770701001", "ООО Ромашка"],
"dates": "12 марта 2026",
"amounts": "50 млн рублей",
"commitment_period": "в течение десяти суток"
}}

Текст:
{text}
"""

def build_messages(text: str) -> list[dict[str, str]]:
    """
    Builds a list of messages for the GPT model based on the provided text.
    """
    return [
        {"role": "system", "text": SYSTEM_PROMPT},
        {"role": "user", "text": USER_PROMPT_TEMPLATE.format(text=text)},
    ]

def clean_json_response(raw_text: str) -> str:
    """
    Cleans the raw text response from the GPT model to ensure it is a valid JSON string.
    Removes any leading/trailing whitespace and markdown formatting.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip().replace("'", '"')

def extract_entities_from_text(text: str) -> str:
    """
    Extracts named entities from the provided text using the GPT model.
    Returns the response as a JSON string.
    """
    messages = build_messages(text)

    response = GPT.get_sync_completion(
        messages=messages,
        temperature=0.2,
        max_tokens=1000,    
    )
    return response

def save_response_to_file(response: str, filename: str) -> None:
    """
    Saves the GPT response to a specified file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response)

if __name__ == "__main__":
    documents = load_legal_docs()
    print(f"Loaded {len(documents)} documents.")
    if documents:
        print(f"Sample length: {len(documents[-1]['text'])} characters")

    """### 2. Text Preparation and Chunking
    Since LLMs have context window limits, we implement a function to split long legal documents into smaller, manageable chunks.
    """

    # Apply preparation
    processed_documents = prepare_documents(documents, 32000)
    print(f'Total chunks/documents after processing: {len(processed_documents)}')
    if processed_documents:
        print(f"Average chunk size: {sum(len(d['text']) for d in processed_documents)/len(processed_documents):.0f} characters")

    import json    
    for i, new_text in enumerate(processed_documents):
        print(f"Processing text: {new_text}")
        
        response = extract_entities_from_text(new_text)
        cleaned_response = clean_json_response(response)
        cleaned_response_dict = json.loads(cleaned_response)
        
        print(json.dumps(cleaned_response_dict, ensure_ascii=False, indent=2))

        cleaned_response_dict["original"] = new_text
        save_response_to_file(json.dumps(cleaned_response_dict, ensure_ascii=False, indent=2), os.path.join("results", f"ner_{i}.json"))