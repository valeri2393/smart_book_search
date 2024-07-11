import streamlit as st
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss

model_name = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

df = pd.read_csv('csv_file.csv')

MAX_LEN = 300

def embed_bert_cls(text, model=model, tokenizer=tokenizer):
    t = tokenizer(text,
                  padding=True,
                  truncation=True,
                  return_tensors='pt',
                  max_length=MAX_LEN)
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().squeeze()

embeddings = np.loadtxt('embeddings.txt')
embeddings_tensor = [torch.tensor(embedding) for embedding in embeddings]

# Создание индекса Faiss
embeddings_matrix = np.stack(embeddings)
index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
index.add(embeddings_matrix)

st.title('Приложение для рекомендации книг')

text = st.text_input('Введите запрос:')
num_results = st.number_input('Введите количество рекомендаций:', min_value=1, max_value=50, value=3)


# Add a button to trigger the recommendation process
recommend_button = st.button('Получить рекомендации')

if text and recommend_button:  # Check if the user entered text and clicked the button

    # Встраивание запроса и поиск ближайших векторов с использованием Faiss
    query_embedding = embed_bert_cls(text)
    query_embedding = query_embedding.numpy().astype('float32')
    _, indices = index.search(np.expand_dims(query_embedding, axis=0), num_results)

    st.subheader('Топ рекомендуемых книг:')
    for i in indices[0]:
        recommended_embedding = embeddings_tensor[i].numpy()  # Вектор рекомендованной книги
        similarity = np.dot(query_embedding, recommended_embedding)  # Косинусное сходство
        similarity_percent = similarity * 100
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image(df['image'][i], use_column_width=True)
        
        with col2:
            st.write(f"**Название книги:** {df['title'][i]}")
            st.write(f"**Автор:** {df['author'][i]}")
            st.write(f"**Описание:** {df['annotation'][i]}")
            st.write(f"**Оценка сходства:** {similarity_percent:.2f}%")
        
        st.write("---")