import streamlit as st
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss
from streamlit.errors import StreamlitAPIException
import urllib.parse



import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load model and tokenizer
model_name = "sentence-transformers/msmarco-distilbert-base-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load data
books = pd.read_csv('data/data_final_version.csv')

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

# Load embeddings
embeddings = np.loadtxt('embeddings.txt')
embeddings_tensor = [torch.tensor(embedding) for embedding in embeddings]

# Create Faiss index
embeddings_matrix = np.stack(embeddings)
index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
index.add(embeddings_matrix)


# CSS стили для заднего фона
background_image = """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/premium-photo/blur-image-book_9563-1100.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
"""

# Вставляем CSS стили в приложение Streamlit
st.markdown(background_image, unsafe_allow_html=True)


# Вставляем CSS стили для окошка с прозрачным фоном
transparent_title = """
    <style>
    .transparent-title {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
"""

transparent_box = """
    <style>
    .transparent-box {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
"""

# Вставляем CSS стили в приложение Streamlit
st.markdown(transparent_title, unsafe_allow_html=True)
st.markdown(transparent_box, unsafe_allow_html=True)

# Streamlit interface
st.markdown('<h1 class="transparent-title">🎓📚Приложение для рекомендаций книг📚🎓</h1>', unsafe_allow_html=True)

# Далее ваш код Streamlit
text = st.text_input('Введите ваш запрос для поиска книг:')
num_results = st.number_input('Количество результатов:', min_value=1, max_value=20, value=3)
recommend_button = st.button('Получить рекомендации')


if text and recommend_button:  # Check if the user entered text and clicked the button

    # Embed the query and search for nearest vectors using Faiss
    query_embedding = embed_bert_cls(text)
    query_embedding = query_embedding.numpy().astype('float32')
    _, indices = index.search(np.expand_dims(query_embedding, axis=0), num_results)

    st.subheader('Рекомендации по вашему запросу:')
    for i in indices[0]:
        recommended_embedding = embeddings_tensor[i].numpy()  # Vector of the recommended book
        similarity = np.dot(query_embedding, recommended_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(recommended_embedding))  # Cosine similarity
        similarity_percent = similarity * 100

        col1, col2 = st.columns([1, 3])
        with col1:
            image_url = books['image_url'][i]
            if pd.isna(image_url) or not image_url or image_url.strip() == '':
                st.write("Обложка не найдена")
            else:
                try:
                    st.image(image_url, use_column_width=True)
                except Exception as e:
                    st.write("Обложка не найдена")
                    st.write(e)

        with col2:
            # Выводим информацию о книге на прозрачном фоне
            st.markdown(f"""
                <div class="transparent-box">
                    <p><b>Название книги:</b> {books['title'][i]}</p>
                    <p><b>Автор:</b> {books['author'][i]}</p>
                    <p><b>Описание:</b>{books['annotation'][i]}")
                    <p><b>Оценка сходства:</b> {similarity_percent:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

        st.write("---")
