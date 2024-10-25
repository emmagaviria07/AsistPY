import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Anthropic
from langchain.callbacks import get_openai_callback
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import numpy as np
from PIL import Image

# Configuración de la página de Streamlit
st.title('Analítica de datos con Agentes 🤖🔍')
image = Image.open('data_analisis.png')
st.image(image, width=350)

with st.sidebar:
    st.subheader("Este Agente de Pandas con Claude te ayudará a realizar análisis sobre tus datos")

# Input para la API key de Anthropic
ke = st.text_input('Ingresa tu Clave de Anthropic')
os.environ['ANTHROPIC_API_KEY'] = ke

# Carga de archivo
uploaded_file = st.file_uploader('Elige el archivo csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, on_bad_lines='skip')
    st.write(df)

st.subheader('Te ayudaré a analizar los datos que cargues.')
user_question = st.text_input("¿Qué deseas saber de los datos?:")

if user_question and ke:
    try:
        # Modificar la pregunta para incluir instrucciones específicas
        enhanced_question = f"{user_question}, busca primero siempre la correspondencia entre las columnas y la información que te pida"
        
        # Crear el agente con Claude
        agent = create_pandas_dataframe_agent(
            Anthropic(
                model="claude-3-sonnet-20240229",  # Usar el modelo Claude más reciente
                anthropic_api_key=ke,
                temperature=0,
                max_tokens_to_sample=1500
            ),
            df,
            allow_dangerous_code=True,
            verbose=True
        )
        
        # Ejecutar la consulta
        response = agent.run(enhanced_question)
        st.write(response)
        
    except Exception as e:
        st.error(f"Ocurrió un error: {str(e)}")
