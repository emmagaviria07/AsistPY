import os
#from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
#from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
import streamlit as st
import json
import pandas as pd
import numpy as np
from PIL import Image

st.title('Analítica de datos con Agentes 🤖🔍')
image = Image.open('data_analisis.png')
st.image(image,width=350)

with st.sidebar:
   st.subheader("Este Agente de Pandas, te ayudará a realizar algo de análisis sobre tus datos")

ke = st.text_input('Ingresa tu Clave')
os.environ['OPENAI_API_KEY'] = ke


uploaded_file = st.file_uploader('Elige el archivo csv')
if uploaded_file is not None:
   df=pd.read_csv(uploaded_file, on_bad_lines='skip') 
   st.write(df)

st.subheader('Te ayduaré a analizar los datos que cargues.')

user_question = st.text_input("Que desesas saber de los datos?:")
if user_question :
      prompt_aux=st.text_area( " ")
      prompt = f"""
      You are a highly knowledgeable scientific data frames analysis expert. The data is about electrical energy consumption 
      and demand. 
      
      
      Instructions:
      - Your task is to examine the following dataframe in detail.
      - {user_question}
      - Provide a comprehensive, factual, and scientifically accurate explanation of what the data depicts
      - If applicable, include any relevant scientific terminology to enhance the explanation
      - Provide a comprehensive, factual, and scientifically accurate explanation of what the image depicts
      - Highlight key elements and their significance, and present your analysis in clear, well-structured markdown format
      - Write when occurs the major and minor consumption, date and hour when this be possible
      - Explain always in spanish.
      
      """
      
      response = client.chat.completions.create(
          model="o1-mini",
          messages=[
              {
                  "role": "user",
                  "content": [
                      {
                          "type": "text",
                          "text": prompt
                      },
                  ],
              }
          ]
      )
      
      st.write(response.choices[0].message.content)
