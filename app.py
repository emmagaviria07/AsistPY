import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Anthropic
from langchain.callbacks import get_openai_callback
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor, Tool
from langchain.agents import initialize_agent
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px

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

def process_agent_output(output: str) -> str:
    """Procesa la salida del agente para extraer solo la respuesta final"""
    if 'Respuesta:' in output:
        return output.split('Respuesta:')[-1].strip()
    elif 'Final Answer:' in output:
        return output.split('Final Answer:')[-1].strip()
    else:
        return output.split('\n')[-1].strip()

def custom_prompt(question):
    return f"""
    Por favor analiza los datos respondiendo a esta pregunta: {question}
    
    Sigue estas instrucciones:
    1. Analiza los datos necesarios
    2. Proporciona una respuesta directa y clara en español
    3. No incluyas el proceso de pensamiento ni el código en la respuesta final
    4. Si el resultado incluye números, formatea los decimales a máximo 2 lugares
    5. Estructura la respuesta en forma de lista si hay múltiples puntos
    6. Usa un lenguaje formal pero fácil de entender
    
    Ejemplo de formato deseado:
    "El promedio de edad es 34.5 años."
    o
    "Los principales resultados son:
    - Valor máximo: 100
    - Valor mínimo: 20
    - Promedio: 45.5"
    """

if user_question and ke and uploaded_file is not None:
    try:
        with st.spinner('Analizando los datos...'):
            # Configurar el LLM
            llm = Anthropic(
                model="claude-2",
                temperature=0,
                max_tokens=1500,
                anthropic_api_key=ke
            )
            
            # Crear el agente
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=False,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                allow_dangerous_code=True
            )
            
            # Configurar el ejecutor del agente
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=agent.tools,
                handle_parsing_errors=True,
                max_iterations=3,
                early_stopping_method="generate"
            )
            
            # Ejecutar la consulta
            response = agent_executor.run(custom_prompt(user_question))
            
            # Procesar y mostrar la respuesta
            clean_response = process_agent_output(response)
            
            # Mostrar resultados
            st.success("Análisis completado")
            
            # Si la respuesta contiene múltiples puntos, mostrarlos como lista
            if '-' in clean_response:
                st.write("### Resultados:")
                points = [point.strip() for point in clean_response.split('-') if point.strip()]
                for point in points:
                    st.write(f"• {point}")
            else:
                st.write("### Resultado:")
                st.write(clean_response)
            
            # Si hay números en la respuesta, intentar mostrarlos como métricas
            try:
                import re
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", clean_response)
                if numbers and len(numbers) == 1:
                    st.metric("Valor numérico encontrado", float(numbers[0]))
                elif numbers and len(numbers) > 1:
                    cols = st.columns(min(len(numbers), 3))
                    for i, num in enumerate(numbers[:3]):
                        cols[i].metric(f"Valor {i+1}", float(num))
            except:
                pass
            
    except Exception as e:
        st.error(f"Ocurrió un error al analizar los datos. Por favor, intenta reformular tu pregunta.")
        st.error(f"Detalle del error: {str(e)}")
