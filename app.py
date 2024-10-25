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
import plotly.graph_objects as go

# Configuración de la página de Streamlit
st.title('Analítica de datos con Agentes 🤖🔍')
image = Image.open('data_analisis.png')
st.image(image, width=350)

with st.sidebar:
    st.subheader("Este Agente de Pandas con Claude te ayudará a realizar análisis sobre tus datos")

# Input para la API key de Anthropic
ke = st.text_input('Ingresa tu Clave de Anthropic')
os.environ['ANTHROPIC_API_KEY'] = ke

# Funciones de visualización
def create_visualization(df, tipo, x=None, y=None, title=None):
    """Crea visualizaciones usando Plotly Express"""
    if tipo == 'hist':
        fig = px.histogram(df, x=x, title=title)
    elif tipo == 'bar':
        fig = px.bar(df, x=x, y=y, title=title)
    elif tipo == 'scatter':
        fig = px.scatter(df, x=x, y=y, title=title)
    elif tipo == 'line':
        fig = px.line(df, x=x, y=y, title=title)
    elif tipo == 'box':
        fig = px.box(df, y=y, title=title)
    return fig

def plot_data(df, plot_type, columns, title=None):
    """Función para crear y mostrar gráficos en Streamlit"""
    try:
        if plot_type == 'histogram':
            fig = create_visualization(df, 'hist', x=columns[0], title=title)
        elif plot_type == 'scatter':
            fig = create_visualization(df, 'scatter', x=columns[0], y=columns[1], title=title)
        elif plot_type == 'bar':
            fig = create_visualization(df, 'bar', x=columns[0], y=columns[1], title=title)
        elif plot_type == 'line':
            fig = create_visualization(df, 'line', x=columns[0], y=columns[1], title=title)
        elif plot_type == 'box':
            fig = create_visualization(df, 'box', y=columns[0], title=title)
        
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error al crear la visualización: {str(e)}")

# Carga de archivo
uploaded_file = st.file_uploader('Elige el archivo csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, on_bad_lines='skip')
    st.write(df)

st.subheader('Te ayudaré a analizar los datos que cargues.')
user_question = st.text_input("¿Qué deseas saber de los datos?:")

def process_agent_output(output: str) -> dict:
    """Procesa la salida del agente para extraer respuesta y comandos de visualización"""
    result = {
        'text': '',
        'plot_type': None,
        'columns': [],
        'title': None
    }
    
    # Extraer la respuesta principal
    if 'Respuesta:' in output:
        result['text'] = output.split('Respuesta:')[-1].strip()
    elif 'Final Answer:' in output:
        result['text'] = output.split('Final Answer:')[-1].strip()
    else:
        result['text'] = output.strip()
    
    # Buscar instrucciones de visualización
    if 'VISUALIZACIÓN:' in output:
        viz_info = output.split('VISUALIZACIÓN:')[-1].strip()
        try:
            parts = viz_info.split(',')
            result['plot_type'] = parts[0].strip()
            result['columns'] = [col.strip() for col in parts[1:] if col.strip()]
            result['title'] = f"Gráfico de {result['plot_type']}"
        except:
            pass
    
    return result

def custom_prompt(question):
    return f"""
    Por favor analiza los datos respondiendo a esta pregunta: {question}
    
    Sigue estas instrucciones:
    1. Analiza los datos necesarios
    2. Si la pregunta requiere visualización, incluye al final de tu respuesta una línea con el formato:
       VISUALIZACIÓN: tipo_grafico, columna1, columna2
       Donde tipo_grafico puede ser: histogram, scatter, bar, line, box
    3. Proporciona una respuesta directa y clara en español
    4. Si el resultado incluye números, formatea los decimales a máximo 2 lugares
    5. Estructura la respuesta en forma de lista si hay múltiples puntos
    
    Ejemplo de respuesta con visualización:
    "La distribución de edades muestra un promedio de 34.5 años con una concentración en el rango de 20-40 años.
    VISUALIZACIÓN: histogram, Age"
    """

if user_question and ke and uploaded_file is not None:
    try:
        with st.spinner('Analizando los datos...'):
            # Configurar el LLM y crear el agente
            agent = create_pandas_dataframe_agent( ChatAnthropic(model='claude-3-haiku-20240307'),
                #Anthropic(
                #    model="claude-2.1",
                #    temperature=0,
                #    max_tokens=1500,
                #    anthropic_api_key=ke
                #),
                df,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                allow_dangerous_code=True,  # Agregado el parámetro requerido
            )         
            
            # Configurar el ejecutor del agente
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=agent.tools,
                handle_parsing_errors=True,
                max_iterations=3,
                early_stopping_method="generate"
            )
            
            # Ejecutar la consulta y procesar la respuesta
            response = agent_executor.run(custom_prompt(user_question))
            result = process_agent_output(response)
            
            # Mostrar resultados
            st.success("Análisis completado")
            
            # Mostrar texto
            if '-' in result['text']:
                st.write("### Resultados:")
                points = [point.strip() for point in result['text'].split('-') if point.strip()]
                for point in points:
                    st.write(f"• {point}")
            else:
                st.write("### Resultado:")
                st.write(result['text'])
            
            # Crear visualización si se especificó
            if result['plot_type'] and result['columns']:
                st.write("### Visualización:")
                plot_data(df, result['plot_type'], result['columns'], result['title'])
            
    except Exception as e:
        st.error(f"Ocurrió un error al analizar los datos. Por favor, intenta reformular tu pregunta.")
        st.error(f"Detalle del error: {str(e)}")
