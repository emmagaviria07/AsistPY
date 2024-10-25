import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Anthropic
from langchain.callbacks import get_openai_callback
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor
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

def format_response_for_streamlit(df, response, question):
    """Formatea la respuesta para mostrarla en Streamlit de manera apropiada"""
    try:
        # Si la respuesta contiene código Python, intenta ejecutarlo
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
            exec(code, globals(), locals())
            
            # Si el código generó algún DataFrame, mostrarlo
            for var in locals():
                if isinstance(locals()[var], pd.DataFrame):
                    st.write("Resultados encontrados:")
                    st.dataframe(locals()[var])
            
            # Si hay análisis numérico, mostrar métricas
            if any(word in question.lower() for word in ['promedio', 'media', 'máximo', 'mínimo', 'suma', 'total']):
                col1, col2, col3 = st.columns(3)
                if 'result' in locals():
                    col1.metric("Resultado", locals()['result'])
        
        # Mostrar texto explicativo
        st.write("Análisis:")
        st.write(response.replace("```python", "").replace("```", ""))
        
        # Si la pregunta sugiere una visualización, intentar crearla
        if any(word in question.lower() for word in ['gráfico', 'gráfica', 'visualiza', 'muestra', 'plot']):
            if 'result' in locals() and isinstance(locals()['result'], pd.Series):
                fig = px.bar(locals()['result'])
                st.plotly_chart(fig)
            
    except Exception as e:
        st.write(response)  # Si algo falla, mostrar la respuesta original
        
def custom_prompt(question):
    return f"""
    Analiza los datos según esta pregunta: {question}
    
    Instrucciones:
    1. Proporciona una respuesta clara y concisa
    2. Si es relevante, incluye código Python que genere resultados visualizables
    3. Si es apropiado, genera visualizaciones usando Streamlit (st.line_chart, st.bar_chart, etc.)
    4. Usa st.write() para mostrar resultados
    5. Para valores numéricos importantes, usa st.metric()
    6. Si generas un DataFrame, usa st.dataframe()
    
    Por favor, estructura tu respuesta de manera que sea fácil de leer en Streamlit.
    """

if user_question and ke:
    try:
        # Crear el agente con Claude
        llm = Anthropic(
            model="claude-2",
            anthropic_api_key=ke,
            temperature=0,
            max_tokens=1500
        )
        
        # Crear el agente con manejo de errores
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            allow_dangerous_code=True,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=False,
        )
        
        # Crear el ejecutor del agente
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=agent.tools,
            handle_parsing_errors=True,
            max_iterations=3
        )
        
        # Ejecutar la consulta con el prompt personalizado
        with st.spinner('Analizando los datos...'):
            response = agent_executor.run(custom_prompt(user_question))
            format_response_for_streamlit(df, response, user_question)
        
    except Exception as e:
        st.error(f"Ocurrió un error: {str(e)}")
        if hasattr(e, 'response'):
            st.error(f"Detalles adicionales: {e.response}")
