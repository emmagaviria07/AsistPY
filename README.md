# AsistPY
# Agente de Análisis de Datos con LangChain y OpenAI 🤖📊

## Descripción
Esta aplicación web, desarrollada con Streamlit, permite realizar análisis de datos interactivo utilizando un agente de IA basado en LangChain y OpenAI. El agente puede interpretar y analizar datos de archivos CSV respondiendo preguntas en lenguaje natural.

## Características Principales
- 📤 Carga de archivos CSV
- 💬 Interfaz conversacional para análisis de datos
- 📊 Visualización de datasets
- 🤖 Análisis automatizado mediante IA
- 🌐 Soporte multilenguaje (respuestas en español)

## Requisitos Previos
```
- Python 3.7+
- OpenAI API Key
- streamlit
- langchain
- pandas
- numpy
- Pillow
- openai
- faiss-cpu
```

## Instalación
1. Clona el repositorio
2. Instala las dependencias:
```bash
pip install streamlit langchain openai pandas numpy Pillow faiss-cpu
```

## Configuración
1. Obtén una API Key de OpenAI
2. Al ejecutar la aplicación, ingresa tu API Key en el campo correspondiente

## Uso
1. Ejecuta la aplicación:
```bash
streamlit run app.py
```

2. Interfaz de Usuario:
   - Panel lateral con instrucciones
   - Campo para ingresar API Key de OpenAI
   - Uploader para archivos CSV
   - Campo de texto para ingresar preguntas sobre los datos

3. Flujo de trabajo:
   - Carga tu archivo CSV
   - Visualiza los datos en pantalla
   - Realiza preguntas sobre el dataset
   - Obtén análisis detallados en español

## Estructura del Código

### Importaciones Principales
```python
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from openai import OpenAI
```

### Componentes Clave
1. **Interfaz de Usuario**
   - Título y banner
   - Sidebar con instrucciones
   - Campos de entrada para API Key y preguntas

2. **Procesamiento de Datos**
   - Carga de archivos CSV
   - Visualización de dataframes
   - Integración con OpenAI

3. **Sistema de Análisis**
   - Prompt engineering personalizado
   - Análisis contextual de datos
   - Respuestas estructuradas en español

## Personalización del Prompt
El sistema utiliza un prompt especializado que:
- Actúa como experto en análisis científico de datos
- Proporciona explicaciones detalladas y precisas
- Incluye terminología científica relevante
- Destaca elementos clave y su significado
- Analiza patrones de consumo y temporalidad
- Genera respuestas en español

## Notas de Seguridad
- No almacena la API Key de OpenAI de forma permanente
- Procesa los datos localmente
- Utiliza conexiones seguras para la API de OpenAI

## Limitaciones
- Requiere conexión a internet
- Necesita una API Key válida de OpenAI
- Procesa únicamente archivos CSV
- Las respuestas dependen de la calidad del dataset

## Contribuciones
Las contribuciones son bienvenidas. Por favor, asegúrate de:
1. Fork del repositorio
2. Crear una rama para nuevas características
3. Realizar pull request con descripción detallada

## Licencia
[Carlos Correa]
