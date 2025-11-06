import json
import requests
from datetime import datetime
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from typing import List

SECTORS_API_KEY = st.secrets["SECTORS_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


def retrieve_from_endpoint(url: str) -> dict:
    """
    A robust, reusable helper function to perform GET requests.
    """
    
    headers = {"Authorization": SECTORS_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()

        return data

    except requests.exceptions.HTTPError as err:
        return {
            "error": f"HTTPError {err.response.status_code} - {err.response.reason}",
            "url": url,
            "detail": err.response.text
        }
    
    except Exception as e:
        return {
            "error": f"Unexpected error: {type(e).__name__} - {str(e)}",
            "url": url
        }

@tool
def get_top_dividend(year: str) -> dict:
    """
    Tool untuk mengambil perusahaan dengan dividend yield tertinggi. 
    Dividend yield yg diambil perlu dikonversikan ke dalam persentase, contoh: 0.5 artinya 50%.
    @param year: Tahun untuk mengambil data dividend yield, selalu isi dengan tahun terakhir.
    """

    url = f"https://api.sectors.app/v1/companies/top/?classifications=dividend_yield&n_stock=10&year={year}&include_none=false"
    
    return retrieve_from_endpoint(url)

@tool
def get_company_overview(ticker: str) -> dict:
    """
    Tool untuk memberikan overview perusahaan.
    Overview perusahaan meliputi informasi umum, ringkasan bisnis, dan data keuangan utama.
    """

    url = f"https://api.sectors.app/v1/company/report/{ticker}/?sections=overview"
    
    return retrieve_from_endpoint(url)

@tool
def get_company_financial(ticker: str) -> dict:
    """
    Tool untuk memberikan data finansial perusahaan.
    Perlu menunjukan tren keuangan terutama revenue, earnings dan free cash flow.
    Tunjukan juga growth atau stabilitas dari data keuangan tersebut.
    """

    url = f"https://api.sectors.app/v1/company/report/{ticker}/?sections=financials"
    
    return retrieve_from_endpoint(url)

@tool
def get_company_dividend(ticker: str) -> dict:
    """
    Tool untuk memberikan data dividend perusahaan.
    Tunjukan stabilitas atau growth dividend payout dari perusahaan tersebut.
    Analisa juga payout ratio jika memungkinkan.
    """

    url = f"https://api.sectors.app/v1/company/report/{ticker}/?sections=dividend"
    
    return retrieve_from_endpoint(url)

def get_finance_agent():

    # Defined Tools
    tools = [
        get_top_dividend,
        get_company_overview,
        get_company_financial,
        get_company_dividend,
    ]

    # Create the Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                Kamu adalah asisten cerdas yang menjawab pertanyaan terkait dividend saham secara faktual.
                Kalau orang bertanya ingin mendapatkan ide saham dengan dividend yield tinggi, 
                berikan list saham tersebut beserta ticker dan dividend yieldnya menggunakan get_top_dividend tool.
                Gunakan tahun terakhir sebagai parameter. Hari ini adalah {datetime.today().strftime("%Y-%m-%d")}
                Kalau orang bertanya tentang overview, financial, atau dividend suatu perusahaan,
                gunakan tool get_company_overview, get_company_financial, atau get_company_dividend yang relevan.
                Berikan jawaban yang lengkap dan ringkas berdasarkan data yang didapat dari tools.
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Initializing the LLM
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
    )

    # Create the Agent and AgentExecutor
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Add Memory to the AgentExecutor
    def get_session_history(session_id: str):

        return StreamlitChatMessageHistory(key=session_id)
    
    agent_with_memory = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_memory
