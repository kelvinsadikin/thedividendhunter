import streamlit as st
from tools import get_finance_agent
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from datetime import datetime


st.set_page_config(
    page_title="The Dividend Hunter",
    page_icon="🚀"
)

st.subheader("🚀 The Dividend Hunter", divider='blue')


##### ------ 🔁 Managing Chat Sessions

if "selectbox_selection" not in st.session_state:
    st.session_state['selectbox_selection'] = ["Default Chat"]

selectbox_selection = st.session_state['selectbox_selection']

if st.sidebar.button("✍️ Create New Chat", use_container_width=True):
    selectbox_selection.append(f"New Chat - {datetime.now().strftime('%H:%M:%S')}")

session_id = st.sidebar.selectbox("Chats", options=selectbox_selection, index=len(selectbox_selection)-1)



##### -------- About
st.sidebar.markdown("---")
st.sidebar.markdown("#")
st.sidebar.markdown('''
                    
ℹ️ **About**

The Dividend Hunter is a chatbot powered by LangChain agents to help investor find high dividend yield stocks in IDX and provide financial insights regarding the stocks.
                                        
🛠️ **Features**
                    
- 📊 Ideation
                    
    Question: shows a list of companies with the highest dividend yield!
                    
- 🔍 Companies Analysis
                    
    Question: provide in depth insight regarding [TICKER]!
                    
''')





### --------- 🕒 Displaying Chat History

chat_history = StreamlitChatMessageHistory(key=session_id)

for message in chat_history.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)


### --- Chat Interface
prompt = st.chat_input("Ask your question here!")
agent = get_finance_agent()

if prompt: 
    with st.chat_message("human"):
        st.markdown(prompt)
    
    with st.chat_message("ai"):
        response = agent.invoke({"input": prompt}, config={"configurable": {"session_id": session_id}})
        st.markdown(response['output'])
