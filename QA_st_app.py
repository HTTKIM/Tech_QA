import streamlit as st
from QA_RAG import *
from streamlit_option_menu import option_menu
import pandas as pd

st.set_page_config(page_title="KFPA QA Search", page_icon="logo_pageicon1.png")
st.title(":book: 기술자료 기반 검색 Q&A System")
st.divider()

## Sidebar 설정 ##

# 위키 링크를 구성을 위한 분류파일 읽어오기
file_path = 'menu_list(241028).xlsx'
menu_df = pd.read_excel(file_path)

# Wiki page 바로가기 설정을 위한 메뉴 읽어오기
menu_data = {}
for _, row in menu_df.iterrows():
    main_menu = row['main_list']
    submenu = row['sub_list']
    link = row['url']
    if main_menu not in menu_data:
        menu_data[main_menu] = {}
    menu_data[main_menu][submenu] = link

# Sidebar 배경색깔 지정 - Bridge color code(#08487d) 적용
# Sidebar 의 header 및 markdown 글자 색깔 & expander 배경색 지정
st.markdown(
    """
    <style>
    /* Sidebar background and text color */
    [data-testid="stSidebar"] {background-color: #08487d;}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {color: white;}

    /* Expander title color */
    [data-testid="stExpander"] button div[role="button"] {color: black;}
    
    /* Expander background color */
    [data-testid="stExpander"] {background-color: white;}

    /* Expander content area background color */
    [data-testid="stExpander"] > div {background-color: white;}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar 기능 설정
with st.sidebar:   
    st.image('bridge.PNG')
    st.header("About")
    st.markdown("Bridge 지식위키에 게시된 법령과 기준 기반으로 답변합니다.")
    st.write("")
    st.header("Wiki page 바로가기")
    app = option_menu(
        menu_title='Select Link',
        options=list(menu_data.keys()),
        icons=[''],
        menu_icon='list-task',
        default_index=0,
        styles={
            "container": {"padding": "5!important","background-color":'white'},
            "icon": {"color": "black", "font-size": "15px"}, 
            "nav-link": {"color":"black","font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#cfe0de"},
            "nav-link-selected": {"background-color": "#e6f9f7"},}                
        )

# expander 설정 - expanded = True 설정해서 메뉴 클릭 시 확장된 상태
    with st.expander(f"{app} 관련 기준 및 법령", expanded = True):
        for submenu, link in menu_data[app].items():
            st.markdown(f"[{submenu}]({link})")

    st.image('KV.png')

# 참조 문서 표시 함수 정의
def display_retrieved_documents(documents):
    with st.expander("참조 문서 확인"):
        for idx, doc in enumerate(documents[:5]):
            source = doc.metadata.get('source', 'N/A')
            url = doc.metadata.get('url', '#')  # URL이 없을 경우 기본 링크 처리
            st.markdown(f"**Document {idx+1}**")
            st.write(f"Source: {source}")
            st.markdown(f'<a href="{url}" target="_blank">Link</a>', unsafe_allow_html=True)

############# 채팅 메시지 구성 #############

# 초기 화면 구성 - 메시지가 없을때는 아래 메시지 표시
if ('messages' not in st.session_state):
    st.session_state.messages = [{'role':'ai', 'content': '안녕하세요?  아래 메시지창에 질문을 입력해 주세요!'}]
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 입력창 설정
prompt = st.chat_input("Enter the Question!")

# 질문 입력 시 처리
if (prompt):
    st.session_state.messages.append({'role':'user', 'content': prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Processing"):
          with st.chat_message('assistant'):
            respond = st.write_stream(inference(prompt))

# AI 답변을 채팅으로 붙이기    
    st.session_state.messages.append({'role':'ai', 'content': respond})

# 참조문서 표시
    documents = retrieve_documents(prompt)
    display_retrieved_documents(documents)  
