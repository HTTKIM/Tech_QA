import streamlit as st
import importlib
from streamlit_option_menu import option_menu
import pandas as pd

# 페이지 설정
st.set_page_config(page_title="KFPA QA Search", page_icon="logo_pageicon1.png")

st.markdown(
    """
    <style>   
    /* 메인 헤더 스타일 */
    .main-header {
        font-size: 28px;  /* 글씨 크기 */
        font-weight: bold;  /* 굵게 */
        color: #333333;  /* 텍스트 색상 */
        display: flex;  /* 아이콘과 텍스트 정렬 */
        align-items: center;  /* 수직 중앙 정렬 */
        gap: 10px;  /* 텍스트와 아이콘 사이 간격 */
        margin-bottom: 20px;  /* 아래 여백 */
    }

    /* 목록 아이템 앞 빈 삼각형 스타일 */
    .triangle-list {
        font-size: 18px;  /* 글씨 크기 */
        color: #555555;  /* 텍스트 색상 */
        margin-left: 20px;  /* 왼쪽 여백 */
        line-height: 1.6;  /* 줄 간격 */
        display: flex; /* 삼각형과 텍스트 정렬 */
        align-items: center;
    }

    /* 목록 스타일 */
    .custom-list {
        font-size: 18px;  /* 글씨 크기 */
        color: #333333;  /* 텍스트 색상 */
        margin-bottom: 15px;  /* 항목 간 간격 */
        text-decoration: underline; /* 밑줄 추가 */
    }

    /* 목록 스타일 */
    .custom-sub-list {
        font-size: 16px;  /* 글씨 크기 */
        color: #333333;  /* 텍스트 색상 */
        margin-bottom: 15px;  /* 항목 간 간격 */
        text-indent: 22px;  /* 글자 앞 공백 추가 */
    }

    /* 빈 삼각형 추가 */
    .custom-list::before {
        content: ""; /* 빈 내용 */
        display: inline-block;
        width: 0;
        height: 0;
        border-top: 6px solid transparent; /* 위쪽 투명 테두리 */
        border-bottom: 6px solid transparent; /* 아래쪽 투명 테두리 */
        border-left: 10px solid black; /* 검은색 테두리 */
        background-color: transparent; /* 삼각형 내부를 투명하게 */
        margin-right: 10px; /* 텍스트와의 간격 */
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-header">
         점검 관련 법령 지식위키 검색<span>📖</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="custom-list">좌측 사이드바에서 DATA를 선택해 주세요</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-list">[키워드] 검색 기반입니다. 법령이나 기준에 있는 단어 검색 용도로 사용하세요</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-sub-list">예시) 방화구획 관련된 내용 모두다 알려줘(X) -> 방화구획 설치대상 건물은?(O)  ', unsafe_allow_html=True)

################ Sidebar 설정 ################

# 위키 링크를 구성을 위한 분류파일 읽어오기
file_path = 'menu_list(241028).xlsx'
menu_df = pd.read_excel(file_path)

# 메뉴 데이터 구성
menu_data = {}
for _, row in menu_df.iterrows():
    main_menu = row['main_list']
    submenu = row['sub_list']
    link = row['url']
    if main_menu not in menu_data:
        menu_data[main_menu] = {}
    menu_data[main_menu][submenu] = link

# DB 목록
db_list = list(menu_data.keys())

# 세션 상태에 'database'가 없으면 초기값 설정
if 'database' not in st.session_state:
    st.session_state['database'] = db_list[0] if db_list else None

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

# 사이드바 설정
with st.sidebar:   
    st.image('bridge.PNG')
    st.header("About")
    st.markdown("Bridge 지식위키에 게시된 법령과 기준 기반으로 답변합니다.")
    st.write("")
    st.header("DATA 선택")

    # 사이드바의 option_menu -> session_state['database'] 동기화
    selected_db_index = db_list.index(st.session_state['database']) if st.session_state['database'] in db_list else 0
    sidebar_selection = option_menu(
        menu_title='DATABASE LIST',
        options=list(menu_data.keys()),
        icons=['']*len(db_list),
        menu_icon='list-task',
        default_index=selected_db_index,
        styles={
            "container": {"padding": "5!important", "background-color": 'white'},
            "icon": {"color": "black", "font-size": "15px"}, 
            "nav-link": {"color":"black","font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#cfe0de"},
            "nav-link-selected": {"background-color": "#e6f9f7"},
        },
        key='sidebar_menu'  # 키 설정
    )
    st.session_state['database'] = sidebar_selection  # 최종적으로 session_state에 저장

    # 메뉴 항목 관련 법령 및 링크 표시
    with st.expander(f"{st.session_state['database']} 관련 기준 및 법령", expanded=True):
        for submenu, link in menu_data[st.session_state['database']].items():
            st.markdown(f"[{submenu}]({link})")

    st.image('KV.png')

######## 메인화면에서 Expander로 DB 선택 ########
with st.expander("DATABASE 선택 (메인화면)", expanded=False):
    db_main_index = db_list.index(st.session_state['database']) if st.session_state['database'] in db_list else 0
    main_selection = st.selectbox("Select DB", db_list, index=db_main_index, key='main_db_select')
    st.session_state['database'] = main_selection
    
    st.write(f"현재 선택된 DATABASE: **{st.session_state['database']}**")

st.divider()

# 동적으로 QA 모듈 임포트 및 연결
try:
    qa_module = importlib.import_module(f"QA_RAG_{app}")
except ModuleNotFoundError:
    st.stop()

# 참조 문서 표시 함수 정의
def display_retrieved_documents(documents):
    with st.expander("참조 문서 확인"):
        for idx, doc in enumerate(documents[:3]):
            source = doc.metadata.get('source', 'N/A')
            url = doc.metadata.get('url', '#')  # URL이 없을 경우 기본 링크 처리
            st.markdown(f"**Document {idx+1}**")
            st.write(f"Source: {source}")
            st.markdown(f'<a href="{url}" target="_blank">Link</a>', unsafe_allow_html=True)

# 초기 메시지 구성
if 'messages' not in st.session_state:
    st.session_state.messages = [{'role':'ai', 'content': '안녕하세요?  아래 메시지창에 질문을 입력해 주세요!'}]

# 기존 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 입력창
prompt = st.chat_input("Enter the Question!")

if prompt:
    # 사용자가 입력한 질문 메시지 기록
    st.session_state.messages.append({'role':'user', 'content': prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 스피너 표시
    with st.spinner("Processing"):
        # QA 모듈의 스트리밍 제너레이터 획득
        response_generator = qa_module.inference(prompt)

        # 부분 응답(Partial Text)을 저장할 변수
        partial_text = ""
        
        # AI 메시지 출력 영역(채팅 형식)
        with st.chat_message("ai"):
            # 매번 새 토큰을 업데이트하기 위해 빈 placeholder 생성
            placeholder = st.empty()

            # Generator 순회
            for token in response_generator:
                partial_text += token  # 새 토큰을 누적
                placeholder.markdown(partial_text)  # 스트리밍 텍스트 업데이트
    
    # 최종 답변을 세션에 저장(전체 답변)
    st.session_state.messages.append({'role':'ai', 'content': partial_text})

    # 참조 문서 표시
    documents = qa_module.retrieve_documents(prompt)
    display_retrieved_documents(documents)
