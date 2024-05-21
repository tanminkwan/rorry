import streamlit as st

from agent import Agent

def init_window(system_name):
    st.write(f"시스템 : {system_name}")
    st.session_state['system'] = system_name
    st.session_state['agent'] = Agent(system_name)
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state.messages = []

if 'initialized' not in st.session_state:

    init_window("자산관리")
    st.session_state['initialized'] = True

agent = st.session_state['agent']

if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI 설정
st.title("No UI! No Code!")
#st.write(st.session_state['system']+" 시스템에 명령을 내리세요.")

# Streamlit 사이드바에 드롭다운 메뉴 추가
st.sidebar.title("시스템 선택")
options = ['자산관리', '코어뱅킹', '채널관리', '인터넷뱅킹']
selected_option = st.sidebar.selectbox("시스템을 선택하세요 : ", options, index=0)

if st.session_state['system'] != selected_option:
    print(st.session_state['system'], selected_option)
    init_window(selected_option)


def generate_response(user_input):    
    #candidate = {"user_input": user_input, "history": "\n".join(st.session_state['past'])}
    say = {"user_input": user_input}
    response = agent.invoke(say)
    agent.memory.save_context(say, {'output':response})
    print(agent.memory.load_memory_variables({})["history"])
    return response

# 채팅 스레드를 위에 배치
chat_placeholder = st.container()

with chat_placeholder:
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            st.write(f"You: {st.session_state['past'][2*i]}")
            st.write(f"AI: {st.session_state['generated'][i]}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(st.session_state['system']+" 시스템에 명령을 내리세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant message in chat message container
    with st.spinner(text="Thinking..."):
        full_response = generate_response(prompt)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    with st.chat_message("assistant"):
        st.markdown(full_response)

# if st.button("Send"):
#     if user_input:
#         response = generate_response(user_input)
#         st.session_state.generated.append(response)
#         st.session_state.past.append(f"{user_input}")
#         st.session_state.past.append(f"AI: {response}")
#         st.rerun()

# 스크롤을 아래로 자동으로 내려줌
# st.write(f"<div style='height:100px;'></div>", unsafe_allow_html=True)
# st.write("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)
