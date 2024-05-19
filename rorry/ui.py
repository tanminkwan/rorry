import streamlit as st

from agent import Agent

if 'initialized' not in st.session_state:

    st.session_state['system'] = "자산관리"
    st.session_state['agent'] = Agent("자산관리")
    st.session_state['initialized'] = True
    st.session_state['generated'] = []
    st.session_state['past'] = []

agent = st.session_state['agent']

# Streamlit UI 설정
st.title("No UI! No Code!")
st.write(st.session_state['system']+" 시스템에 명령을 내리세요.")

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

# 사용자 입력 받기 (화면 아래에 고정)
user_input = st.text_input(label = "You: ", key = "input")

if st.button("Send"):
    if user_input:
        response = generate_response(user_input)
        st.session_state.generated.append(response)
        st.session_state.past.append(f"{user_input}")
        st.session_state.past.append(f"AI: {response}")
        st.rerun()

# 스크롤을 아래로 자동으로 내려줌
st.write(f"<div style='height:100px;'></div>", unsafe_allow_html=True)
st.write("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)
