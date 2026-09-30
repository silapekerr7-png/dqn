import streamlit as st
import gymnasium as gym
import numpy as np
import torch
from dqn_agent import DQNAgent

st.title("DQN Ajanı - CartPole :video_game:")
st.write("Notebook'ta eğitilen `checkpoint.pth` yüklenir ve ajan CartPole oyununu oynar.")

bolum=st.slider('Kaç oyun oynasın?',1,20,10)
if st.button('Ajanı oynat'):
    env=gym.make("CartPole-v1")
    state_size=env.observation_space.shape[0]
    action_size=env.action_space.n
    agent=DQNAgent(state_size=state_size,action_size=action_size)
    try:
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    except FileNotFoundError:
        st.error("checkpoint.pth bulunamadı. Önce notebook'ta eğitimi çalıştırın.")
        st.stop()

    skorlar=[]
    for i in range(bolum):
        state,info=env.reset()
        done=False
        episode_reward=0
        while not done:
            action=agent.act(state)
            state,reward,terminated,truncated,info=env.step(action)
            done=terminated or truncated
            episode_reward+=reward
        skorlar.append(episode_reward)
    env.close()
    st.line_chart(skorlar)
    st.success(f'Ortalama skor: {np.mean(skorlar):.1f} (maksimum 500)')
