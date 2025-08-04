import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="드라마 평점 예측", layout="centered")
st.title("🎬 배우·드라마 조합 평점 예측 대시보드")
st.markdown("배우와 드라마 속성 조합으로 예상 평점을 예측합니다.")

model = joblib.load("drama_rating_model.pkl")

actor_list = ["김수현", "송혜교", "이병헌", "전지현", "박은빈", "조인성"]
genre_list = ["로맨스", "스릴러", "코미디", "액션", "시대극", "판타지"]
platform_list = ["Netflix", "tvN", "SBS", "MBC", "KBS", "ENA"]

actor1 = st.selectbox("주연 배우 1", actor_list)
actor2 = st.selectbox("주연 배우 2", [a for a in actor_list if a != actor1])
genre = st.selectbox("장르", genre_list)
platform = st.selectbox("방송사 / OTT", platform_list)
release_date = st.date_input("방영 예정일")

if st.button("예상 평점 계산"):
    input_data = pd.DataFrame({
        "actor1": [actor1],
        "actor2": [actor2],
        "genre": [genre],
        "platform": [platform],
        "release_date": [release_date]
    })
    rating = model.predict(input_data)[0]
    st.metric(label="예상 평점", value=f"{rating:.2f} / 10")
    st.success("예측이 완료되었습니다!")
