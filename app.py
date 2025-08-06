import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv('merged_kdrama.csv')

df = load_data()

# ========================
# 1. 분석(EDA) 사이드바
# ========================
# 사이드바: 필터/탐색
st.sidebar.title("1. 분석(EDA) 패널")
with st.sidebar.expander("필터 및 탐색", expanded=True):
    genre_options = st.multiselect('장르 선택', sorted(df['Genres'].unique()))
    min_score = st.slider('최소 IMDB 평점', 7.0, 10.0, 8.0, 0.1)
    year_range = st.slider('방영연도 범위', int(df['year'].min()), int(df['year'].max()), (2010, 2022))

filtered = df[
    (df['imdb_rating'] >= min_score) &
    (df['year'] >= year_range[0]) &
    (df['year'] <= year_range[1])
]
if genre_options:
    filtered = filtered[filtered['Genres'].isin(genre_options)]

# =======================
# 본문: EDA 핵심 항목 나열
# =======================

st.header("📊 분석(EDA) 결과 요약")

# 1. 기본 데이터 정보
st.subheader("1. 기본 데이터 정보")
st.write(f"전체 샘플 수: {df.shape[0]}")
st.write(f"컬럼 개수: {df.shape[1]}")
st.write(f"주요 컬럼: {list(df.columns)}")
st.write("결측치(Null) 비율:")
st.write(df.isnull().mean())

st.write("예시 5행:")
st.dataframe(df.head())

# 2. 기초 통계
st.subheader("2. 기초 통계")
st.write("IMDB 평점 요약 통계:")
st.write(df['imdb_rating'].describe())
st.write("연도별 유니크값:", df['year'].nunique())
st.write("장르별 유니크값:", df['Genres'].nunique())
if 'actor' in df.columns:
    st.write("배우 유니크값:", df['actor'].nunique())

# 3. 분포 시각화
st.subheader("3. 분포 시각화")
st.write("IMDB 평점 분포")
st.hist(df['imdb_rating'], bins=20)
st.bar_chart(df['Genres'].value_counts())
st.line_chart(df['year'].value_counts().sort_index())

# 4. 교차분석
st.subheader("4. 교차분석")
st.write("장르별 평균 평점")
st.dataframe(df.groupby('Genres')['imdb_rating'].mean().sort_values(ascending=False))
if 'org_net' in df.columns:
    st.write("방송사별 평균 평점")
    st.dataframe(df.groupby('org_net')['imdb_rating'].mean().sort_values(ascending=False))

# 5. 상관관계 분석 (숫자형 변수 있을 때)
if 'actor_age' in df.columns or 'drama_pop' in df.columns:
    st.subheader("5. 상관관계 분석")
    st.write("상관관계 히트맵")
    import seaborn as sns
    import matplotlib.pyplot as plt
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# 6. 텍스트 데이터 분석
if 'synopsis' in df.columns:
    st.subheader("6. 줄거리 워드클라우드")
    wc_text = ' '.join(df['synopsis'].fillna(''))
    wc = WordCloud(width=800, height=400, background_color='white').generate(wc_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# 7. 실시간 필터/슬라이더/검색(사이드바에서 이미 구현)
# (추가적으로 검색창을 본문에 더 넣을 수도 있음)

# 8. 상세 데이터 보기
st.subheader("8. 상세 데이터 미리보기")
st.dataframe(filtered.head(10))


# ========================
# 2. 모델링 사이드바
# ========================
st.sidebar.title("2. 머신러닝 모델링")
with st.sidebar.expander("모델 및 하이퍼파라미터", expanded=True):
    model_type = st.selectbox('모델 선택', ['Random Forest', 'Linear Regression'])
    test_size = st.slider('테스트셋 비율', 0.1, 0.5, 0.2, 0.05)
    rf_n_estimators = st.number_input('RF 트리 개수', 10, 500, 100, step=10) if model_type == 'Random Forest' else None
    feature_cols = st.multiselect(
        '특성(Feature) 선택',
        ['actor_age', 'drama_pop', 'year', 'Genres', 'actor', 'director'], # 실제 컬럼명에 맞게 조정
        default=['year', 'genre', 'actor_age']
    )

# ========================
# 메인: 분석/시각화 & ML 예측
# ========================
st.title("K-Drama & Actor 평점 예측 대시보드")

st.header("1. 데이터 탐색 및 시각화")
st.write(f"필터링된 샘플: {filtered.shape[0]}")
st.dataframe(filtered[['drama_name','imdb_rating','genre','year','actor','actor_age']].head())

# 장르/연도별 분포
st.subheader("장르별 분포")
st.bar_chart(filtered['genre'].value_counts())

st.subheader("연도별 분포")
st.line_chart(filtered['year'].value_counts().sort_index())

# 줄거리 워드클라우드
if 'synopsis' in filtered.columns:
    st.subheader("줄거리 워드클라우드")
    wc_text = ' '.join(filtered['synopsis'].fillna(''))
    if st.button('워드클라우드 생성'):
        wc = WordCloud(width=800, height=400, background_color='white').generate(wc_text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# ========================
# 머신러닝: 평점 예측
# ========================
st.header("2. 머신러닝 평점 예측")
if st.button("모델 학습 및 예측"):
    # 간단한 데이터 전처리 (예시, 실제로는 카테고리 인코딩 등 추가 필요)
    X = filtered[feature_cols].copy()
    y = filtered['imdb_rating']

    # 예시: 카테고리형 특성 One-hot 인코딩
    X = pd.get_dummies(X, columns=[col for col in X.columns if X[col].dtype == 'object'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=rf_n_estimators, random_state=42)
    else:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    st.write(f"**R2 Score:** {r2:.3f}")
    st.write(f"**Test MSE:** {mse:.3f}")
    st.write("실제 vs 예측", pd.DataFrame({'실제': y_test, '예측': y_pred}).head())

# ========================
# 데이터 다운로드
# ========================
st.sidebar.download_button('필터링 데이터 다운로드', filtered.to_csv(index=False), file_name='filtered_kdrama.csv')

