import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import ast
import matplotlib
# 한글 폰트 설정 (Windows: Malgun Gothic, macOS/Linux는 적절한 한글 폰트로)
matplotlib.rc('font', family='Malgun Gothic')

# =========================
# 0. 페이지 설정
# =========================
st.set_page_config(
    page_title="K-드라마 데이터 분석 및 예측",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 1. 데이터 로드
# =========================
@st.cache_data
def load_data():
    raw = pd.read_json('drama_data.json')
    return pd.DataFrame({col: pd.Series(vals) for col, vals in raw.items()})

df = load_data()

# =========================
# 2. 전처리 함수
# =========================
def safe_eval(val):
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except:
            return []
    return []

def flatten_list_str(x):
    if isinstance(x, list):
        return ','.join([str(i).strip() for i in x])
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return ','.join([str(i).strip() for i in parsed])
        except:
            return x
    try:
        if pd.isna(x):
            return ''
    except:
        pass
    return str(x)

def preprocess_ml_features(X):
    for col in ['장르', '플랫폼', '방영요일']:
        if col in X.columns:
            X[col] = X[col].apply(safe_eval).apply(flatten_list_str)
    return X.fillna('')

# =========================
# 3. 리스트형 컬럼 풀기 (EDA 탭용)
# =========================
genres = df['장르'].dropna().apply(safe_eval)
genre_list = [g for sub in genres for g in sub]

broadcasters = df['플랫폼'].dropna().apply(safe_eval)
broadcaster_list = [b for sub in broadcasters for b in sub]

weeks = df['방영요일'].dropna().apply(safe_eval)
week_list = [w for sub in weeks for w in sub]

# 고유 장르 수
unique_genres = set(genre_list)

# =========================
# 5. 사이드바: ML 파라미터 & 예측 입력
# =========================
with st.sidebar:
    st.header("🤖 모델 설정")
    model_type = st.selectbox('모델 선택', ['Random Forest', 'Linear Regression'])
    test_size = st.slider('테스트셋 비율', 0.1, 0.5, 0.2, 0.05)
    feature_cols = st.multiselect(
        '특성 선택',
        ['나이','방영년도','성별','장르','배우명','플랫폼','결혼여부'],
        default=['나이','방영년도','장르']
    )

    st.markdown("---")
    st.header("🎯 예상 평점 예측")
    input_age     = st.number_input("배우 나이", 10, 80, 30)
    input_year    = st.number_input("방영년도", 2000, 2025, 2021)
    input_gender  = st.selectbox("성별", sorted(df['성별'].dropna().unique()))
    genre_opts    = sorted(unique_genres)
    default_genre = [genre_opts[0]] if genre_opts else []
    input_genre   = st.multiselect("장르", genre_opts, default=default_genre)
    platform_opts = sorted(set(broadcaster_list))
    default_plat  = [platform_opts[0]] if platform_opts else []
    input_plat    = st.multiselect("플랫폼", platform_opts, default=default_plat)
    input_married = st.selectbox("결혼여부", sorted(df['결혼여부'].dropna().unique()))
    predict_btn   = st.button("예측 실행")
    
# =========================
# 4. 본문: 탭으로 EDA & ML
# =========================
st.title("K-드라마 데이터 분석 및 예측 대시보드")

tab_labels = [
    "🗂 데이터 개요",
    "📊 기초통계",
    "📈 분포/교차분석",
    "💬 워드클라우드",
    "⚙️ 실시간 필터",
    "🔍 상세 미리보기",
    "🤖 머신러닝 모델링"
]
tabs = st.tabs(tab_labels)

# 4.1 데이터 개요
with tabs[0]:
    st.header("데이터 개요")
    col1, col2, col3 = st.columns(3)
    col1.metric("전체 샘플 수", df.shape[0])
    col2.metric("전체 컬럼 수", df.shape[1])
    col3.metric("고유 장르 수", len(unique_genres))
    st.subheader("결측치 비율")
    st.write(df.isnull().mean())
    st.subheader("데이터 미리보기")
    st.dataframe(df.head(), use_container_width=True)

# 4.2 기초통계
with tabs[1]:
    st.header("기초 통계")
    st.write(df['점수'].astype(float).describe())
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(df['점수'].astype(float), bins=20)
    ax.set_title("점수 분포 히스토그램")
    st.pyplot(fig, use_container_width=True)

# 4.3 분포/교차분석
with tabs[2]:
    st.header("분포/교차분석")

    # 1) 드라마 점수 분포 & Top10 평점 작품
    st.subheader("1) 드라마 점수 분포 & Top 10 평점 작품")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # (a) 점수 분포
    axes[0].hist(df['점수'].astype(float), bins=20, color='skyblue')
    axes[0].set_title("전체 드라마 평점 분포")
    axes[0].set_xlabel("평점")
    axes[0].set_ylabel("빈도")
    # (b) Top10 평점 작품
    top10 = df.nlargest(10, '점수')[['드라마명','점수']].set_index('드라마명')
    top10 = top10.sort_values('점수')  # 막대 차트를 위에서 아래로
    axes[1].barh(top10.index, top10['점수'], color='salmon')
    axes[1].set_title("Top 10 평점 작품")
    axes[1].set_xlabel("평점")
    # 소수점 표시
    for y, v in enumerate(top10['점수']):
        axes[1].text(v + 0.02, y, f"{v:.2f}", va='center')
    st.pyplot(fig, use_container_width=True)

    # 2) 연도별 플랫폼별 드라마 수
    st.subheader("2) 연도별 플랫폼별 드라마 수")
    ct = df.explode('플랫폼').groupby(['방영년도','플랫폼']).size().unstack(fill_value=0)
    st.line_chart(ct, use_container_width=True)

    # 3) 멀티장르 vs 단일장르 배우 평균 평점 비교
    st.subheader("3) 멀티장르 배우 vs 단일장르 배우 평균 평점")
    actor_genre_counts = df.explode('장르').groupby('배우명')['장르'].nunique()
    multi_idx = actor_genre_counts[actor_genre_counts > 1].index
    df['장르구분'] = df['배우명'].apply(lambda x: '멀티장르' if x in multi_idx else '단일장르')
    grp = df.groupby('장르구분')['점수'].mean()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(grp.index, grp.values, color=['#66c2a5','#fc8d62'])
    ax.set_title("배우 장르구분별 평균 평점")
    ax.set_ylabel("평균 평점")
    for i, v in enumerate(grp.values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
    st.pyplot(fig, use_container_width=True)

    # 4) 신인 vs 경력 배우 평균 평점 비교
    st.subheader("4) 신인(1-2개) vs 경력(3+개) 배우 평균 평점")
    actor_counts = df.groupby('배우명').size()
    newbies = actor_counts[actor_counts <= 2].index
    vets    = actor_counts[actor_counts >= 3].index
    df['경력구분'] = df['배우명'].apply(
        lambda x: '신인(1-2개)' if x in newbies else ('경력(3+개)' if x in vets else '')
    )
    grp2 = df[df['경력구분']!=''].groupby('경력구분')['점수'].mean()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(grp2.index, grp2.values, color=['#8da0cb','#e78ac3'])
    ax.set_title("경력구분별 평균 평점")
    ax.set_ylabel("평균 평점")
    for i, v in enumerate(grp2.values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
    st.pyplot(fig, use_container_width=True)

    # 5) 연도별 Top5 장르 드라마 수 변화 (라인 플롯)
    st.subheader("5) 연도별 Top5 장르 드라마 수 변화")
    top5_genres = pd.Series(genre_list).value_counts().head(5).index
    df_top5 = df.explode('장르').query("장르 in @top5_genres")
    ct2 = df_top5.groupby(['방영년도','장르']).size().unstack(fill_value=0)
    st.line_chart(ct2, use_container_width=True)

    # 6) 배우별 평점 변동성 (표준편차 상위 10)
    st.subheader("6) 배우별 평점 변동성 (표준편차 상위 10)")
    actor_std = df.groupby('배우명')['점수'].std().dropna().nlargest(10)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(actor_std.index, actor_std.values, color='tan')
    ax.set_title("평점 변동성(표준편차) 상위 10 배우")
    ax.set_ylabel("표준편차")
    ax.set_xticklabels(actor_std.index, rotation=45, ha='right')
    st.pyplot(fig, use_container_width=True)

# 4.4 워드클라우드
with tabs[3]:
    st.header("워드클라우드")
    # 장르
    if genre_list:
        wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(genre_list))
        fig, ax = plt.subplots(figsize=(8,4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("장르 데이터 부족")

    # 플랫폼
    if broadcaster_list:
        wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(broadcaster_list))
        fig, ax = plt.subplots(figsize=(8,4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("플랫폼 데이터 부족")

    # 요일
    if week_list:
        wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(week_list))
        fig, ax = plt.subplots(figsize=(8,4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("요일 데이터 부족")

# 4.5 실시간 필터
with tabs[4]:
    st.header("실시간 필터")
    score_min, score_max = float(df['점수'].min()), float(df['점수'].max())
    score_slider = st.slider("점수 이상", score_min, score_max, score_min)
    genre_opts = sorted(unique_genres)
    genre_select = st.multiselect("장르 필터", genre_opts)
    year_min, year_max = int(df['방영년도'].min()), int(df['방영년도'].max())
    year_select = st.slider("방영년도 범위", year_min, year_max, (year_min, year_max))

    filtered = df[
        (df['점수'].astype(float) >= score_slider) &
        df['방영년도'].between(year_select[0], year_select[1])
    ]
    if genre_select:
        filtered = filtered[filtered['장르'].apply(lambda x: any(g in x for g in genre_select))]
    st.dataframe(filtered.head(10), use_container_width=True)

# 4.6 상세 미리보기
with tabs[5]:
    st.header("상세 미리보기")
    st.dataframe(df, use_container_width=True)

# 4.7 머신러닝 모델링
with tabs[6]:
    st.header("머신러닝 모델링")
    st.info("Random Forest / Linear Regression 회귀 예시")
    # 사이드바에서 선택한 feature_cols, model_type, test_size 사용
    if len(feature_cols) > 0:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error

        X = df[feature_cols].copy()
        y = df['점수'].astype(float)
        X = preprocess_ml_features(X)
        X = pd.get_dummies(X, columns=[c for c in X.columns if X[c].dtype == 'object'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42) if model_type=="Random Forest" else LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
        st.metric("Test MSE", f"{mean_squared_error(y_test, y_pred):.3f}")
        st.subheader("실제 vs 예측 (상위 5)")
        st.dataframe(pd.DataFrame({'실제': y_test, '예측': y_pred}).head())
    else:
        st.warning("사이드바에서 특성을 1개 이상 선택하세요.")


# =========================
# 6. 예측 실행
# =========================
if predict_btn:
    user_input = pd.DataFrame([{
        '나이': input_age,
        '방영년도': input_year,
        '성별': input_gender,
        '장르': input_genre,
        '배우명': st.selectbox("배우명", sorted(df['배우명'].dropna().unique())),  # 모델링 탭과 동일하게
        '플랫폼': input_plat,
        '결혼여부': input_married
    }])

    # 전처리 & 인코딩
    X_all = df[feature_cols].copy()
    y_all = df['점수'].astype(float)
    X_all = preprocess_ml_features(X_all)
    X_all = pd.get_dummies(X_all, columns=[c for c in X_all.columns if X_all[c].dtype == 'object'])

    user_proc = preprocess_ml_features(user_input)
    user_proc = pd.get_dummies(user_proc, columns=[c for c in user_proc.columns if user_proc[c].dtype == 'object'])
    for col in X_all.columns:
        if col not in user_proc.columns:
            user_proc[col] = 0
    user_proc = user_proc[X_all.columns]

    # 모델 학습 & 예측
    model = RandomForestRegressor(n_estimators=100, random_state=42) if model_type=="Random Forest" else LinearRegression()
    model.fit(X_all, y_all)
    prediction = model.predict(user_proc)[0]

    st.success(f"💡 예상 평점: {prediction:.2f}")
