import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import ast

st.set_page_config(layout="wide")
st.title("K-드라마 데이터 분석 및 예측 대시보드")

# =========================
# 0. 데이터 불러오기 (json → DataFrame)
# =========================
@st.cache_data
def load_data():
    raw = pd.read_json('drama_data.json')
    df = pd.DataFrame({col: pd.Series(val) for col, val in raw.items()})
    return df

df = load_data()

# =========================
# 리스트/멀티형 컬럼 파싱 및 flatten 함수
# =========================
def safe_eval(val):
    try: return ast.literal_eval(val)
    except: return []

def flatten_list_str(x):
    # 리스트면 콤마로 연결
    if isinstance(x, list):
        return ','.join([str(i).strip() for i in x])
    # 문자열인 경우 리스트형 문자열 처리
    if isinstance(x, str):
        try:
            obj = ast.literal_eval(x)
            if isinstance(obj, list):
                return ','.join([str(i).strip() for i in obj])
        except:
            return x
        return x
    # 결측치 확인 (리스트/문자열 아닌 경우에만)
    try:
        if pd.isnull(x):
            return ''
    except Exception:
        pass
    return str(x)

def preprocess_ml_features(X):
    # 컬럼이 실제로 존재할 때만 flatten 적용
    for col in ['장르', '플랫폼', '방영요일']:
        if col in X.columns:
            X[col] = X[col].apply(flatten_list_str)
    X = X.fillna('')
    return X


# =========================
# 장르/플랫폼/요일 등 리스트 데이터 추출(워드클라우드 등)
# =========================
genres = df['장르'].dropna().apply(safe_eval)
genre_list = [g.strip() for sublist in genres for g in sublist]
broadcasters = df['플랫폼'].dropna().apply(safe_eval)
broadcaster_list = [b.strip() for sublist in broadcasters for b in sublist]
week = df['방영요일'].dropna().apply(safe_eval)
week_list = [w.strip() for sublist in week for w in sublist]

# =========================
# 1. 사이드바(EDA 분석 메뉴)
# =========================
with st.sidebar:
    st.title("사이드바 1: EDA 분석")
    eda_tab = st.radio(
        "분석 항목 선택",
        [
            "데이터 개요", 
            "기초통계", 
            "분포/교차분석", 
            "워드클라우드", 
            "실시간 필터", 
            "상세 미리보기"
        ],
        key='eda_radio'
    )

# =========================
# 2. 사이드바(머신러닝 모델링)
# =========================
with st.sidebar:
    st.markdown("---")
    st.title("사이드바 2: 머신러닝 모델링")
    with st.expander("모델/파라미터 선택", expanded=False):
        model_type = st.selectbox('모델 선택', ['Random Forest', 'Linear Regression'])
        test_size = st.slider('테스트셋 비율', 0.1, 0.5, 0.2, 0.05)
        feature_cols = st.multiselect(
            '특성(Feature) 선택',
            ['나이', '방영년도', '성별', '장르', '배우명', '플랫폼', '결혼여부'],
            default=['나이', '방영년도', '장르']
        )

# =========================
# 3. 본문 탭: EDA + ML 탭 통합
# =========================
tab_labels = ["데이터 개요", "기초통계", "분포/교차분석", "워드클라우드", "실시간 필터", "상세 미리보기", "머신러닝 모델링"]
tabs = st.tabs(tab_labels)

# 1. 데이터 개요
with tabs[0]:
    if eda_tab == "데이터 개요":
        st.header("데이터 개요")
        st.write(f"전체 샘플 수: {df.shape[0]}")
        st.write(f"컬럼 개수: {df.shape[1]}")
        st.write(f"컬럼명: {list(df.columns)}")
        st.write("결측치 비율:")
        st.write(df.isnull().mean())
        st.write("데이터 예시:")
        st.dataframe(df.head())

# 2. 기초통계
with tabs[1]:
    if eda_tab == "기초통계":
        st.header("기초 통계")
        st.write(df['점수'].astype(float).describe())
        st.write(f"방영년도 유니크값: {df['방영년도'].nunique()}")
        st.write(f"장르 유니크값: {df['장르'].nunique()}")
        st.write(f"배우 유니크값: {df['배우명'].nunique()}")
        st.write("점수(평점) 히스토그램")
        fig, ax = plt.subplots()
        ax.hist(df['점수'].astype(float), bins=20, color='skyblue')
        st.pyplot(fig)

# 3. 분포/교차분석
with tabs[2]:
    if eda_tab == "분포/교차분석":
        st.header("분포/교차분석")
        genre_count = pd.Series(genre_list).value_counts().head(10)
        st.write("장르별 출연 횟수 (Top 10)")
        st.bar_chart(genre_count)
        st.write("방영년도별 작품 수")
        st.line_chart(df['방영년도'].value_counts().sort_index())
        genre_mean = {}
        for g in pd.Series(genre_list).unique():
            genre_mean[g] = df[df['장르'].str.contains(g, na=False)]['점수'].astype(float).mean()
        genre_mean_df = pd.DataFrame({'장르': genre_mean.keys(), '평균점수': genre_mean.values()}).sort_values('평균점수', ascending=False)
        st.write("장르별 평균 점수(상위 10)")
        st.dataframe(genre_mean_df.head(10))
        broadcaster_mean = {}
        for b in pd.Series(broadcaster_list).unique():
            broadcaster_mean[b] = df[df['플랫폼'].str.contains(b, na=False)]['점수'].astype(float).mean()
        broadcaster_mean_df = pd.DataFrame({'플랫폼': broadcaster_mean.keys(), '평균점수': broadcaster_mean.values()}).sort_values('평균점수', ascending=False)
        st.write("플랫폼별 평균 점수")
        st.dataframe(broadcaster_mean_df)

# 4. 워드클라우드
with tabs[3]:
    if eda_tab == "워드클라우드":
        st.header("텍스트 데이터 분석 (워드클라우드)")
        # 장르 워드클라우드
        if genre_list and ''.join(genre_list).strip():
            genre_words = ' '.join([g for g in genre_list if g])
            wc = WordCloud(width=800, height=400, background_color='white').generate(genre_words)
            fig1, ax1 = plt.subplots(figsize=(10,5))
            ax1.imshow(wc, interpolation='bilinear')
            ax1.axis('off')
            st.pyplot(fig1)
        else:
            st.info("장르 데이터가 충분하지 않아 워드클라우드를 생성할 수 없습니다.")

        # 플랫폼 워드클라우드
        if broadcaster_list and ''.join(broadcaster_list).strip():
            bc_words = ' '.join([b for b in broadcaster_list if b])
            wc2 = WordCloud(width=800, height=400, background_color='white').generate(bc_words)
            fig2, ax2 = plt.subplots(figsize=(10,5))
            ax2.imshow(wc2, interpolation='bilinear')
            ax2.axis('off')
            st.pyplot(fig2)
        else:
            st.info("플랫폼 데이터가 충분하지 않아 워드클라우드를 생성할 수 없습니다.")

        # 방영요일 워드클라우드
        if week_list and ''.join(week_list).strip():
            week_words = ' '.join([w for w in week_list if w])
            wc3 = WordCloud(width=800, height=400, background_color='white').generate(week_words)
            fig3, ax3 = plt.subplots(figsize=(10,5))
            ax3.imshow(wc3, interpolation='bilinear')
            ax3.axis('off')
            st.pyplot(fig3)
        else:
            st.info("방영요일 데이터가 충분하지 않아 워드클라우드를 생성할 수 없습니다.")

# 5. 실시간 필터
with tabs[4]:
    if eda_tab == "실시간 필터":
        st.header("실시간 필터")
        score_slider = st.slider("점수(이상)", float(df['점수'].min()), float(df['점수'].max()), 8.0, 0.1)
        genre_select = st.multiselect("장르 필터", sorted(set(genre_list)))
        year_select = st.slider("방영년도", int(df['방영년도'].min()), int(df['방영년도'].max()), (2010, 2022))
        filtered = df[
            (df['점수'].astype(float) >= score_slider) &
            (df['방영년도'] >= year_select[0]) & (df['방영년도'] <= year_select[1])
        ]
        if genre_select:
            filtered = filtered[filtered['장르'].apply(lambda x: any(g in x for g in genre_select))]
        st.write("필터 적용 데이터 미리보기 (TOP 10)")
        st.dataframe(filtered.head(10))

# 6. 상세 미리보기
with tabs[5]:
    if eda_tab == "상세 미리보기":
        st.header("상세 미리보기 (전체)")
        st.dataframe(df)

# 7. 머신러닝 모델링 (예시)
with tabs[6]:
    st.header("머신러닝 모델링")
    st.info("※ 여기서는 예시로 RandomForest/LinearRegression 회귀 예측을 보여줍니다.")
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

        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"**R2 Score:** {r2:.3f}")
        st.write(f"**Test MSE:** {mse:.3f}")
        st.write("실제 vs 예측", pd.DataFrame({'실제': y_test, '예측': y_pred}).head())
    else:
        st.warning("머신러닝 특성을 1개 이상 선택하세요.")

# =========================
# 3. 사이드바(평점 예측)
# =========================
with st.sidebar:
    st.markdown("---")
    st.title("사이드바 3: 평점 예측(입력→예상평점)")

    st.markdown("#### [아래 정보를 입력하면 예상 평점을 예측합니다]")
    input_dict = {}
    input_dict['나이'] = st.number_input("배우 나이", min_value=10, max_value=80, value=30)
    input_dict['방영년도'] = st.number_input("방영년도", min_value=2000, max_value=2025, value=2021)
    input_dict['성별'] = st.selectbox("배우 성별", sorted(df['성별'].dropna().unique()))
    input_dict['장르'] = st.multiselect("장르", sorted(set(genre_list)), default=['드라마'])
    input_dict['배우명'] = st.selectbox("배우명", sorted(df['배우명'].dropna().unique()))
    input_dict['플랫폼'] = st.multiselect("플랫폼", sorted(set(broadcaster_list)), default=['NETFLIX'])
    input_dict['결혼여부'] = st.selectbox("결혼여부", sorted(df['결혼여부'].dropna().unique()))

    predict_btn = st.button("예상 평점 예측하기")

st.write("왼쪽 사이드바 메뉴를 선택하세요.")

# =========================
# 실제 평점 예측(사이드바3) 처리
# =========================
if predict_btn:
    user_input = pd.DataFrame([{
        '나이': input_dict['나이'],
        '방영년도': input_dict['방영년도'],
        '성별': input_dict['성별'],
        '장르': input_dict['장르'],
        '배우명': input_dict['배우명'],
        '플랫폼': input_dict['플랫폼'],
        '결혼여부': input_dict['결혼여부']
    }])

    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    # 1. 훈련 데이터 전처리
    X = df[feature_cols].copy()
    y = df['점수'].astype(float)
    X = preprocess_ml_features(X)
    X = pd.get_dummies(X, columns=[col for col in feature_cols if X[col].dtype == 'object'])

    # 2. 입력 데이터 전처리
    user_input = preprocess_ml_features(user_input)
    user_input = pd.get_dummies(user_input, columns=[col for col in feature_cols if user_input[col].dtype == 'object'])

    # 3. 누락된 컬럼 채우기
    for col in X.columns:
        if col not in user_input.columns:
            user_input[col] = 0
    user_input = user_input[X.columns]

    # 4. 예측
    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(user_input)[0]

    st.success(f"💡 입력값 기준 예상 평점: **{pred:.2f}**")
    st.write("입력 정보:", user_input)
    st.write("모델 사용 특성:", feature_cols)
