import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import ast
import matplotlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer,PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.stats as stats
from sklearn.ensemble      import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

# 한글 폰트 설정 (Windows: Malgun Gothic, macOS/Linux는 적절한 한글 폰트로)
# 1) 사용할 한글 폰트 이름 설정
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False 

class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    """리스트형 멀티카테고리(예: ['로맨스','스릴러'])를 이진 벡터로 변환"""
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
    def fit(self, X, y=None):
        # X: 2D array or DataFrame of lists
        lists = X.squeeze()  # 한 개 컬럼이면 Series로 변환
        self.mlb.fit(lists)
        return self
    def transform(self, X):
        lists = X.squeeze()
        return self.mlb.transform(lists)

def clean_cell(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return [x.strip()]
    return [str(x)]

# =========================
# 0. 페이지 설정
# =========================
st.set_page_config(
    page_title="K-드라마 데이터 분석 및 예측",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# 1. 데이터 로드
# =========================
@st.cache_data
def load_data():
    raw = pd.read_json('drama_data.json')
    return pd.DataFrame({col: pd.Series(vals) for col, vals in raw.items()})

df = load_data()

mlb_cols = ['장르','플랫폼','방영요일']
for col in mlb_cols:
    df[col] = df[col].apply(clean_cell)            # 리스트로 정리
    mlb = MultiLabelBinarizer()
    arr = mlb.fit_transform(df[col])
    new_cols = [f"{col}_{c.upper()}" for c in mlb.classes_]
    df = pd.concat([df, pd.DataFrame(arr, columns=new_cols, index=df.index)], axis=1)
df.drop(columns=mlb_cols, inplace=True)

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
    "🤖 머신러닝 모델링",
    "🔍 GridSearchCV",
    "🎯 예상 평점예측"
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
import plotly.express as px

with tabs[2]:
    st.header("분포/교차분석")

    # 1) 드라마 점수 분포 & Top10 평점 작품
    st.subheader("1) 드라마 평점 분포 & Top 10 평점 작품")
    # (a) 분포
    fig1 = px.histogram(
        df, x='점수', nbins=20,
        title='전체 드라마 평점 분포',
        labels={'점수':'평점','count':'빈도'}
    )
    st.plotly_chart(fig1, use_container_width=True)
    # (b) Top10
    top10 = df.nlargest(10, '점수')[['드라마명','점수']].sort_values('점수')
    top10_fig = px.bar(
        top10, x='점수', y='드라마명', orientation='h', 
        text=top10['점수'].map(lambda x: f"{x:.2f}"),
        title='Top 10 평점 작품',
        labels={'점수':'평점','드라마명':'드라마명'}
    )
    top10_fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(top10_fig, use_container_width=True)

    # 2) 연도별 플랫폼별 드라마 수
    st.subheader("2) 연도별 플랫폼별 드라마 수")

    # 원본 집계
    ct = df.explode('플랫폼').groupby(['방영년도','플랫폼']).size().reset_index(name='count')

    # 보고 싶은 플랫폼 리스트
    focus_plats = ['KBS', 'MBC', 'TVN', 'NETFLIX', 'JTBC']

    # 대문자/소문자 섞여 있을 수 있으니 모두 대문자로 맞춰 필터
    ct['플랫폼_up'] = ct['플랫폼'].str.upper()
    ct_focused = ct[ct['플랫폼_up'].isin(focus_plats)].copy()
    
    # 차트 그리기
    fig2 = px.line(
        ct_focused,
        x='방영년도', y='count', color='플랫폼',
        title='연도별 주요 플랫폼별 드라마 수',
        labels={'count':'작품 수','방영년도':'방영년도'}
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3) 인사이트 텍스트 출력
    st.markdown("""
**넷플릭스(OTT)의 급성장**  
넷플릭스는 2017년까지는 사실상 영향력이 거의 없었으나,  
2018년부터 급격하게 작품 수가 증가하여  
2020년에는 전통 방송사들과 어깨를 나란히 하고,  
2021년에는 최상위권(21편)까지 치고 올라감  
→ 국내 드라마 산업의 OTT 중심 재편 흐름이 명확하게 드러남  

**지상파(KBS, MBC)의 지속적 감소**  
KBS, MBC 등 전통 3사의 드라마 편수는 매년 꾸준히 감소  
이는 OTT, 케이블, 신생 플랫폼과의 경쟁 격화와  
제작 예산·시청률 하락의 영향으로 풀이됨

**TVN의 성장과 정체**  
tvN 작품 중 상당수가 넷플릭스 등 OTT와 동시방영(멀티플랫폼) 되는 구조  
→ 자체 채널 성장과 동시에 OTT 제휴로 시청자 접점을 확보

**2022년 전반적 감소**  
2022년에는 모든 플랫폼에서 작품 수가 전년 대비 감소  
이는 코로나19 등 외부 요인과  
OTT-방송사 동시방영 등 시장 재편이 복합적으로 작용한 결과
""")


    # 3) 멀티장르 vs 단일장르 배우 평균 평점 비교
    st.subheader("3) 멀티장르 vs 단일장르 배우 평균 평점")
    actor_genre_counts = df.explode('장르').groupby('배우명')['장르'].nunique()
    multi = actor_genre_counts[actor_genre_counts>1].index
    df['장르구분'] = df['배우명'].apply(lambda x: '멀티장르' if x in multi else '단일장르')
    grp = df.groupby('장르구분')['점수'].mean().reset_index()
    grp['점수'] = grp['점수'].round(2)
    fig3 = px.bar(
        grp, x='장르구분', y='점수', text='점수',
        title='배우 장르구분별 평균 평점',
        labels={'점수':'평균 평점','장르구분':'배우구분'}
    )
    fig3.update_yaxes(
    tickformat=".2f",   # 소수점 2자리까지 레이블
    dtick=0.5,          # 0.1 단위 눈금
    showgrid=False,      # 그리드 켜기
    gridcolor="LightGray",
    gridwidth=1
    )
    st.plotly_chart(fig3, use_container_width=True)

    # 4) 신인 vs 경력 배우 평균 평점 비교
    st.subheader("4) 신인(1-2개) vs 경력(3+개) 배우 평균 평점")
    actor_counts = df.groupby('배우명').size()
    newbies = actor_counts[actor_counts<=2].index
    vets    = actor_counts[actor_counts>=3].index
    df['경력구분'] = df['배우명'].apply(
        lambda x: '신인(1-2개)' if x in newbies else ('경력(3+개)' if x in vets else '기타')
    )
    grp2 = (
        df[df['경력구분']!='기타']
        .groupby('경력구분')['점수']
        .mean()
        .reset_index()
    )
    grp2['점수'] = grp2['점수'].round(2)
    fig4 = px.bar(
        grp2, x='경력구분', y='점수', text='점수',
        title='경력구분별 평균 평점',
        labels={'점수':'평균 평점','경력구분':'배우구분'}
    )
    fig4.update_yaxes(
    tickformat=".2f",   # 소수점 2자리까지 레이블
    dtick=0.5,          # 0.1 단위 눈금
    showgrid=False,      # 그리드 켜기
    gridcolor="LightGray",
    gridwidth=1
    )
    st.plotly_chart(fig4, use_container_width=True)

    # 5) 연도별 Top5 장르 드라마 수 변화
    st.subheader("5) 연도별 Top5 장르 드라마 수 변화")
    top5 = pd.Series(genre_list).value_counts().head(5).index
    df_top5 = df.explode('장르').query("장르 in @top5")
    ct5 = df_top5.groupby(['방영년도','장르']).size().reset_index(name='count')
    fig5 = px.line(
        ct5, x='방영년도', y='count', color='장르',
        title='연도별 Top5 장르 작품 수 변화',
        labels={'count':'작품 수','방영년도':'방영년도'}
    )
    st.plotly_chart(fig5, use_container_width=True)

    # 6) 배우별 평점 변동성(표준편차 상위 10)
    st.subheader("6) 배우별 평점 변동성 (표준편차 상위 10)")
    actor_std = (
        df.groupby('배우명')['점수']
        .std()
        .dropna()
        .nlargest(10)
        .reset_index(name='std')
    )
    actor_std['std'] = actor_std['std'].round(2)
    fig6 = px.bar(
        actor_std, x='배우명', y='std', text='std',
        title='평점 변동성 상위 10 배우',
        labels={'std':'표준편차','배우명':'배우명'}
    )
    fig6.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig6, use_container_width=True)

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
        
with tabs[7]:
    st.header("🔧 모델 튜닝 (GridSearchCV)")

    # 1) 모델 선택
    model_name = st.selectbox(
        label= "모델 선택",
        options = [
            "KNN", "LinearRegression", "Ridge", "Lasso",
            "ElasticNet", "SGDRegressor", "SVR",
            "DecisionTree", "RandomForest", "XGBRegressor"
        ],
        key="tune_model"
    )

    # 2) 공통 파라미터 입력
    test_size    = st.slider("테스트셋 비율", 0.1, 0.5, 0.2, 0.05, key="tune_ts")
    safe_feats   = [c for c in df.columns if c != "점수"]
    feature_cols = st.multiselect("튜닝할 특성 선택", safe_feats, key="tune_feats")

    if not feature_cols:
        st.warning("특성을 1개 이상 선택하세요.")
    else:
        # 3) 데이터 분할
        X = df[feature_cols]
        y = df["점수"].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # 4) 컬럼 분류
        num_cols   = [c for c in feature_cols if X_train[c].dtype in ("int64","float64")]
        cat_single = [c for c in feature_cols
                        if X_train[c].dtype == "object"
                           and not isinstance(X_train[c].iloc[0], list)]
        cat_multi  = [c for c in feature_cols
                        if X_train[c].dtype == "object"
                           and isinstance(X_train[c].iloc[0], list)]

        # 5) 전처리기
        preprocessor = ColumnTransformer([
            ("num",    "passthrough",                           num_cols),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_single),
            ("mlb",    MultiLabelBinarizerTransformer(),        cat_multi),
        ], remainder="drop")

        # 6) 모델 맵핑
        from sklearn.neighbors     import KNeighborsRegressor
        from sklearn.linear_model  import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
        from sklearn.svm           import SVR
        from sklearn.tree          import DecisionTreeRegressor
        from sklearn.ensemble      import RandomForestRegressor
        from xgboost               import XGBRegressor

        model_map = {
            "KNN":              KNeighborsRegressor(),
            "LinearRegression": LinearRegression(),
            "Ridge":            Ridge(),
            "Lasso":            Lasso(),
            "ElasticNet":       ElasticNet(),
            "SGDRegressor":     SGDRegressor(max_iter=1000, tol=1e-3),
            "SVR":              SVR(),
            "DecisionTree":     DecisionTreeRegressor(random_state=42),
            "RandomForest":     RandomForestRegressor(random_state=42),
            "XGBRegressor":     XGBRegressor(random_state=42, use_label_encoder=False, eval_metric="rmse")
        }
        model = model_map[model_name]

        # 7) 파이프라인
        steps = [("pre", preprocessor)]
        if model_name == "KNN":
            steps += [
                ("poly",  PolynomialFeatures(include_bias=False)),
                ("scale", StandardScaler())
            ]
        steps.append(("model", model))
        pipe = Pipeline(steps)

        # 8) 하이퍼파라미터 그리드
        grids = {
            "KNN":            {"poly__degree":[1,2,3], "model__n_neighbors":list(range(3,11))},
            "LinearRegression":{},
            "Ridge":          {"model__alpha":[0.1,1.0,10.0]},
            "Lasso":          {"model__alpha":[0.001,0.01,0.1,1.0]},
            "ElasticNet":     {"model__alpha":[0.01,0.1,1.0], "model__l1_ratio":[0.2,0.5,0.8]},
            "SGDRegressor":   {"model__alpha":[1e-4,1e-3,1e-2], "model__penalty":["l2","l1","elasticnet"]},
            "SVR":            {"model__C":[0.1,1,10], "model__gamma":["scale","auto"]},
            "DecisionTree":   {"model__max_depth":[None,5,10,20]},
            "RandomForest":   {"model__n_estimators":[50,100,200], "model__max_depth":[None,5,10]},
            "XGBRegressor":   {"model__n_estimators":[50,100,200], "model__max_depth":[3,6,9]}
        }
        param_grid = grids[model_name]

        # 9) GridSearchCV
        with st.spinner("GridSearchCV 실행 중…"):
            gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1,
                              scoring="r2", error_score="raise")
            gs.fit(X_train, y_train)

        # 10) 출력
        st.subheader("최적 파라미터")
        st.json(gs.best_params_)
        st.metric("Best CV R²", f"{gs.best_score_:.3f}")

        y_pred = gs.predict(X_test)
        st.subheader("테스트 세트 성능")
        st.metric("Test R²",   f"{r2_score(y_test, y_pred):.3f}")
        st.metric("Test RMSE", f"{mean_squared_error(y_test, y_pred, squared=False):.3f}")


with tabs[8]:
    st.header("🎯 예상 평점예측")

    st.subheader("1) 모델 설정")
    model_type  = st.selectbox('모델 선택', ['Random Forest', 'Linear Regression'])
    test_size   = st.slider('테스트셋 비율', 0.1, 0.5, 0.2, 0.05)
    feature_cols = st.multiselect(
        '특성 선택',
        ['나이','방영년도','성별','장르','배우명','플랫폼','결혼여부'],
        default=['나이','방영년도','장르']
    )

    st.markdown("---")
    st.subheader("2) 예측 입력")
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

    if predict_btn:
        # --- 1. 훈련 데이터 전처리 ---
        X_all = df[feature_cols].copy()
        y_all = df['점수'].astype(float)
        X_all = preprocess_ml_features(X_all)
        X_all = pd.get_dummies(X_all, columns=[c for c in X_all.columns if X_all[c].dtype=='object'])

        # --- 2. 입력 데이터 전처리 ---
        user_df = pd.DataFrame([{
            '나이': input_age,
            '방영년도': input_year,
            '성별': input_gender,
            '장르': input_genre,
            '배우명': df['배우명'].dropna().iloc[0],  # 필요시 selectbox로 변경
            '플랫폼': input_plat,
            '결혼여부': input_married
        }])
        u = preprocess_ml_features(user_df)
        u = pd.get_dummies(u, columns=[c for c in u.columns if u[c].dtype=='object'])
        for c in X_all.columns:
            if c not in u.columns:
                u[c] = 0
        u = u[X_all.columns]

        # --- 3. 모델 학습 & 예측 ---
        model = RandomForestRegressor(n_estimators=100, random_state=42) \
                if model_type=="Random Forest" else LinearRegression()
        model.fit(X_all, y_all)
        pred = model.predict(u)[0]

        st.success(f"💡 예상 평점: {pred:.2f}")


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
