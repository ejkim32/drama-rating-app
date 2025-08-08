import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import ast
import matplotlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import plotly.express as px

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False 

# =========================
# 0. 유틸리티 함수 및 클래스
# =========================
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
    def fit(self, X, y=None):
        lists = X.squeeze()
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
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except:
            return [x.strip()]
    return [str(x)]

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
            pass
    return []

def flatten_list_str(x):
    if isinstance(x, list):
        return ','.join(map(str, x))
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return ','.join(map(str, parsed))
        except:
            pass
    return str(x) if not pd.isna(x) else ''

def preprocess_ml_features(X: pd.DataFrame) -> pd.DataFrame:
    for col in ['장르','플랫폼','방영요일']:
        if col in X.columns:
            X[col] = X[col].apply(safe_eval).apply(flatten_list_str)
    return X.fillna('')

# =========================
# 1. 데이터 로드
# =========================
@st.cache_data
def load_data():
    raw = pd.read_json('drama_data.json')
    return pd.DataFrame({c: pd.Series(v) for c,v in raw.items()})

raw_df = load_data()              # EDA용 원본
df      = raw_df.copy()           # ML용 복사본

# =========================
# 2. 머신러닝용 전처리
# =========================
mlb_cols = ['장르','플랫폼','방영요일']
for col in mlb_cols:
    df[col] = df[col].apply(clean_cell)
    mlb = MultiLabelBinarizer()
    arr = mlb.fit_transform(df[col])
    df = pd.concat([
        df,
        pd.DataFrame(arr, columns=[f"{col}_{c.upper()}" for c in mlb.classes_], index=df.index)
    ], axis=1)
#df.drop(columns=mlb_cols, inplace=True)

# =========================
# 3. EDA용 리스트 풀기
# =========================
genre_list       = [g for sub in raw_df['장르'].dropna().apply(safe_eval) for g in sub]
broadcaster_list = [b for sub in raw_df['플랫폼'].dropna().apply(safe_eval) for b in sub]
week_list        = [w for sub in raw_df['방영요일'].dropna().apply(safe_eval) for w in sub]
unique_genres    = sorted(set(genre_list))

# =========================
# 4. Streamlit 레이아웃
# =========================
st.set_page_config(page_title="K-드라마 분석/예측", page_icon="🎬", layout="wide")
st.title("K-드라마 데이터 분석 및 예측 대시보드")

# 사이드바: ML 파라미터 + 예측 입력
with st.sidebar:
    st.header("🤖 모델 설정")
    model_type   = st.selectbox('모델 선택', ['Random Forest','Linear Regression'])
    test_size    = st.slider('테스트셋 비율', 0.1,0.5,0.2,0.05)
    feature_cols = st.multiselect('특성 선택',['나이','방영년도','성별','장르','배우명','플랫폼','결혼여부'], default=['나이','방영년도','장르'])
    st.markdown("---")
    st.header("🎯 평점 예측 입력")
    input_age     = st.number_input("배우 나이",10,80,30)
    input_year    = st.number_input("방영년도",2000,2025,2021)
    input_gender  = st.selectbox("성별", sorted(raw_df['성별'].dropna().unique()))
    input_genre   = st.multiselect("장르", unique_genres, default=unique_genres[:1])
    input_plat    = st.multiselect("플랫폼", sorted(set(broadcaster_list)), default=list({broadcaster_list[0]}))
    input_married = st.selectbox("결혼여부", sorted(raw_df['결혼여부'].dropna().unique()))
    predict_btn   = st.button("예측 실행")

# 탭 구성
tabs = st.tabs(["🗂개요","📊기초통계","📈분포/교차","💬워드클라우드","⚙️필터","🔍전체보기","🤖ML모델","🔧튜닝","🎯예측"])

# --- 4.1 데이터 개요 ---
with tabs[0]:
    st.header("데이터 개요")
    c1,c2,c3 = st.columns(3)
    c1.metric("샘플 수", df.shape[0])
    c2.metric("컬럼 수", df.shape[1])
    c3.metric("고유 장르", len(unique_genres))
    st.subheader("결측치 비율")
    st.dataframe(raw_df.isnull().mean())
    st.subheader("원본 샘플")
    st.dataframe(raw_df.head(), use_container_width=True)

# --- 4.2 기초통계 ---
with tabs[1]:
    st.header("기초 통계: 점수")
    st.write(raw_df['점수'].astype(float).describe())
    fig,ax=plt.subplots(figsize=(6,3))
    ax.hist(raw_df['점수'].astype(float), bins=20)
    ax.set_title("히스토그램")
    st.pyplot(fig)

# --- 4.3 분포/교차분석 ---
with tabs[2]:
    st.header("분포 및 교차분석")
    # 1) 점수 분포 & Top10
    st.subheader("전체 평점 분포")
    fig1 = px.histogram(raw_df, x='점수', nbins=20); st.plotly_chart(fig1)
    st.subheader("Top 10 평점 작품")
    top10 = raw_df.nlargest(10,'점수')[['드라마명','점수']].sort_values('점수')
    fig2 = px.bar(top10, x='점수', y='드라마명', orientation='h', text='점수'); st.plotly_chart(fig2)
    # 2) 연도별 플랫폼 수(원본 explode)
    st.subheader("연도별 주요 플랫폼 작품 수")
    ct = (pd.DataFrame({'방영년도':raw_df['방영년도'],'플랫폼':raw_df['플랫폼']})
          .explode('플랫폼')
          .groupby(['방영년도','플랫폼']).size().reset_index(name='count'))
    ct['플랫폼_up']=ct['플랫폼'].str.upper()
    focus = ['KBS','MBC','TVN','NETFLIX','JTBC']
    fig3=px.line(ct[ct['플랫폼_up'].isin(focus)], x='방영년도',y='count',color='플랫폼'); st.plotly_chart(fig3)
    # 3) 멀티장르 vs 단일장르
    st.subheader("멀티장르 vs 단일장르 평균 평점")
    ag = (pd.DataFrame({'배우명':raw_df['배우명'],'장르':raw_df['장르']})
          .explode('장르')
          .groupby('배우명')['장르'].nunique())
    multi = set(ag[ag>1].index)
    raw_df['장르구분']=raw_df['배우명'].apply(lambda x:'멀티장르' if x in multi else '단일장르')
    grp = raw_df.groupby('장르구분')['점수'].mean().round(2).reset_index()
    fig4=px.bar(grp,x='장르구분',y='점수',text='점수'); st.plotly_chart(fig4)

# --- 4.4 워드클라우드 ---
with tabs[3]:
    st.header("워드클라우드")
    if genre_list:
        wc = WordCloud(width=800,height=400,background_color='white').generate(' '.join(genre_list))
        fig,ax=plt.subplots(); ax.imshow(wc,interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)
    if broadcaster_list:
        wc = WordCloud(width=800,height=400,background_color='white').generate(' '.join(broadcaster_list))
        fig,ax=plt.subplots(); ax.imshow(wc,interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)
    if week_list:
        wc = WordCloud(width=800,height=400,background_color='white').generate(' '.join(week_list))
        fig,ax=plt.subplots(); ax.imshow(wc,interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)

# --- 4.5 실시간 필터 ---
with tabs[4]:
    st.header("실시간 필터")
    smin,smax = float(raw_df['점수'].min()), float(raw_df['점수'].max())
    sfilter = st.slider("최소 평점", smin,smax,smin)
    yfilter = st.slider("방영년도 범위", int(raw_df['방영년도'].min()),int(raw_df['방영년도'].max()),(2000,2025))
    filt = raw_df[(raw_df['점수']>=sfilter)&raw_df['방영년도'].between(*yfilter)]
    st.dataframe(filt.head(20))

# --- 4.6 전체 미리보기 ---
with tabs[5]:
    st.header("원본 전체보기")
    st.dataframe(raw_df, use_container_width=True)

# --- 4.7 머신러닝 모델링 ---
with tabs[6]:
    st.header("머신러닝 모델링")
    if feature_cols:
        X = df[feature_cols].copy()
        y = raw_df['점수'].astype(float)
        X = preprocess_ml_features(X)
        X = pd.get_dummies(X, drop_first=True)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=42)
        model = RandomForestRegressor(random_state=42) if model_type=="Random Forest" else LinearRegression()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        st.metric("R²", f"{r2_score(y_test,y_pred):.3f}")
        st.metric("MSE", f"{mean_squared_error(y_test,y_pred):.3f}")
    else:
        st.warning("사이드바에서 특성을 선택하세요.")

# --- 4.8 GridSearch 튜닝 ---
with tabs[7]:
    st.header("GridSearchCV 튜닝")
    st.info("이 탭은 필요 시 추가 구현")

# --- 4.9 예측 실행 ---
with tabs[8]:
    st.header("평점 예측")
    if predict_btn and feature_cols:
        X_all = df[feature_cols].copy()
        y_all = raw_df['점수'].astype(float)
        X_all = preprocess_ml_features(X_all)
        X_all = pd.get_dummies(X_all, drop_first=True)
        user = pd.DataFrame([{
            '나이':input_age,'방영년도':input_year,
            '성별':input_gender,'장르':input_genre,
            '배우명':raw_df['배우명'].dropna().iloc[0],
            '플랫폼':input_plat,'결혼여부':input_married
        }])
        u = preprocess_ml_features(user)
        u = pd.get_dummies(u, drop_first=True)
        for c in X_all.columns:
            if c not in u.columns: u[c]=0
        u = u[X_all.columns]
        mdl = RandomForestRegressor(random_state=42) if model_type=="Random Forest" else LinearRegression()
        mdl.fit(X_all,y_all)
        pred = mdl.predict(u)[0]
        st.success(f"💡 예상 평점: {pred:.2f}")
    elif predict_btn:
        st.error("특성을 먼저 선택하세요.")
