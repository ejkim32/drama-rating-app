# app.py
# ---- dependency guard (optional) ----
import importlib.util, streamlit as st
_missing = [m for m in ("numpy","scipy","sklearn","joblib","threadpoolctl","xgboost") if importlib.util.find_spec(m) is None]
if _missing:
    st.error(f"필수 라이브러리 미설치: {_missing}. requirements.txt / runtime.txt 버전을 고정해 재배포하세요.")
    st.stop()

# ===== 페이지 설정 (반드시 첫 스트림릿 호출) =====
st.set_page_config(page_title="케미스코어 | K-드라마 분석/예측", page_icon="🎬", layout="wide")

import os
import ast
import random
import numpy as np
import pandas as pd
from pathlib import Path
import platform
from sklearn.metrics import mean_squared_error
import plotly.express as px
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.base import clone
import re
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer

# XGB가 설치돼 있으면 쓰도록 안전하게 추가
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ===== 전역 시드 고정 =====
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)

# ===== 한글 폰트 부트스트랩 =====
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

if st.session_state.get("font_cache_cleared") is not True:
    import shutil
    shutil.rmtree(matplotlib.get_cachedir(), ignore_errors=True)
    st.session_state["font_cache_cleared"] = True

def ensure_korean_font():
    matplotlib.rcParams['axes.unicode_minus'] = False
    base = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    candidates = [
        base / "fonts" / "NanumGothic-Regular.ttf",
        base / "fonts" / "NanumGothic.ttf",
    ]
    if platform.system() == "Windows":
        candidates += [Path(r"C:\Windows\Fonts\malgun.ttf"), Path(r"C:\Windows\Fonts\malgunbd.ttf")]
    wanted = ("nanum","malgun","applegothic","notosanscjk","sourcehan","gulim","dotum",
              "batang","pretendard","gowun","spoqa")
    for f in fm.findSystemFonts(fontext="ttf"):
        if any(k in os.path.basename(f).lower() for k in wanted):
            candidates.append(Path(f))
    for p in candidates:
        try:
            if p.exists():
                fm.fontManager.addfont(str(p))
                family = fm.FontProperties(fname=str(p)).get_name()
                matplotlib.rcParams['font.family'] = family
                st.session_state["kfont_path"] = str(p)  # WordCloud용
                return family
        except Exception:
            continue
    st.session_state["kfont_path"] = None
    return None

_ = ensure_korean_font()

# ===== 전처리 유틸 =====
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

def clean_cell_colab(x):
    if isinstance(x, list):
        return [str(i).strip() for i in x if pd.notna(i)]
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return [str(i).strip() for i in parsed if pd.notna(i)]
            return [x.strip()]
        except Exception:
            return [x.strip()]
    return [str(x).strip()]

def colab_multilabel_fit_transform(df: pd.DataFrame, cols=('genres','day','network')) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].apply(clean_cell_colab)
        mlb = MultiLabelBinarizer()
        arr = mlb.fit_transform(out[col])
        new_cols = [f"{col}_{c.strip().upper()}" for c in mlb.classes_]
        out = out.drop(columns=[c for c in new_cols if c in out.columns], errors='ignore')
        out = pd.concat([out, pd.DataFrame(arr, columns=new_cols, index=out.index)], axis=1)
        st.session_state[f"mlb_{col}"] = mlb
        st.session_state[f"mlb_classes_{col}"] = mlb.classes_.tolist()
    return out

def colab_multilabel_transform(df: pd.DataFrame, cols=('genres','day','network')) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].apply(clean_cell_colab)
        mlb = st.session_state.get(f"mlb_{col}", None)
        if mlb is None:
            classes = st.session_state.get(f"mlb_classes_{col}", [])
            mlb = MultiLabelBinarizer()
            if classes:
                mlb.classes_ = np.array(classes)
            else:
                try:
                    prefix = f"{col}_"
                    labels = [c[len(prefix):] for c in df_mlb.columns if c.startswith(prefix)]
                    if labels:
                        mlb.classes_ = np.array(labels)
                    else:
                        mlb.fit(out[col])
                except Exception:
                    mlb.fit(out[col])
        arr = mlb.transform(out[col])
        new_cols = [f"{col}_{c.strip().upper()}" for c in mlb.classes_]
        out = out.drop(columns=[c for c in new_cols if c in out.columns], errors='ignore')
        out = pd.concat([out, pd.DataFrame(arr, columns=new_cols, index=out.index)], axis=1)
    return out

def expand_feature_cols_for_training(base: pd.DataFrame, selected: list):
    cols = []
    for c in selected:
        if c in ('genres','day','network'):
            prefix = f"{c}_"
            cols += [k for k in base.columns if k.startswith(prefix)]
        else:
            cols.append(c)
    return cols

def build_X_from_selected(base: pd.DataFrame, selected: list) -> pd.DataFrame:
    X = pd.DataFrame(index=base.index)
    use_cols = expand_feature_cols_for_training(base, selected)
    if use_cols:
        X = pd.concat([X, base[use_cols]], axis=1)
    singles = [c for c in selected if c not in ('genres','day','network')]
    for c in singles:
        if c in base.columns and base[c].dtype == 'object':
            X = pd.concat([X, pd.get_dummies(base[c], prefix=c)], axis=1)
        elif c in base.columns:
            X[c] = base[c]
    return X

# ===== 데이터 로드 =====
@st.cache_data
def load_data():
    raw = pd.read_json('drama_d.json')
    if isinstance(raw, pd.Series):
        raw = pd.DataFrame({c: pd.Series(v) for c, v in raw.to_dict().items()})
    else:
        raw = pd.DataFrame({c: pd.Series(v) for c, v in raw.to_dict().items()})
    return raw

raw_df = load_data()

# ===== 멀티라벨 인코딩 결과 생성 (genres / day / network) =====
df_mlb = colab_multilabel_fit_transform(raw_df, cols=('genres','day','network'))

# ===== Colab 스타일 X/y, 전처리 정의 =====
drop_cols = [c for c in ['배우명','드라마명','genres','day','network','score','start airing'] if c in df_mlb.columns]

if 'score' in df_mlb.columns:
    df_mlb['score'] = pd.to_numeric(df_mlb['score'], errors='coerce')

X_colab_base = df_mlb.drop(columns=drop_cols, errors='ignore')
y_all = df_mlb['score']

categorical_features = [c for c in ['role','gender','air_q','married','age_group'] if c in X_colab_base.columns]

try:
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='(missing)')),
                ('ohe', ohe),
            ]),
            categorical_features
        )
    ],
    remainder='passthrough'
)

# ===== 공통 리스트 =====
def _flat_unique(series, cleaner=clean_cell_colab):
    return sorted({v for sub in series.dropna().apply(cleaner) for v in sub})

genre_list = _flat_unique(raw_df.get('genres', pd.Series(dtype=object)))
broadcaster_list = _flat_unique(raw_df.get('network', pd.Series(dtype=object)))
week_list = _flat_unique(raw_df.get('day', pd.Series(dtype=object)))
unique_genres = sorted(set(genre_list))

def age_to_age_group(age: int) -> str:
    s = raw_df.get('age_group')
    if s is None or s.dropna().empty:
        if age < 20: return "10대"
        if age < 30: return "20대"
        if age < 40: return "30대"
        if age < 50: return "40대"
        if age < 60: return "50대"
        return "60대 이상"

    series = s.dropna().astype(str)
    vocab = series.unique().tolist()
    counts = series.value_counts()
    decade = (int(age)//10)*10

    exact = [g for g in vocab if re.search(rf"{decade}\s*대", g)]
    if exact: return counts[exact].idxmax()

    loose = [g for g in vocab if str(decade) in g]
    if loose: return counts[loose].idxmax()

    if decade >= 60:
        over = [g for g in vocab if ('60' in g) or ('이상' in g)]
        if over: return counts[over].idxmax()

    with_num = []
    for g in vocab:
        m = re.search(r'(\d+)', g)
        if m: with_num.append((g, int(m.group(1))))
    if with_num:
        nearest_num = min(with_num, key=lambda t: abs(t[1]-decade))[1]
        candidates = [g for g,n in with_num if n==nearest_num]
        return counts[candidates].idxmax()

    return counts.idxmax()

# ===== 공통 파이프라인 빌더 =====
def make_pipeline(model_name, kind, estimator):
    if kind == "tree":
        return Pipeline([('preprocessor', preprocessor), ('model', estimator)])
    if model_name == "SVR":
        return Pipeline([('preprocessor', preprocessor), ('scaler', StandardScaler()), ('model', estimator)])
    if model_name == "KNN":
        return Pipeline([('preprocessor', preprocessor), ('poly', PolynomialFeatures(include_bias=False)),
                         ('scaler', StandardScaler()), ('knn', estimator)])
    if model_name == "Linear Regression (Poly)":
        return Pipeline([('preprocessor', preprocessor), ('poly', PolynomialFeatures(include_bias=False)),
                         ('scaler', StandardScaler()), ('linreg', estimator)])
    return Pipeline([('preprocessor', preprocessor), ('poly', PolynomialFeatures(include_bias=False)),
                     ('scaler', StandardScaler()), ('model', estimator)])

# ==============================
# 사이드바 네비게이션
# ==============================
st.title("💞 케미스코어")
MENU_ITEMS = ["🏁 개요","📋 기초통계","📈 분포·교차","🔧 필터","🗂 전체보기","🧪 튜닝","🤖 ML모델","🎯 예측"]
with st.sidebar:
    st.markdown("### 📂 메뉴")
    menu = st.radio("", MENU_ITEMS, index=0, key="nav_radio")
    st.markdown("<style>section[data-testid='stSidebar']{width:260px !important}</style>", unsafe_allow_html=True)
    # 모델 설정 (공통)
    st.header("🤖 모델 설정")
    test_size = 0.2
    st.caption("노트북 재현 모드: test_size=0.2, random_state=42")

# ==============================
# 페이지 함수들
# ==============================
def page_overview():
    st.header("데이터 개요")
    c1,c2,c3 = st.columns(3)
    c1.metric("샘플 수", raw_df.shape[0])
    c2.metric("컬럼 수", raw_df.shape[1])
    c3.metric("고유 장르", len(unique_genres))
    st.subheader("결측치 비율")
    st.dataframe(raw_df.isnull().mean())
    st.subheader("원본 샘플")
    st.dataframe(raw_df.head(), use_container_width=True)

def page_basic_stats():
    st.header("기초 통계: score")
    st.write(pd.to_numeric(raw_df['score'], errors='coerce').describe())
    fig,ax=plt.subplots(figsize=(6,3))
    ax.hist(pd.to_numeric(raw_df['score'], errors='coerce'), bins=20)
    ax.set_title("전체 평점 분포")
    st.pyplot(fig)

def page_dist_cross():
    st.header("분포 및 교차분석")

    # 연도별 주요 플랫폼 작품 수
    st.subheader("연도별 주요 플랫폼 작품 수")
    ct = (
        pd.DataFrame({'start airing': raw_df['start airing'],
                      'network': raw_df['network'].apply(clean_cell_colab)})
        .explode('network').groupby(['start airing','network']).size().reset_index(name='count')
    )
    ct['NETWORK_UP'] = ct['network'].astype(str).str.upper()
    focus = ['KBS','MBC','TVN','NETFLIX','SBS']
    fig3 = px.line(ct[ct['NETWORK_UP'].isin(focus)], x='start airing', y='count', color='network',
                   log_y=True, title="연도별 주요 플랫폼 작품 수")
    st.plotly_chart(fig3, use_container_width=True)

    p = (ct.pivot_table(index='start airing', columns='NETWORK_UP', values='count', aggfunc='sum')
           .fillna(0).astype(int))
    years = sorted(p.index)
    insights = []

    if 'NETFLIX' in p.columns:
        s = p['NETFLIX']; nz = s[s > 0]
        if not nz.empty:
            first_year = int(nz.index.min())
            max_year, max_val = int(s.idxmax()), int(s.max())
            txt = f"- **넷플릭스(OTT)의 급성장**: {first_year}년 이후 빠르게 증가, **{max_year}년 {max_val}편** 최고치."
            insights.append(txt)

    import numpy as np
    down_ter = []
    for b in ['KBS','MBC','SBS']:
        if b in p.columns and len(years) >= 2:
            slope = np.polyfit(years, p[b].reindex(years, fill_value=0), 1)[0]
            if slope < 0: down_ter.append(b)
    if down_ter:
        insights.append(f"- **지상파 감소 추세**: {' / '.join(down_ter)} 전반적 하락.")

    st.markdown("**인사이트**\n" + "\n".join(insights))

    # ===== 장르 개수별 배우 평균 평점 (배우 단위) =====
    st.subheader("장르 개수별 평균 평점 (배우 단위, 1~2 / 3~4 / 5~6 / 7+)")
    actor_col = '배우명' if '배우명' in raw_df.columns else ('actor' if 'actor' in raw_df.columns else None)
    if actor_col is None:
        st.info("배우 식별 컬럼을 찾을 수 없어(배우명/actor) 이 섹션을 건너뜁니다.")
    else:
        gdf = (
            pd.DataFrame({actor_col: raw_df[actor_col], 'genres': raw_df['genres'].apply(clean_cell_colab)})
            .explode('genres').dropna(subset=[actor_col,'genres'])
        )
        genre_cnt = gdf.groupby(actor_col)['genres'].nunique().rename('장르개수')
        actor_mean = (raw_df.groupby(actor_col, as_index=False)['score']
                      .mean().rename(columns={'score':'배우평균점수'}))
        df_actor = actor_mean.merge(genre_cnt.reset_index(), on=actor_col, how='left')
        df_actor['장르개수'] = df_actor['장르개수'].fillna(0).astype(int)
        df_actor = df_actor[df_actor['장르개수'] > 0].copy()

        def bucket(n: int) -> str:
            if n <= 2:  return '1~2개'
            if n <= 4:  return '3~4개'
            if n <= 6:  return '5~6개'
            return '7개 이상'

        df_actor['장르개수구간'] = pd.Categorical(
            df_actor['장르개수'].apply(bucket),
            categories=['1~2개','3~4개','5~6개','7개 이상'],
            ordered=True
        )

        fig_box = px.box(
            df_actor, x='장르개수구간', y='배우평균점수',
            category_orders={'장르개수구간': ['1~2개','3~4개','5~6개','7개 이상']},
            title="장르 개수별 배우 평균 점수 분포"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        stats = (df_actor.groupby('장르개수구간')['배우평균점수']
                 .agg(평균='mean', 중앙값='median', 표본수='count')
                 .reindex(['1~2개','3~4개','5~6개','7개 이상']).dropna(how='all').round(3))
        if not stats.empty and stats['표본수'].sum() > 0:
            st.markdown("**요약 통계(배우 단위)**")
            try: st.markdown(stats.to_markdown())
            except Exception: st.dataframe(stats.reset_index(), use_container_width=True)

    # ===== 결혼 상태별 평균 점수 =====
    st.subheader("주연 배우 결혼 상태별 평균 점수 비교")
    main_roles = raw_df[raw_df['role']=='주연'].copy()
    main_roles['결혼상태'] = main_roles['married'].apply(lambda x: '미혼' if x=='미혼' else '미혼 외')
    avg_scores_by_marriage = main_roles.groupby('결혼상태')['score'].mean()
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    bars = ax.bar(avg_scores_by_marriage.index, avg_scores_by_marriage.values, color=['mediumseagreen','gray'])
    for b in bars:
        yv = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, yv+0.005, f'{yv:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('주연 배우 결혼 상태별 평균 점수 비교'); ax.set_ylabel('평균 점수'); ax.set_xlabel('결혼 상태')
    ax.set_ylim(min(avg_scores_by_marriage.values)-0.05, max(avg_scores_by_marriage.values)+0.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)

    # ===== 장르별 작품 수 및 평균 점수 =====
    st.subheader("장르별 작품 수 및 평균 점수")
    dfg = raw_df.copy()
    dfg['genres'] = dfg['genres'].apply(clean_cell_colab)
    dfg = dfg.explode('genres').dropna(subset=['genres','score'])
    g_score = dfg.groupby('genres')['score'].mean().round(3)
    g_count = dfg['genres'].value_counts()
    gdf = pd.DataFrame({'평균 점수': g_score, '작품 수': g_count}).reset_index().rename(columns={'index':'장르','genres':'장르'})
    gdf = gdf.sort_values('작품 수', ascending=False).reset_index(drop=True)
    fig, ax1 = plt.subplots(figsize=(12,6))
    bars = ax1.bar(range(len(gdf)), gdf['작품 수'], color='lightgray')
    ax1.set_ylabel('작품 수'); ax1.set_xticks(range(len(gdf))); ax1.set_xticklabels(gdf['장르'], rotation=45, ha='right')
    for i, r in enumerate(bars):
        h = r.get_height()
        ax1.text(i, h+max(2, h*0.01), f'{int(h)}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#444')
    ax2 = ax1.twinx()
    ax2.plot(range(len(gdf)), gdf['평균 점수'], marker='o', linewidth=2)
    ax2.set_ylabel('평균 점수'); ax2.set_ylim(gdf['평균 점수'].min()-0.1, gdf['평균 점수'].max()+0.1)
    for i, v in enumerate(gdf['평균 점수']):
        ax2.text(i, v+0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.title('장르별 작품 수 및 평균 점수'); ax1.set_xlabel('장르'); ax1.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)

    # ===== 요일/년도 등 (요약) =====
    st.subheader("방영 요일별 작품 수 및 평균 점수 (월→일)")
    dfe = raw_df.copy(); dfe['day'] = dfe['day'].apply(clean_cell_colab)
    dfe = dfe.explode('day').dropna(subset=['day','score']).copy()
    dfe['day'] = dfe['day'].astype(str).str.strip().str.lower()
    ordered = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    day_ko = {'monday':'월','tuesday':'화','wednesday':'수','thursday':'목','friday':'금','saturday':'토','sunday':'일'}
    mean_by = dfe.groupby('day')['score'].mean().reindex(ordered)
    cnt_by = dfe['day'].value_counts().reindex(ordered).fillna(0).astype(int)
    fig, ax1 = plt.subplots(figsize=(12,6))
    bars = ax1.bar(ordered, cnt_by.values, alpha=0.3, color='tab:gray')
    ax1.set_ylabel('작품 수', color='tab:gray')
    for b in bars:
        h = b.get_height(); ax1.text(b.get_x()+b.get_width()/2, h+0.5, f'{int(h)}', ha='center', va='bottom', fontsize=9)
    ax2 = ax1.twinx(); ax2.plot(ordered, mean_by.values, marker='o')
    ax2.set_ylabel('평균 점수')
    if mean_by.notna().any(): ax2.set_ylim(mean_by.min()-0.05, mean_by.max()+0.05)
    for x, yv in zip(ordered, mean_by.values):
        if pd.notna(yv): ax2.text(x, yv+0.005
