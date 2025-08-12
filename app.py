# =========================================
# app.py  —  K-드라마 케미스코어 (Sparrow UI 스킨 + 사이드 네비)
# =========================================

# ---- page config MUST be first ----
import streamlit as st
st.set_page_config(page_title="케미스코어 | K-드라마 분석/예측", page_icon="🎬", layout="wide")

# ---- dependency guard (optional) ----
import importlib.util
_missing = [m for m in ("numpy","scipy","sklearn","joblib","threadpoolctl","xgboost") if importlib.util.find_spec(m) is None]
if _missing:
    st.error(f"필수 라이브러리 미설치: {_missing}. requirements.txt / runtime.txt 버전을 고정해 재배포하세요.")
    st.stop()

# ---- imports ----
import os, ast, random, re, platform
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import clone

# XGB가 설치돼 있으면 쓰도록 안전하게 추가
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ===== Global seed =====
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)

# ===== Matplotlib (한글 폰트) =====
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
    wanted = ("nanum","malgun","applegothic","notosanscjk","sourcehan","gulim","dotum","batang",
              "pretendard","gowun","spoqa")
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

# ====== Sparrow UI CSS ======
def _inject_sparrow_css():
    st.markdown("""
    <style>
      /* ---------- Layout / Typography ---------- */
      .block-container{padding-top:0.8rem; padding-bottom:2rem;}
      h1,h2,h3{font-weight:800;}
      /* ---------- Topbar ---------- */
      .topbar{display:flex; align-items:flex-end; justify-content:space-between; margin:6px 0 14px;}
      .topbar .title{font-size:28px; letter-spacing:-.2px; display:flex; gap:10px; align-items:center;}
      .crumb{font-size:12px; color:#6b7280; margin-top:4px;}
      .top-right{display:flex; gap:8px; align-items:center;}
      .chip{background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:6px 10px; font-size:12px;}

     # ... 기존 CSS 함수 안에서 Sidebar 관련 부분만 아래로 교체 ...
      /* ---------- Sidebar ---------- */
      section[data-testid="stSidebar"]{
        width:220px !important; min-width:220px;
        background:#0b1220; color:#e5e7eb; border-right:1px solid #070c16;
      }
      .sb-wrap{display:flex; flex-direction:column; height:100%;}
      .sb-brand{display:flex; align-items:center; gap:10px; padding:14px 12px 10px;}
      .sb-brand .logo{font-size:20px}
      .sb-brand .name{font-size:16px; font-weight:800; letter-spacing:.2px}

      .sb-menu{padding:6px 8px 8px; display:flex; flex-direction:column;}
      .sb-nav{margin:2px 0;}             /* 버튼 간격 최소화 */
      .sb-nav .stButton>button{
        width:100% !important;
        display:flex; align-items:center; gap:10px; justify-content:flex-start;
        background:transparent !important;
        color:#e5e7eb !important;
        border:1px solid #162033 !important;
        border-radius:10px !important;
        padding:8px 10px !important;
        font-size:14px !important;
        box-shadow:none !important;
        opacity:1 !important;            /* 희미해 보이는 문제 방지 */
      }
      .sb-nav .stButton>button:hover{
        background:#111a2b !important;
        border-color:#25324a !important;
      }
      .sb-nav.active .stButton>button{
        background:#2563eb !important;
        border-color:#2563eb !important;
        color:#ffffff !important;
      }

      .sb-card{background:#0f172a; border:1px solid #1f2937; border-radius:12px; padding:10px; margin-top:8px;}
      .sb-card h4{margin:0 0 6px 0; font-size:12px; color:#cbd5e1; font-weight:800;}
      .sb-footer{margin-top:auto; padding:10px 12px; font-size:11px; color:#9ca3af; border-top:1px solid #070c16;}


      /* ---------- Cards ---------- */
      .kpi-row{display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:8px 0 6px;}
      .kpi{
        background:#ffffff; border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px;
        box-shadow:0 6px 18px rgba(17,24,39,.04);
      }
      .kpi h6{margin:0 0 4px; font-size:12px; color:#6b7280; font-weight:800;}
      .kpi .v{font-size:22px; font-weight:800; line-height:1;}
      .kpi .d{font-size:12px; color:#10b981; font-weight:700;}

      /* Plot containers tighter top margin */
      div[data-testid="stPlotlyChart"], div.stPlot {margin-top:8px;}

      /* 메인 컨테이너 상단 여백 살짝 키워서 카드 잘림 방지 */
      .block-container{padding-top:1.4rem; padding-bottom:2.2rem;}
    
      /* KPI 줄과 다음 섹션 간 간격 */
      .kpi-row{ margin-bottom: 18px; }
    
      /* Plotly 차트 바깥쪽 여백 줄이기 + 기본 높이 */
      div[data-testid="stPlotlyChart"]{ margin-top:8px; }
    </style>
    """, unsafe_allow_html=True)

_inject_sparrow_css()

# ===== Data helpers =====
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

@st.cache_data
def load_data():
    raw = pd.read_json('drama_d.json')
    if isinstance(raw, pd.Series):
        raw = pd.DataFrame({c: pd.Series(v) for c, v in raw.to_dict().items()})
    else:
        raw = pd.DataFrame({c: pd.Series(v) for c, v in raw.to_dict().items()})
    return raw

raw_df = load_data()

# ===== Multi-label encoding base =====
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

df_mlb = colab_multilabel_fit_transform(raw_df, cols=('genres','day','network'))

# ===== Feature base =====
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
    transformers=[('cat',
                   Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='(missing)')),
                                   ('ohe', ohe)]),
                   categorical_features)],
    remainder='passthrough'
)

# ===== EDA lists =====
genre_list = [g for sub in raw_df.get('genres', pd.Series(dtype=object)).dropna().apply(clean_cell_colab) for g in sub]
broadcaster_list = [b for sub in raw_df.get('network', pd.Series(dtype=object)).dropna().apply(clean_cell_colab) for b in sub]
week_list = [w for sub in raw_df.get('day', pd.Series(dtype=object)).dropna().apply(clean_cell_colab) for w in sub]
unique_genres = sorted(set(genre_list))

# ===== Age-group helper =====
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
        nearest = min(with_num, key=lambda t: abs(t[1]-decade))[1]
        candidates = [g for g,n in with_num if n==nearest]
        return counts[candidates].idxmax()
    return counts.idxmax()

# =============================
# 네비게이션 정의 & 쿼리파람 동기화
# =============================
# 페이지 함수들은 아래에서 정의됩니다.
def _get_nav_from_query():
    if hasattr(st, "query_params"):  # Streamlit 1.30+
        val = st.query_params.get("nav", None)
        return val[0] if isinstance(val, list) else val
    else:
        qp = st.experimental_get_query_params()
        val = qp.get("nav", [None])
        return val[0]

def _set_nav_query(slug: str):
    if hasattr(st, "query_params"):
        st.query_params["nav"] = slug
    else:
        st.experimental_set_query_params(nav=slug)



# ---------- 각 페이지 ----------
def page_overview():
    total_titles = int(raw_df['드라마명'].nunique()) if '드라마명' in raw_df.columns else int(raw_df.shape[0])
    total_actors = int(raw_df['배우명'].nunique()) if '배우명' in raw_df.columns else \
                   (int(raw_df['actor'].nunique()) if 'actor' in raw_df.columns else int(raw_df.shape[0]))
    avg_score = float(pd.to_numeric(raw_df['score'], errors='coerce').mean())

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi"><h6>TOTAL TITLES</h6><div class="v">{total_titles}</div><div class="d">전체 작품</div></div>
      <div class="kpi"><h6>TOTAL ACTORS</h6><div class="v">{total_actors}</div><div class="d">명</div></div>
      <div class="kpi"><h6>AVG CHEMI SCORE</h6><div class="v">{0.0 if np.isnan(avg_score) else round(avg_score,2):.2f}</div><div class="d">전체 평균</div></div>
      <div class="kpi"><h6>GENRES</h6><div class="v">{len(unique_genres)}</div><div class="d">유니크</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("연도별 평균 케미스코어")
    df_year = raw_df.copy()
    df_year['start airing'] = pd.to_numeric(df_year['start airing'], errors='coerce')
    df_year['score'] = pd.to_numeric(df_year['score'], errors='coerce')
    df_year = df_year.dropna(subset=['start airing','score'])
    fig = px.line(df_year.groupby('start airing')['score'].mean().reset_index(),
                  x='start airing', y='score', markers=True)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("최근 연도 상위 작품 (중복 제거)")
        _df = raw_df.copy()
        _df['start airing'] = pd.to_numeric(_df['start airing'], errors='coerce')
        _df['score'] = pd.to_numeric(_df['score'], errors='coerce')
        _df = _df.dropna(subset=['start airing', 'score'])
    
        if not _df.empty:
            last_year = int(_df['start airing'].max())
            # 최근 1년 또는 2년 범위 (원하면 범위 조정 가능)
            recent = _df[_df['start airing'].between(last_year-1, last_year)]
    
            name_col = '드라마명' if '드라마명' in recent.columns else (
                'title' if 'title' in recent.columns else recent.columns[0]
            )
    
            # 드라마명 기준 중복 제거 (가장 높은 점수만 남김)
            recent_unique = (
                recent.sort_values('score', ascending=False)
                      .drop_duplicates(subset=[name_col], keep='first')
            )
    
            top_recent = recent_unique.sort_values('score', ascending=False).head(10)
            fig_recent = px.bar(top_recent, x=name_col, y='score', text='score')
            fig_recent.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_recent.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=40))
            st.plotly_chart(fig_recent, use_container_width=True)
        else:
            st.info("최근 연도 데이터가 없습니다.")

    # (오른쪽) 플랫폼별 작품 수 TOP 10
    with c2:
        st.subheader("플랫폼별 작품 수 (TOP 10)")
        _p = raw_df.copy()
        _p['network'] = _p['network'].apply(clean_cell_colab)
        _p = _p.explode('network').dropna(subset=['network'])
        p_cnt = (
                raw_df.assign(network=raw_df["network"].apply(clean_cell_colab))
                          .explode("network")
                          .dropna(subset=["network"])
                          .groupby("network")
                          .size()
                          .reset_index(name="count")  # 중복 방지
                )
        p_cnt = p_cnt.loc[:, ~p_cnt.columns.duplicated()].copy()

        fig_p = px.bar(p_cnt, x='network', y='count')
        fig_p.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=40))
        st.plotly_chart(fig_p, use_container_width=True)

def page_basic():
    st.header("기초 통계: score")
    st.write(pd.to_numeric(raw_df['score'], errors='coerce').describe())
    fig,ax=plt.subplots(figsize=(6,3))
    ax.hist(pd.to_numeric(raw_df['score'], errors='coerce'), bins=20)
    ax.set_title("전체 평점 분포")
    st.pyplot(fig)


def page_dist():
    st.header("분포 및 교차분석")
    st.subheader("연도별 주요 플랫폼 작품 수")
    ct = (
        pd.DataFrame({'start airing': raw_df['start airing'], 'network': raw_df['network'].apply(clean_cell_colab)})
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
            insights.append(f"- **넷플릭스(OTT)의 급성장**: {first_year}년 이후 증가, **{max_year}년 {max_val}편** 최고치.")
    down_ter = []
    for b in ['KBS','MBC','SBS']:
        if b in p.columns and len(years) >= 2:
            slope = np.polyfit(years, p[b].reindex(years, fill_value=0), 1)[0]
            if slope < 0: down_ter.append(b)
    if down_ter:
        insights.append(f"- **지상파 감소 추세**: {' / '.join(down_ter)} 전반적 하락.")
    st.markdown("**인사이트**\n" + "\n".join(insights))

    st.subheader("장르 개수별 평균 평점 (배우 단위)")
    actor_col = '배우명' if '배우명' in raw_df.columns else ('actor' if 'actor' in raw_df.columns else None)
    if actor_col is None:
        st.info("배우 식별 컬럼을 찾을 수 없어(배우명/actor) 이 섹션을 건너뜁니다.")
        return
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

def page_filter():
    st.header("실시간 필터")
    smin = float(pd.to_numeric(raw_df['score'], errors='coerce').min())
    smax = float(pd.to_numeric(raw_df['score'], errors='coerce').max())
    sfilter = st.slider("최소 평점", smin, smax, smin)
    y_min = int(pd.to_numeric(raw_df['start airing'], errors='coerce').min())
    y_max = int(pd.to_numeric(raw_df['start airing'], errors='coerce').max())
    yfilter = st.slider("방영년도 범위", y_min, y_max, (y_min, y_max))
    filt = raw_df[(pd.to_numeric(raw_df['score'], errors='coerce')>=sfilter) &
                  pd.to_numeric(raw_df['start airing'], errors='coerce').between(*yfilter)]
    st.dataframe(filt.head(20), use_container_width=True)

def page_all():
    st.header("원본 전체보기")
    st.dataframe(raw_df, use_container_width=True)

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

def page_tuning():
    st.header("GridSearchCV 튜닝")
    if "split_colab" not in st.session_state or st.session_state.get("split_key") != float(0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X_colab_base, y_all, test_size=0.2, random_state=SEED, shuffle=True
        )
        st.session_state["split_colab"] = (X_train, X_test, y_train, y_test)
        st.session_state["split_key"] = float(0.2)
    X_train, X_test, y_train, y_test = st.session_state["split_colab"]

    scoring = st.selectbox("스코어링", ["neg_root_mean_squared_error", "r2"], index=0)
    cv = st.number_input("CV 폴드 수", 3, 10, 5, 1)
    cv_shuffle = st.checkbox("CV 셔플(shuffle)", value=False)

    def render_param_selector(label, options):
        display_options, to_py = [], {}
        for v in options:
            if v is None: s="(None)"; to_py[s]=None
            else:
                s = str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
                to_py[s] = v
            display_options.append(s)
        sel = st.multiselect(f"{label}", display_options, default=display_options, key=f"sel_{label}")
        extra = st.text_input(f"{label} 추가값(콤마, 예: 50,75,100 또는 None)", value="", key=f"extra_{label}")
        chosen = [to_py[s] for s in sel]
        if extra.strip():
            for tok in extra.split(","):
                t = tok.strip()
                if not t: continue
                if t.lower()=="none": val=None
                else:
                    try: val=int(t)
                    except:
                        try: val=float(t)
                        except: val=t
                chosen.append(val)
        uniq=[];  [uniq.append(v) for v in chosen if v not in uniq]
        return uniq

    model_zoo = {
        "KNN": ("nonsparse", KNeighborsRegressor()),
        "Linear Regression (Poly)": ("nonsparse", LinearRegression()),
        "Ridge": ("nonsparse", Ridge()),
        "Lasso": ("nonsparse", Lasso()),
        "ElasticNet": ("nonsparse", ElasticNet(max_iter=10000)),
        "SGDRegressor": ("nonsparse", SGDRegressor(max_iter=10000, random_state=SEED)),
        "SVR": ("nonsparse", SVR()),
        "Decision Tree": ("tree", DecisionTreeRegressor(random_state=SEED)),
        "Random Forest": ("tree", RandomForestRegressor(random_state=SEED)),
    }
    if 'XGBRegressor' in globals() and XGB_AVAILABLE:
        model_zoo["XGBRegressor"] = ("tree", XGBRegressor(
            random_state=SEED, objective="reg:squarederror", n_jobs=-1, tree_method="hist"
        ))

    default_param_grids = {
        "KNN": {"poly__degree":[1,2,3], "knn__n_neighbors":[3,4,5,6,7,8,9,10]},
        "Linear Regression (Poly)": {"poly__degree":[1,2,3]},
        "Ridge": {"poly__degree":[1,2,3], "model__alpha":[0.001,0.01,0.1,1,10,100,1000]},
        "Lasso": {"poly__degree":[1,2,3], "model__alpha":[0.001,0.01,0.1,1,10,100,1000]},
        "ElasticNet": {"poly__degree":[1,2,3], "model__alpha":[0.001,0.01,0.1,1,10,100,1000], "model__l1_ratio":[0.1,0.5,0.9]},
        "SGDRegressor": {"poly__degree":[1,2,3], "model__learning_rate":["constant","invscaling","adaptive"]},
        "SVR": {"model__kernel":["poly","rbf","sigmoid"], "model__degree":[1,2,3]},
        "Decision Tree": {"model__max_depth":[10,15,20,25,30], "model__min_samples_split":[5,6,7,8,9,10],
                          "model__min_samples_leaf":[2,3,4,5], "model__max_leaf_nodes":[None,10,20,30]},
        "Random Forest": {"model__n_estimators":[100,200,300], "model__min_samples_split":[5,6,7,8,9,10],
                          "model__max_depth":[5,10,15,20,25,30]},
    }
    if "XGBRegressor" in model_zoo:
        default_param_grids["XGBRegressor"] = {
            "model__n_estimators":[200,400],
            "model__max_depth":[3,5,7],
            "model__learning_rate":[0.03,0.1,0.3],
            "model__subsample":[0.8,1.0],
            "model__colsample_bytree":[0.8,1.0],
        }

    model_name = st.selectbox("튜닝할 모델 선택", list(model_zoo.keys()), index=0)
    kind, estimator = model_zoo[model_name]
    pipe = make_pipeline(model_name, kind, estimator)

    st.markdown("**하이퍼파라미터 선택**")
    base_grid = default_param_grids.get(model_name, {})
    user_grid = {k: render_param_selector(k, v) for k, v in base_grid.items()}

    with st.expander("선택한 파라미터 확인"):
        st.write(user_grid)

    if st.button("GridSearch 실행"):
        cv_obj = KFold(n_splits=int(cv), shuffle=True, random_state=SEED) if cv_shuffle else int(cv)
        gs = GridSearchCV(estimator=pipe, param_grid=user_grid, cv=cv_obj,
                          scoring=scoring, n_jobs=-1, refit=True, return_train_score=True)
        with st.spinner("GridSearchCV 실행 중..."):
            gs.fit(X_train, y_train)

        st.subheader("베스트 결과")
        st.write("Best Params:", gs.best_params_)
        if scoring == "neg_root_mean_squared_error":
            st.write("Best CV RMSE (음수):", gs.best_score_)
        else:
            st.write(f"Best CV {scoring}:", gs.best_score_)

        y_pred_tr = gs.predict(X_train); y_pred_te = gs.predict(X_test)
        st.write("Train RMSE:", rmse(y_train, y_pred_tr))
        st.write("Test RMSE:", rmse(y_test, y_pred_te))
        st.write("Train R² Score:", r2_score(y_train, y_pred_tr))
        st.write("Test R² Score:", r2_score(y_test, y_pred_te))

        st.session_state["best_estimator"] = gs.best_estimator_
        st.session_state["best_params"] = gs.best_params_
        st.session_state["best_name"] = model_name
        st.session_state["best_cv_score"] = gs.best_score_
        st.session_state["best_scoring"] = scoring
        st.session_state["best_split_key"] = st.session_state.get("split_key")

        cvres = pd.DataFrame(gs.cv_results_)
        safe_cols = [c for c in ["rank_test_score","mean_test_score","std_test_score",
                                 "mean_train_score","std_train_score","params"] if c in cvres.columns]
        sorted_cvres = cvres.loc[:, safe_cols].sort_values("rank_test_score").reset_index(drop=True)
        st.dataframe(sorted_cvres, use_container_width=True)

    if model_name == "XGBRegressor" and not XGB_AVAILABLE:
        st.warning("xgboost가 설치되어 있지 않습니다. requirements.txt에 `xgboost`를 추가하고 재배포해 주세요.")

def page_ml():
    st.header("머신러닝 모델링")
    if "split_colab" not in st.session_state or st.session_state.get("split_key") != float(0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X_colab_base, y_all, test_size=0.2, random_state=SEED, shuffle=True
        )
        st.session_state["split_colab"] = (X_train, X_test, y_train, y_test)
        st.session_state["split_key"] = float(0.2)
    X_train, X_test, y_train, y_test = st.session_state["split_colab"]

    if "best_estimator" in st.session_state:
        model = st.session_state["best_estimator"]
        st.caption(f"현재 모델: GridSearch 베스트 모델 사용 ({st.session_state.get('best_name')})")
        if st.session_state.get("best_split_key") != st.session_state.get("split_key"):
            st.warning("주의: 베스트 모델은 이전 분할로 학습됨. 새 분할로 다시 튜닝해 주세요.", icon="⚠️")
    else:
        model = Pipeline([('preprocessor', preprocessor),
                          ('model', RandomForestRegressor(random_state=SEED))])
        model.fit(X_train, y_train)
        st.caption("현재 모델: 기본 RandomForest (미튜닝)")

    y_pred_tr = model.predict(X_train); y_pred_te = model.predict(X_test)
    st.metric("Train R²", f"{r2_score(y_train, y_pred_tr):.3f}")
    st.metric("Test  R²", f"{r2_score(y_test,  y_pred_te):.3f}")
    st.metric("Train RMSE", f"{rmse(y_train, y_pred_tr):.3f}")
    st.metric("Test  RMSE", f"{rmse(y_test,  y_pred_te):.3f}")

    if "best_params" in st.session_state:
        with st.expander("베스트 하이퍼파라미터 보기"):
            st.json(st.session_state["best_params"])

def page_predict():
    st.header("평점 예측")

    genre_opts   = sorted({g for sub in raw_df['genres'].dropna().apply(clean_cell_colab) for g in sub})
    week_opts    = sorted({d for sub in raw_df['day'].dropna().apply(clean_cell_colab) for d in sub})
    plat_opts    = sorted({p for sub in raw_df['network'].dropna().apply(clean_cell_colab) for p in sub})
    gender_opts  = sorted(raw_df['gender'].dropna().unique())
    role_opts    = sorted(raw_df['role'].dropna().unique())
    quarter_opts = sorted(raw_df['air_q'].dropna().unique())
    married_opts = sorted(raw_df['married'].dropna().unique())

    st.subheader("1) 입력")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**① 컨텐츠 특성**")
        input_age     = st.number_input("나이", 10, 80, 30)
        input_gender  = st.selectbox("성별", gender_opts) if gender_opts else st.text_input("성별 입력", "")
        input_role    = st.selectbox("역할", role_opts) if role_opts else st.text_input("역할 입력", "")
        input_married = st.selectbox("결혼여부", married_opts) if married_opts else st.text_input("결혼여부 입력", "")
        input_genre   = st.multiselect("장르 (멀티 선택)", genre_opts, default=genre_opts[:1] if genre_opts else [])
        derived_age_group = age_to_age_group(int(input_age))

        n_genre = len(input_genre)
        if n_genre == 0:  genre_bucket = "장르없음"
        elif n_genre <= 2: genre_bucket = "1~2개"
        elif n_genre <= 4: genre_bucket = "3~4개"
        elif n_genre <= 6: genre_bucket = "5~6개"
        else: genre_bucket = "7개 이상"
        st.caption(f"자동 연령대: **{derived_age_group}**  |  장르 개수: **{genre_bucket}**")

    with col_right:
        st.markdown("**② 편성 특성**")
        input_quarter = st.selectbox("방영분기", quarter_opts) if quarter_opts else st.text_input("방영분기 입력", "")
        input_week    = st.multiselect("방영요일 (멀티 선택)", week_opts, default=week_opts[:1] if week_opts else [])
        input_plat    = st.multiselect("플랫폼 (멀티 선택)", plat_opts, default=plat_opts[:1] if plat_opts else [])
        age_group_candidates = ["10대", "20대", "30대", "40대", "50대", "60대 이상"]
        data_age_groups = sorted(set(str(x) for x in raw_df.get("age_group", pd.Series([], dtype=object)).dropna().unique()))
        opts_age_group = data_age_groups if data_age_groups else age_group_candidates
        safe_index = 0 if not opts_age_group else min(1, len(opts_age_group)-1)
        target_age_group = st.selectbox("🎯 타깃 시청자 연령대",
                                        options=opts_age_group if opts_age_group else ["(데이터 없음)"],
                                        index=safe_index,
                                        key="target_age_group_main")
        st.session_state["target_age_group"] = target_age_group
        st.session_state["actor_age"] = int(input_age)
        predict_btn = st.button("예측 실행")

    if not predict_btn:
        return

    if "best_estimator" in st.session_state:
        model_full = clone(st.session_state["best_estimator"])
        st.caption(f"예측 모델: GridSearch 베스트 재학습 사용 ({st.session_state.get('best_name')})")
    else:
        model_full = Pipeline([('preprocessor', preprocessor),
                               ('model', RandomForestRegressor(n_estimators=100, random_state=SEED))])
        st.caption("예측 모델: 기본 RandomForest (미튜닝)")
    model_full.fit(X_colab_base, y_all)

    user_raw = pd.DataFrame([{
        'age': int(input_age), 'gender': input_gender, 'role': input_role, 'married': input_married,
        'air_q': input_quarter, 'age_group': derived_age_group,
        'genres': input_genre, 'day': input_week, 'network': input_plat, '장르구분': genre_bucket,
    }])

    def _build_user_base(df_raw: pd.DataFrame) -> pd.DataFrame:
        _user_mlb = colab_multilabel_transform(df_raw, cols=('genres','day','network'))
        _base = pd.concat([X_colab_base.iloc[:0].copy(), _user_mlb], ignore_index=True)
        _base = _base.drop(columns=[c for c in drop_cols if c in _base.columns], errors='ignore')
        for c in X_colab_base.columns:
            if c not in _base.columns:
                _base[c] = 0
        _base = _base[X_colab_base.columns].tail(1)
        num_cols_ = X_colab_base.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols_) > 0:
            _base[num_cols_] = _base[num_cols_].apply(pd.to_numeric, errors="coerce")
            _base[num_cols_] = _base[num_cols_].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return _base

    user_base = _build_user_base(user_raw)
    pred = float(model_full.predict(user_base)[0])
    st.success(f"💡 예상 평점: {pred:.2f}")

# ================== 사이드바 (네비 + 설정) ==================
NAV_ITEMS = [
    ("overview", "🏠", "개요",        page_overview),
    ("basic",    "📋", "기초통계",    page_basic),
    ("dist",     "📈", "분포/교차",   page_dist),
    ("filter",   "🛠️", "필터",        page_filter),
    ("all",      "🗂️", "전체보기",    page_all),
    ("tuning",   "🧪", "튜닝",        page_tuning),
    ("ml",       "🤖", "ML모델",      page_ml),
    ("predict",  "🎯", "예측",        page_predict),
]

if "nav" not in st.session_state:
    st.session_state["nav"] = _get_nav_from_query() or "overview"
current = st.session_state["nav"]

with st.sidebar:
    st.markdown('<div class="sb-wrap">', unsafe_allow_html=True)
    st.markdown("""
<style>
.sb-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 900;
}
.sb-brand .logo {
    font-size: 55px;  /* 이모티콘 크기 */
}
.sb-brand .name {
    font-size: 50px;  /* 글자 크기 */
}
</style>
<div class="sb-brand">
    <span class="logo">🎬</span>
    <span class="name">케미스코어</span>
</div>
""", unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="sb-menu">', unsafe_allow_html=True)
    for slug, icon, label, _fn in NAV_ITEMS:
        active = (slug == current)
        st.markdown(f'<div class="sb-nav {"active" if active else ""}">', unsafe_allow_html=True)
        if st.button(f"{icon}  {label}", key=f"nav_{slug}", use_container_width=True):
            st.session_state["nav"] = slug
            _set_nav_query(slug)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    # Card: model config
    st.markdown('<div class="sb-card"><h4>모델 설정: \ntest_size=0.2, random_state=42</h4>', unsafe_allow_html=True)
    test_size = 0.2
    st.markdown('</div>', unsafe_allow_html=True)  # /sb-card
    st.markdown('</div>', unsafe_allow_html=True)  # /sb-menu
    st.markdown('<div class="sb-footer">© Chemiscore • <span class="ver">v0.1</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # /sb-wrap

# ================== 라우팅 ==================
PAGES = {slug: fn for slug, _, _, fn in NAV_ITEMS}
PAGES.get(st.session_state["nav"], page_overview)()
