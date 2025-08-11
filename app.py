# app.py
import os
import ast
import random
import numpy as np
import pandas as pd
from pathlib import Path
import platform
from sklearn.metrics import mean_squared_error
import streamlit as st
import plotly.express as px

#XGB가 설치돼 있으면 쓰도록 안전하게 추가
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def rmse(y_true, y_pred):
    # squared=False 미지원 환경에서도 동작
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
# ===== 페이지 설정 =====
st.set_page_config(page_title="K-드라마 분석/예측", page_icon="🎬", layout="wide")

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
    wanted = ("nanum","malgun","applegothic","notosanscjk","sourcehan","gulim","dotum","batang","pretendard","gowun","spoqa")
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
from sklearn.metrics import r2_score, mean_squared_error
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

def colab_multilabel_fit_transform(df: pd.DataFrame, cols=('장르','방영요일','플랫폼')) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].apply(clean_cell_colab)
        mlb = MultiLabelBinarizer()
        arr = mlb.fit_transform(out[col])
        new_cols = [f"{col}_{c.strip().upper()}" for c in mlb.classes_]
        out = out.drop(columns=[c for c in new_cols if c in out.columns], errors='ignore')
        out = pd.concat([out, pd.DataFrame(arr, columns=new_cols, index=out.index)], axis=1)
        st.session_state[f"mlb_classes_{col}"] = mlb.classes_.tolist()
    return out

def colab_multilabel_transform(df: pd.DataFrame, cols=('장르','방영요일','플랫폼')) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].apply(clean_cell_colab)
        classes = st.session_state.get(f"mlb_classes_{col}", [])
        mlb = MultiLabelBinarizer(classes=classes)
        arr = mlb.transform(out[col])
        new_cols = [f"{col}_{c.strip().upper()}" for c in mlb.classes_]
        out = out.drop(columns=[c for c in new_cols if c in out.columns], errors='ignore')
        out = pd.concat([out, pd.DataFrame(arr, columns=new_cols, index=out.index)], axis=1)
    return out

# (선택) 사용자 선택 기반 X 생성 유틸 — EDA/예측 탭에서 사용
def expand_feature_cols_for_training(base: pd.DataFrame, selected: list):
    cols = []
    for c in selected:
        if c in ('장르','방영요일','플랫폼'):
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
    singles = [c for c in selected if c not in ('장르','방영요일','플랫폼')]
    for c in singles:
        if c in base.columns and base[c].dtype == 'object':
            X = pd.concat([X, pd.get_dummies(base[c], prefix=c)], axis=1)
        elif c in base.columns:
            X[c] = base[c]
    return X

# ===== 데이터 로드 =====
@st.cache_data
def load_data():
    raw = pd.read_json('drama_data.json')
    return pd.DataFrame({c: pd.Series(v) for c,v in raw.items()})

raw_df = load_data()

# ===== Colab과 동일 멀티라벨 인코딩 결과 생성 =====
df_mlb = colab_multilabel_fit_transform(raw_df, cols=('장르','방영요일','플랫폼'))
df_mlb['점수'] = pd.to_numeric(df_mlb['점수'], errors='coerce')

# ===== Colab 스타일 X/y, 전처리 정의 =====
drop_cols = [c for c in ['배우명','드라마명','장르','방영요일','플랫폼','점수','방영년도'] if c in df_mlb.columns]
X_colab_base = df_mlb.drop(columns=drop_cols)
y_all = df_mlb['점수']
categorical_features = [c for c in ['역할','성별','방영분기','결혼여부','연령대'] if c in X_colab_base.columns]
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# ===== EDA용 리스트 =====
genre_list = [g for sub in raw_df['장르'].dropna().apply(clean_cell_colab) for g in sub]
broadcaster_list = [b for sub in raw_df['플랫폼'].dropna().apply(clean_cell_colab) for b in sub]
week_list = [w for sub in raw_df['방영요일'].dropna().apply(clean_cell_colab) for w in sub]
unique_genres = sorted(set(genre_list))

# ===== 사이드바 =====
with st.sidebar:
    st.header("🤖 모델 설정")
    # 노트북과 동일하게 기본값 0.3으로 설정
    test_size = st.slider('테스트셋 비율', 0.1, 0.5, 0.3, 0.05)
    feature_cols = st.multiselect(
        '특성 선택(예측 탭용)',
        ['나이','방영년도','성별','장르','배우명','플랫폼','결혼여부'],
        default=['나이','방영년도','장르']
    )

# ===== 탭 구성 =====
tabs = st.tabs(["🗂개요","📊기초통계","📈분포/교차","💬워드클라우드","⚙️필터","🔍전체보기","🤖ML모델","🔧튜닝","🎯예측"])

# --- 4.1 데이터 개요 ---
with tabs[0]:
    st.header("데이터 개요")
    c1,c2,c3 = st.columns(3)
    c1.metric("샘플 수", raw_df.shape[0])
    c2.metric("컬럼 수", raw_df.shape[1])
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
    st.subheader("전체 평점 분포")
    fig1 = px.histogram(raw_df, x='점수', nbins=20, title="전체 평점 분포")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Top 10 평점 작품")
    top10 = raw_df.nlargest(10, '점수')[['드라마명','점수']].sort_values('점수')
    fig2 = px.bar(top10, x='점수', y='드라마명', orientation='h', text='점수', title="Top 10 평점 작품")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("연도별 주요 플랫폼 작품 수 (로그 스케일)")
    ct = (
        pd.DataFrame({'방영년도': raw_df['방영년도'], '플랫폼': raw_df['플랫폼'].apply(clean_cell_colab)})
        .explode('플랫폼').groupby(['방영년도','플랫폼']).size().reset_index(name='count')
    )
    ct['플랫폼_up'] = ct['플랫폼'].str.upper()
    focus = ['KBS','MBC','TVN','NETFLIX','SBS']
    fig3 = px.line(ct[ct['플랫폼_up'].isin(focus)], x='방영년도', y='count', color='플랫폼',
                   log_y=True, title="연도별 주요 플랫폼 작품 수 (로그 스케일)")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("멀티장르 vs 단일장르 평균 평점 (배우 단위 박스플롯)")
    ag = (
        pd.DataFrame({'배우명': raw_df['배우명'], '장르': raw_df['장르'].apply(clean_cell_colab)})
        .explode('장르').groupby('배우명')['장르'].nunique()
    )
    multi_set = set(ag[ag > 1].index)
    label_map = {name: ('멀티장르' if name in multi_set else '단일장르') for name in ag.index}
    actor_mean = raw_df.groupby('배우명', as_index=False)['점수'].mean().rename(columns={'점수':'배우평균점수'})
    actor_mean['장르구분'] = actor_mean['배우명'].map(label_map)
    fig_box = px.box(actor_mean, x='장르구분', y='배우평균점수', title="멀티장르 vs 단일장르 배우 단위 평균 점수 분포")
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("주연 배우 결혼 상태별 평균 점수 비교")
    main_roles = raw_df[raw_df['역할']=='주연'].copy()
    main_roles['결혼상태'] = main_roles['결혼여부'].apply(lambda x: '미혼' if x=='미혼' else '미혼 외')
    avg_scores_by_marriage = main_roles.groupby('결혼상태')['점수'].mean()
    fig, ax = plt.subplots(figsize=(6,5))
    bars = ax.bar(avg_scores_by_marriage.index, avg_scores_by_marriage.values, color=['mediumseagreen','gray'])
    for b in bars:
        yv = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, yv+0.005, f'{yv:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title('주연 배우 결혼 상태별 평균 점수 비교'); ax.set_ylabel('평균 점수'); ax.set_xlabel('결혼 상태')
    ax.set_ylim(min(avg_scores_by_marriage.values)-0.05, max(avg_scores_by_marriage.values)+0.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)

    st.subheader("장르별 작품 수 및 평균 점수")
    dfg = raw_df.copy(); dfg['장르'] = dfg['장르'].apply(clean_cell_colab)
    dfg = dfg.explode('장르').dropna(subset=['장르','점수'])
    g_score = dfg.groupby('장르')['점수'].mean().round(3)
    g_count = dfg['장르'].value_counts()
    gdf = (pd.DataFrame({'평균 점수': g_score, '작품 수': g_count}).reset_index().rename(columns={'index':'장르'}))
    gdf = gdf.sort_values('작품 수', ascending=False).reset_index(drop=True)
    fig, ax1 = plt.subplots(figsize=(12,6))
    bars = ax1.bar(range(len(gdf)), gdf['작품 수'], color='lightgray')
    ax1.set_ylabel('작품 수'); ax1.set_xticks(range(len(gdf))); ax1.set_xticklabels(gdf['장르'], rotation=45, ha='right')
    for i, r in enumerate(bars):
        h = r.get_height(); ax1.text(i, h+max(2, h*0.01), f'{int(h)}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#444')
    ax2 = ax1.twinx(); ax2.plot(range(len(gdf)), gdf['평균 점수'], marker='o', linewidth=2, color='tab:blue')
    ax2.set_ylabel('평균 점수', color='tab:blue'); ax2.tick_params(axis='y', colors='tab:blue')
    ax2.set_ylim(gdf['평균 점수'].min()-0.1, gdf['평균 점수'].max()+0.1)
    for i, v in enumerate(gdf['평균 점수']):
        ax2.text(i, v+0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='tab:blue')
    plt.title('장르별 작품 수 및 평균 점수'); ax1.set_xlabel('장르'); ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout(); st.pyplot(fig)

    st.subheader("방영 요일별 작품 수 및 평균 점수 (월→일)")
    dfe = raw_df.copy(); dfe['방영요일'] = dfe['방영요일'].apply(clean_cell_colab)
    dfe = dfe.explode('방영요일').dropna(subset=['방영요일','점수']).copy()
    dfe['방영요일'] = dfe['방영요일'].astype(str).str.strip().str.lower()
    ordered = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    day_ko = {'monday':'월','tuesday':'화','wednesday':'수','thursday':'목','friday':'금','saturday':'토','sunday':'일'}
    mean_by = dfe.groupby('방영요일')['점수'].mean().reindex(ordered)
    cnt_by = dfe['방영요일'].value_counts().reindex(ordered).fillna(0).astype(int)
    fig, ax1 = plt.subplots(figsize=(10,6))
    bars = ax1.bar(ordered, cnt_by.values, alpha=0.3, color='tab:gray')
    ax1.set_ylabel('작품 수', color='tab:gray'); ax1.tick_params(axis='y', labelcolor='tab:gray')
    for b in bars:
        h = b.get_height(); ax1.text(b.get_x()+b.get_width()/2, h+0.5, f'{int(h)}', ha='center', va='bottom', fontsize=9, color='black')
    ax2 = ax1.twinx(); ax2.plot(ordered, mean_by.values, marker='o', color='tab:blue')
    ax2.set_ylabel('평균 점수', color='tab:blue'); ax2.tick_params(axis='y', labelcolor='tab:blue')
    if mean_by.notna().any(): ax2.set_ylim(mean_by.min()-0.05, mean_by.max()+0.05)
    for x, yv in zip(ordered, mean_by.values):
        if pd.notna(yv): ax2.text(x, yv+0.005, f'{yv:.3f}', color='tab:blue', fontsize=9, ha='center')
    ax1.set_xticks(ordered); ax1.set_xticklabels([day_ko[d] for d in ordered])
    plt.title('방영 요일별 작품 수 및 평균 점수 (월요일 → 일요일 순)'); plt.tight_layout(); st.pyplot(fig)

# --- 4.4 워드클라우드 ---
from wordcloud import WordCloud
with tabs[3]:
    st.header("워드클라우드")
    font_path = st.session_state.get("kfont_path")
    if genre_list:
        wc = WordCloud(width=800,height=400,background_color='white',font_path=font_path).generate(' '.join(genre_list))
        fig,ax=plt.subplots(); ax.imshow(wc,interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)
    if broadcaster_list:
        wc = WordCloud(width=800,height=400,background_color='white',font_path=font_path).generate(' '.join(broadcaster_list))
        fig,ax=plt.subplots(); ax.imshow(wc,interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)
    if week_list:
        wc = WordCloud(width=800,height=400,background_color='white',font_path=font_path).generate(' '.join(week_list))
        fig,ax=plt.subplots(); ax.imshow(wc,interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)

# --- 4.5 실시간 필터 ---
with tabs[4]:
    st.header("실시간 필터")
    smin,smax = float(raw_df['점수'].min()), float(raw_df['점수'].max())
    sfilter = st.slider("최소 평점", smin,smax,smin)
    yfilter = st.slider("방영년도 범위", int(raw_df['방영년도'].min()), int(raw_df['방영년도'].max()), (2000,2025))
    filt = raw_df[(raw_df['점수']>=sfilter) & raw_df['방영년도'].between(*yfilter)]
    st.dataframe(filt.head(20))

# --- 4.6 전체 미리보기 ---
with tabs[5]:
    st.header("원본 전체보기")
    st.dataframe(raw_df, use_container_width=True)

# --- 4.7 머신러닝 모델링 (Colab 설정 그대로) ---
with tabs[6]:
    st.header("머신러닝 모델링 (Colab 설정)")
    # split 고정
    if "split_colab" not in st.session_state or st.session_state.get("split_key") != float(test_size):
        X_train, X_test, y_train, y_test = train_test_split(X_colab_base, y_all, test_size=test_size, random_state=SEED, shuffle=True)
        st.session_state["split_colab"] = (X_train, X_test, y_train, y_test)
        st.session_state["split_key"] = float(test_size)
    X_train, X_test, y_train, y_test = st.session_state["split_colab"]

    rf_pipe = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=SEED))])
    rf_pipe.fit(X_train, y_train)
    y_pred_tr = rf_pipe.predict(X_train)
    y_pred_te = rf_pipe.predict(X_test)

    st.metric("Train R²", f"{r2_score(y_train,y_pred_tr):.3f}")
    st.metric("Test  R²", f"{r2_score(y_test,y_pred_te):.3f}")
    st.metric("Train RMSE", f"{rmse(y_train,y_pred_tr):.3f}")
    st.metric("Test  RMSE", f"{rmse(y_test,y_pred_te):.3f}")

# --- 4.8 GridSearch 튜닝 (RandomForest, Colab 그리드) ---
with tabs[7]:
    st.header("GridSearchCV 튜닝")
    if "split_colab" not in st.session_state:
        st.info("먼저 '머신러닝 모델링 (Colab 설정)' 탭을 한 번 실행해 주세요.")
    else:
        X_train, X_test, y_train, y_test = st.session_state["split_colab"]

        scoring = st.selectbox("스코어링", ["neg_root_mean_squared_error", "r2"], index=0)
        cv = st.number_input("CV 폴드 수", 3, 10, 5, 1)

        # 1) 모델 풀
        model_zoo = {
            "KNN": ("nonsparse", KNeighborsRegressor()),
            "Linear Regression (Poly)": ("nonsparse", LinearRegression()),
            "Ridge": ("nonsparse", Ridge()),
            "Lasso": ("nonsparse", Lasso()),
            "ElasticNet": ("nonsparse", ElasticNet(max_iter=10000)),
            "SGDRegressor": ("nonsparse", SGDRegressor(max_iter=10000)),
            "SVR": ("nonsparse", SVR()),
            "Decision Tree": ("tree", DecisionTreeRegressor(random_state=SEED)),
            "Random Forest": ("tree", RandomForestRegressor(random_state=SEED)),
        }
        if 'XGBRegressor' in globals() and XGB_AVAILABLE:
            model_zoo["XGBRegressor"] = ("tree", XGBRegressor(
                random_state=SEED,
                objective="reg:squarederror",
                n_jobs=-1,
                tree_method="hist"
            ))

        # 2) 파이프라인 빌더
        # - 'nonsparse'는 poly+scaler(+preprocessor) 사용
        #   (OneHot 결과가 희소일 수 있어 scaler는 with_mean=False)
        def make_pipeline(kind, estimator):
            if kind == "tree":
                return Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', estimator),
                ])
            else:
                return Pipeline([
                    ('preprocessor', preprocessor),
                    ('poly', PolynomialFeatures(include_bias=False)),
                    ('scaler', StandardScaler(with_mean=False)),
                    ('model', estimator),
                ])

        # 3) 파라미터 그리드
        param_grids = {
            "KNN": {
                "poly__degree": [1, 2, 3],
                "model__n_neighbors": [3,4,5,6,7,8,9,10],
            },
            "Linear Regression (Poly)": {
                "poly__degree": [1, 2, 3],
            },
            "Ridge": {
                "poly__degree": [1, 2, 3],
                "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            },
            "Lasso": {
                "poly__degree": [1, 2, 3],
                "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            },
            "ElasticNet": {
                "poly__degree": [1, 2, 3],
                "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                "model__l1_ratio": [0.1, 0.5, 0.9],
            },
            "SGDRegressor": {
                "poly__degree": [1, 2, 3],
                "model__learning_rate": ["constant", "invscaling", "adaptive"],
                # 필요 시: "model__eta0": [0.001, 0.01, 0.1]
            },
            "SVR": {
                "poly__degree": [1, 2, 3],  # poly 커널일 때만 의미, 같이 둬도 OK
                "model__kernel": ["poly", "rbf", "sigmoid"],
                "model__degree": [1, 2, 3],
            },
            "Decision Tree": {
                "model__max_depth": [10, 15, 20, 25, 30],
                "model__min_samples_split": [5, 6, 7, 8, 9, 10],
                "model__min_samples_leaf": [2, 3, 4, 5],
                "model__max_leaf_nodes": [None, 10, 20, 30],
            },
            "Random Forest": {
                "model__n_estimators": [100, 200, 300],
                "model__min_samples_split": [5, 6, 7, 8, 9, 10],
                "model__max_depth": [5, 10, 15, 20, 25, 30],
            },
        }
        if "XGBRegressor" in model_zoo:
            param_grids["XGBRegressor"] = {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.03, 0.1, 0.3],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            }

        # 4) 실행
        model_name = st.selectbox("튜닝할 모델 선택", list(model_zoo.keys()), index=0)
        kind, estimator = model_zoo[model_name]
        pipe = make_pipeline(kind, estimator)
        grid = param_grids[model_name]

        if st.button("GridSearch 실행"):
            gs = GridSearchCV(
                pipe, grid,
                cv=int(cv), scoring=scoring, n_jobs=-1,
                refit=True, return_train_score=True
            )
            with st.spinner("GridSearchCV 실행 중..."):
                gs.fit(X_train, y_train)

            st.subheader("베스트 결과")
            st.json(gs.best_params_)
            if scoring == "neg_root_mean_squared_error":
                st.write(f"Best CV RMSE: {-gs.best_score_:.6f}")
            else:
                st.write(f"Best CV {scoring}: {gs.best_score_:.6f}")

            y_pred = gs.predict(X_test)
            st.write(f"Test RMSE: {rmse(y_test, y_pred):.6f}")
            st.write(f"Test R²  : {r2_score(y_test, y_pred):.6f}")

            cvres = pd.DataFrame(gs.cv_results_)
            cols = ["rank_test_score","mean_test_score","std_test_score","mean_train_score","std_train_score","params"]
            st.dataframe(cvres[cols].sort_values("rank_test_score").reset_index(drop=True))

        # 설치가 안 되어 있을 때 XGB 알림
        if model_name == "XGBRegressor" and not XGB_AVAILABLE:
            st.warning("xgboost가 설치되어 있지 않습니다. requirements.txt에 `xgboost`를 추가하고 재배포해 주세요.")

# --- 4.9 예측 실행 (선택형 유틸 사용) ---
with tabs[8]:
    st.header("평점 예측")
    st.subheader("1) 입력")
    genre_opts    = sorted({g for sub in raw_df['장르'].dropna().apply(clean_cell_colab) for g in sub})
    plat_opts     = sorted({p for sub in raw_df['플랫폼'].dropna().apply(clean_cell_colab) for p in sub})
    actor_opts    = sorted(raw_df['배우명'].dropna().unique())
    gender_opts   = sorted(raw_df['성별'].dropna().unique())
    married_opts  = sorted(raw_df['결혼여부'].dropna().unique())

    input_age     = st.number_input("배우 나이", 10, 80, 30)
    input_year    = st.number_input("방영년도", 2000, 2025, 2021)
    input_gender  = st.selectbox("성별", gender_opts) if gender_opts else st.text_input("성별 입력", "")
    input_genre   = st.multiselect("장르", genre_opts, default=genre_opts[:1] if genre_opts else [])
    input_actor   = st.selectbox("배우명", actor_opts) if actor_opts else st.text_input("배우명 입력", "")
    input_plat    = st.multiselect("플랫폼", plat_opts, default=plat_opts[:1] if plat_opts else [])
    input_married = st.selectbox("결혼여부", married_opts) if married_opts else st.text_input("결혼여부 입력", "")
    predict_btn   = st.button("예측 실행")

    if predict_btn:
        # Colab 파이프라인으로 전체 학습
        rf_pipe_full = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=SEED))])
        rf_pipe_full.fit(X_colab_base, y_all)

        # 사용자 입력 1행
        user_raw = pd.DataFrame([{
            '나이': input_age, '방영년도': input_year, '성별': input_gender,
            '장르': input_genre, '배우명': input_actor, '플랫폼': input_plat, '결혼여부': input_married
        }])
        user_mlb = colab_multilabel_transform(user_raw, cols=('장르','방영요일','플랫폼'))

        # Colab X 스키마와 정합(드랍 리스트 동일 적용)
        user_base = pd.concat([X_colab_base.iloc[:0].copy(), user_mlb], ignore_index=True)
        user_base = user_base.drop(columns=[c for c in drop_cols if c in user_base.columns], errors='ignore')
        user_base = user_base.tail(1)

        pred = rf_pipe_full.predict(user_base)[0]
        st.success(f"💡 예상 평점: {pred:.2f}")
