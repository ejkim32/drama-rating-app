# app.py
import os
import ast
import random
import numpy as np
import pandas as pd
from pathlib import Path
import platform

import streamlit as st
import plotly.express as px

# ===== 페이지 설정(가장 처음 st.* 호출) =====
st.set_page_config(page_title="K-드라마 분석/예측", page_icon="🎬", layout="wide")

# ===== 전역: 시드 고정(재현성) =====
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ===== 전역: Matplotlib/WordCloud 한글 폰트 부트스트랩 =====
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 캐시는 한 번만 삭제(새 폰트 인식)
if st.session_state.get("font_cache_cleared") is not True:
    import shutil
    shutil.rmtree(matplotlib.get_cachedir(), ignore_errors=True)
    st.session_state["font_cache_cleared"] = True

def ensure_korean_font():
    """Matplotlib + WordCloud용 한글 폰트 세팅 (로컬/클라우드/Windows 모두 안전)"""
    matplotlib.rcParams['axes.unicode_minus'] = False
    base = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()

    candidates = [
        base / "fonts" / "NanumGothic.ttf",
        base / "fonts" / "NanumGothic-Regular.ttf",
    ]
    if platform.system() == "Windows":
        candidates += [
            Path(r"C:\Windows\Fonts\malgun.ttf"),
            Path(r"C:\Windows\Fonts\malgunbd.ttf"),
        ]
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
                st.session_state["kfont_path"] = str(p)  # WordCloud에서 사용
                return family
        except Exception:
            continue
    st.session_state["kfont_path"] = None
    return None

_ = ensure_korean_font()

# ===== Colab과 동일한 멀티라벨 전처리 =====
from sklearn.preprocessing import MultiLabelBinarizer

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

    # 단일 범주형(성별/결혼여부/배우명 등) 원핫
    singles = [c for c in selected if c not in ('장르','방영요일','플랫폼')]
    for c in singles:
        if c in base.columns and base[c].dtype == 'object':
            d = pd.get_dummies(base[c], prefix=c)
            X = pd.concat([X, d], axis=1)
        elif c in base.columns:
            X[c] = base[c]
    return X

# ===== 데이터 로드 =====
@st.cache_data
def load_data():
    raw = pd.read_json('drama_data.json')
    return pd.DataFrame({c: pd.Series(v) for c,v in raw.items()})

raw_df = load_data()

# Colab과 동일 인코딩 결과(모델용 DF)
df_mlb = colab_multilabel_fit_transform(raw_df, cols=('장르','방영요일','플랫폼'))
df_mlb['점수'] = pd.to_numeric(df_mlb['점수'], errors='coerce')
y_all = df_mlb['점수']

# ===== EDA용 리스트 준비 =====
genre_list = [g for sub in raw_df['장르'].dropna().apply(clean_cell_colab) for g in sub]
broadcaster_list = [b for sub in raw_df['플랫폼'].dropna().apply(clean_cell_colab) for b in sub]
week_list = [w for sub in raw_df['방영요일'].dropna().apply(clean_cell_colab) for w in sub]
unique_genres = sorted(set(genre_list))

# ===== 사이드바 =====
with st.sidebar:
    st.header("🤖 모델 설정")
    model_type   = st.selectbox('모델 선택', ['Random Forest','Linear Regression'])
    test_size    = st.slider('테스트셋 비율', 0.1, 0.5, 0.2, 0.05)
    feature_cols = st.multiselect(
        '특성 선택',
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

    # 1) 전체 평점 분포
    st.subheader("전체 평점 분포")
    fig1 = px.histogram(raw_df, x='점수', nbins=20, title="전체 평점 분포")
    st.plotly_chart(fig1, use_container_width=True)

    # 2) Top 10 평점 작품
    st.subheader("Top 10 평점 작품")
    top10 = raw_df.nlargest(10, '점수')[['드라마명','점수']].sort_values('점수')
    fig2 = px.bar(top10, x='점수', y='드라마명', orientation='h', text='점수', title="Top 10 평점 작품")
    st.plotly_chart(fig2, use_container_width=True)

    # 3) 연도별 주요 플랫폼 작품 수 (로그 스케일)
    st.subheader("연도별 주요 플랫폼 작품 수")
    ct = (
        pd.DataFrame({
            '방영년도': raw_df['방영년도'],
            '플랫폼': raw_df['플랫폼'].apply(clean_cell_colab)
        })
        .explode('플랫폼')
        .groupby(['방영년도', '플랫폼']).size().reset_index(name='count')
    )
    ct['플랫폼_up'] = ct['플랫폼'].str.upper()
    focus = ['KBS','MBC','TVN','NETFLIX','SBS']
    fig3 = px.line(
        ct[ct['플랫폼_up'].isin(focus)],
        x='방영년도', y='count', color='플랫폼',
        log_y=True, title="연도별 주요 플랫폼 작품 수 (로그 스케일)"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # 4) 멀티장르 vs 단일장르 (배우 단위 박스플롯)
    st.subheader("멀티장르 vs 단일장르 평균 평점 (배우 단위 박스플롯)")
    ag = (
        pd.DataFrame({'배우명': raw_df['배우명'], '장르': raw_df['장르'].apply(clean_cell_colab)})
        .explode('장르')
        .groupby('배우명')['장르'].nunique()
    )
    multi_set = set(ag[ag > 1].index)
    label_map = {name: ('멀티장르' if name in multi_set else '단일장르') for name in ag.index}
    actor_mean = (
        raw_df.groupby('배우명', as_index=False)['점수'].mean()
              .rename(columns={'점수': '배우평균점수'})
    )
    actor_mean['장르구분'] = actor_mean['배우명'].map(label_map)
    fig_box = px.box(
        actor_mean, x='장르구분', y='배우평균점수',
        title="멀티장르 vs 단일장르 배우 단위 평균 점수 분포"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # 5) 주연 배우 결혼 상태별 평균 점수 비교
    st.subheader("주연 배우 결혼 상태별 평균 점수 비교")
    main_roles = raw_df[raw_df['역할'] == '주연'].copy()
    main_roles['결혼상태'] = main_roles['결혼여부'].apply(lambda x: '미혼' if x == '미혼' else '미혼 외')
    avg_scores_by_marriage = main_roles.groupby('결혼상태')['점수'].mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(avg_scores_by_marriage.index, avg_scores_by_marriage.values, color=['mediumseagreen', 'gray'])
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title('주연 배우 결혼 상태별 평균 점수 비교', fontsize=14)
    ax.set_ylabel('평균 점수'); ax.set_xlabel('결혼 상태')
    ax.set_ylim(min(avg_scores_by_marriage.values) - 0.05, max(avg_scores_by_marriage.values) + 0.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)

    # 6) 장르별 작품 수 & 평균 점수 (막대+선)
    st.subheader("장르별 작품 수 및 평균 점수")
    df_exploded = raw_df.copy()
    df_exploded['장르'] = df_exploded['장르'].apply(clean_cell_colab)
    df_exploded = df_exploded.explode('장르').dropna(subset=['장르','점수'])
    genre_score = df_exploded.groupby('장르')['점수'].mean().round(3)
    genre_count = df_exploded['장르'].value_counts()
    genre_df = (pd.DataFrame({'평균 점수': genre_score, '작품 수': genre_count})
                .reset_index().rename(columns={'index': '장르'}))
    genre_df = genre_df.sort_values('작품 수', ascending=False).reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.bar(range(len(genre_df)), genre_df['작품 수'], color='lightgray')
    ax1.set_ylabel('작품 수', fontsize=12)
    ax1.set_xticks(range(len(genre_df)))
    ax1.set_xticklabels(genre_df['장르'], rotation=45, ha='right')
    for i, rect in enumerate(bars):
        h = rect.get_height()
        ax1.text(i, h + max(2, h*0.01), f'{int(h)}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#444')

    ax2 = ax1.twinx()
    ax2.plot(range(len(genre_df)), genre_df['평균 점수'], marker='o', linewidth=2, color='tab:blue')
    ax2.set_ylabel('평균 점수', fontsize=12, color='tab:blue')
    ax2.tick_params(axis='y', colors='tab:blue')
    ax2.set_ylim(genre_df['평균 점수'].min() - 0.1, genre_df['평균 점수'].max() + 0.1)
    for i, v in enumerate(genre_df['평균 점수']):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='tab:blue')
    plt.title('장르별 작품 수 및 평균 점수', fontsize=14)
    ax1.set_xlabel('장르', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    # 7) 방영 요일별 작품 수 & 평균 점수
    st.subheader("방영 요일별 작품 수 및 평균 점수 (월→일)")
    dfe = raw_df.copy()
    dfe['방영요일'] = dfe['방영요일'].apply(clean_cell_colab)
    dfe = dfe.explode('방영요일').dropna(subset=['방영요일','점수']).copy()
    dfe['방영요일'] = dfe['방영요일'].astype(str).str.strip().str.lower()

    ordered_days_en = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    day_label_ko = {'monday':'월','tuesday':'화','wednesday':'수','thursday':'목','friday':'금','saturday':'토','sunday':'일'}

    mean_score_by_day = dfe.groupby('방영요일')['점수'].mean().reindex(ordered_days_en)
    count_by_day = dfe['방영요일'].value_counts().reindex(ordered_days_en).fillna(0).astype(int)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(ordered_days_en, count_by_day.values, alpha=0.3, color='tab:gray')
    ax1.set_ylabel('작품 수', color='tab:gray', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:gray')
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{int(h)}', ha='center', va='bottom', fontsize=9, color='black')

    ax2 = ax1.twinx()
    ax2.plot(ordered_days_en, mean_score_by_day.values, marker='o', color='tab:blue')
    ax2.set_ylabel('평균 점수', color='tab:blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    if mean_score_by_day.notna().any():
        ax2.set_ylim(mean_score_by_day.min() - 0.05, mean_score_by_day.max() + 0.05)
    for x, y in zip(ordered_days_en, mean_score_by_day.values):
        if pd.notna(y):
            ax2.text(x, y + 0.005, f'{y:.3f}', color='tab:blue', fontsize=9, ha='center')

    ax1.set_xticks(ordered_days_en)
    ax1.set_xticklabels([day_label_ko[d] for d in ordered_days_en])
    plt.title('방영 요일별 작품 수 및 평균 점수 (월요일 → 일요일 순)', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

# --- 4.4 워드클라우드 ---
from wordcloud import WordCloud
with tabs[3]:
    st.header("워드클라우드")
    font_path = st.session_state.get("kfont_path")
    if genre_list:
        wc = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(' '.join(genre_list))
        fig,ax=plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)
    if broadcaster_list:
        wc = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(' '.join(broadcaster_list))
        fig,ax=plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)
    if week_list:
        wc = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(' '.join(week_list))
        fig,ax=plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)

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

# --- 4.7 머신러닝 모델링 ---
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

with tabs[6]:
    st.header("머신러닝 모델링")
    if feature_cols:
        X_all = build_X_from_selected(df_mlb, feature_cols)

        # split을 세션에 고정 (재실행에도 유지)
        key = ("split", tuple(X_all.columns), float(test_size))
        if st.session_state.get("split_key") != key:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=test_size, random_state=SEED, shuffle=True
            )
            st.session_state["split"] = (X_train, X_test, y_train, y_test)
            st.session_state["split_key"] = key
        else:
            X_train, X_test, y_train, y_test = st.session_state["split"]

        model = RandomForestRegressor(random_state=SEED) if model_type=="Random Forest" else LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.metric("R²", f"{r2_score(y_test,y_pred):.3f}")
        st.metric("MSE", f"{mean_squared_error(y_test,y_pred):.3f}")
    else:
        st.warning("사이드바에서 특성을 선택하세요.")

# --- 4.8 GridSearch 튜닝 ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

with tabs[7]:
    st.header("GridSearchCV 튜닝")
    if "split" not in st.session_state:
        st.info("먼저 '머신러닝 모델링' 탭에서 학습/스플릿을 생성하세요.")
    else:
        X_train, X_test, y_train, y_test = st.session_state["split"]
        scoring = st.selectbox("스코어링", ["neg_mean_squared_error", "r2"], index=0)
        cv = st.number_input("CV 폴드 수", min_value=3, max_value=10, value=5, step=1)

        model_zoo = {
            "KNN": KNeighborsRegressor(),
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet(max_iter=10000),
            "SGDRegressor": SGDRegressor(max_iter=10000),
            "SVR": SVR(),
            "Decision Tree": DecisionTreeRegressor(random_state=SEED),
            "Random Forest": RandomForestRegressor(random_state=SEED),
        }

        def make_pipeline(name):
            if name in ["Decision Tree", "Random Forest"]:
                return Pipeline([("model", model_zoo[name])])
            else:
                return Pipeline([
                    ("poly", PolynomialFeatures(include_bias=False)),
                    ("scaler", StandardScaler()),
                    ("model", model_zoo[name]),
                ])

        param_grids = {
            "KNN": {"poly__degree": [1, 2, 3], "model__n_neighbors": [3,4,5,6,7,8,9,10]},
            "Linear Regression": {"poly__degree": [1, 2, 3]},
            "Ridge": {"poly__degree": [1, 2, 3], "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            "Lasso": {"poly__degree": [1, 2, 3], "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            "ElasticNet": {"poly__degree": [1, 2, 3], "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000], "model__l1_ratio": [0.1, 0.5, 0.9]},
            "SGDRegressor": {"poly__degree": [1, 2, 3], "model__learning_rate": ["constant", "invscaling", "adaptive"]},
            "SVR": {"poly__degree": [1, 2, 3], "model__kernel": ["poly", "rbf", "sigmoid"], "model__degree": [1, 2, 3]},
            "Decision Tree": {"model__max_depth": [10, 15, 20, 25, 30], "model__min_samples_split": [5, 6, 7, 8, 9, 10], "model__min_samples_leaf": [2, 3, 4, 5], "model__max_leaf_nodes": [None, 10, 20, 30]},
            "Random Forest": {"model__n_estimators": [100, 200, 300], "model__min_samples_split": [5, 6, 7, 8, 9, 10], "model__max_depth": [5, 10, 15, 20, 25, 30]},
        }

        model_name = st.selectbox("튜닝할 모델 선택", list(model_zoo.keys()), index=0)
        if st.button("GridSearch 실행"):
            pipe = make_pipeline(model_name)
            grid = param_grids[model_name]
            gs = GridSearchCV(pipe, grid, scoring=scoring, cv=int(cv), n_jobs=-1, refit=True, return_train_score=True)
            with st.spinner("GridSearchCV 실행 중..."):
                gs.fit(X_train, y_train)

            st.subheader("베스트 결과")
            st.write("Best Params:", gs.best_params_)
            st.write("Best CV Score:", gs.best_score_)

            y_pred = gs.predict(X_test)
            st.write(f"Test MSE: {mean_squared_error(y_test,y_pred):.6f}")
            st.write(f"Test R2 : {r2_score(y_test,y_pred):.6f}")

            cvres = pd.DataFrame(gs.cv_results_)
            cols = ["rank_test_score","mean_test_score","std_test_score","mean_train_score","std_train_score","params"]
            st.dataframe(cvres[cols].sort_values("rank_test_score").reset_index(drop=True))

# --- 4.9 예측 실행 ---
with tabs[8]:
    st.header("평점 예측")
    st.subheader("1) 모델 설정")
    model_type2  = st.selectbox('모델 선택', ['Random Forest', 'Linear Regression'])
    test_size2   = st.slider('테스트셋 비율', 0.1, 0.5, 0.2, 0.05, key="ts2")
    feature_cols2 = st.multiselect('특성 선택',
        ['나이','방영년도','성별','장르','배우명','플랫폼','결혼여부'],
        default=['나이','방영년도','장르'],
        key="feat2"
    )

    st.markdown("---")
    st.subheader("2) 예측 입력")

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
        # 학습 데이터(Colab 인코딩과 동일)
        X_all = build_X_from_selected(df_mlb, feature_cols2)
        model = RandomForestRegressor(n_estimators=100, random_state=SEED) if model_type2=="Random Forest" else LinearRegression()
        model.fit(X_all, y_all)

        # 입력 1행 생성
        user_raw = pd.DataFrame([{
            '나이': input_age,
            '방영년도': input_year,
            '성별': input_gender,
            '장르': input_genre,     # 리스트 그대로
            '배우명': input_actor,
            '플랫폼': input_plat,    # 리스트
            '결혼여부': input_married
        }])

        # Colab과 동일 멀티라벨 transform
        user_mlb = colab_multilabel_transform(user_raw, cols=('장르','방영요일','플랫폼'))
        user_X = build_X_from_selected(user_mlb, feature_cols2)

        # 학습 X와 컬럼 정합
        for c in X_all.columns:
            if c not in user_X.columns:
                user_X[c] = 0
        user_X = user_X[X_all.columns]

        pred = model.predict(user_X)[0]
        st.success(f"💡 예상 평점: {pred:.2f}")
