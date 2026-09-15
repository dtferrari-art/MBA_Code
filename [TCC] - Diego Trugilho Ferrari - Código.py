# UNIVERSIDADE DE SÃO PAULO
# MBA DATA SCIENCE & ANALYTICS USP/ESALQ
# NOME: DIEGO TRUGILHO FERRARI
#
# Aplicação: previsão do consumo mensal de energia elétrica
# Versão: validação temporal 80% treino / 20% validação
#
# Objetivo desta versão
# ---------------------
# 1) Preservar os dois modelos finais definidos no trabalho:
#    - Regressão Linear Múltipla com transformação Box-Cox;
#    - Modelo Multinível HLM2 final log-linear.
# 2) Separar treinamento e validação no tempo, evitando que meses futuros
#    sejam usados para estimar o modelo que será avaliado nesses mesmos meses.
# 3) Estimar transformações e centralizações SOMENTE na base de treinamento.
# 4) Avaliar a capacidade de predição fora da amostra na escala original (kWh).
# 5) Comparar os dois modelos por métricas de erro calculadas na mesma escala
#    original de Consumo (RMSE, MAE e WMAPE). O R² do OLS com Box-Cox é
#    reportado somente como diagnóstico do ajuste na escala transformada. Ele
#    não é usado como métrica comum com o HLM2, pois um modelo misto exige uma
#    definição específica de R² (não adotada no roteiro multinível deste trabalho).
#
#!/usr/bin/env python
# coding: utf-8

# %% In[0.1]: Importação dos pacotes

import json
import warnings
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from patsy import dmatrices
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# %% In[0.2]: Configurações iniciais do projeto

BASE_DIR = Path.cwd()

# Para garantir reprodutibilidade, esta versão NÃO escolhe silenciosamente entre
# diferentes arquivos. O TCC deve ser executado sempre com a mesma base final.
ARQUIVO_BASE = BASE_DIR / "Dados_Consumo_Mensal(2).csv"
if not ARQUIVO_BASE.exists():
    alternativa = Path("/mnt/data/Dados_Consumo_Mensal(2).csv")
    if alternativa.exists():
        ARQUIVO_BASE = alternativa
    else:
        raise FileNotFoundError(
            "Base não localizada. Esta versão exige o arquivo "
            "'Dados_Consumo_Mensal(2).csv'."
        )

PERCENTUAL_TREINO = 0.80

PASTA_RESULTADOS = BASE_DIR / "resultados_previsao_80_20"
PASTA_GRAFICOS = PASTA_RESULTADOS / "graficos"
PASTA_TABELAS = PASTA_RESULTADOS / "tabelas"
PASTA_MODELOS = PASTA_RESULTADOS / "modelos"

for pasta in [PASTA_RESULTADOS, PASTA_GRAFICOS, PASTA_TABELAS, PASTA_MODELOS]:
    pasta.mkdir(parents=True, exist_ok=True)


# Identificação do arquivo efetivamente utilizado e hash para auditoria.
def sha256_arquivo(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()

HASH_BASE = sha256_arquivo(ARQUIVO_BASE)
print(f"\nBase utilizada: {ARQUIVO_BASE.resolve()}")
print(f"SHA-256 da base: {HASH_BASE}")


# %% In[1]: Carregamento e validação da base

# A base final está em CSV separado por ponto e vírgula e com decimal vírgula.
df = pd.read_csv(ARQUIVO_BASE, sep=";", decimal=",", encoding="utf-8")
df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

colunas_obrigatorias = [
    "Regiao", "Sistema", "Classe", "TipoConsumidor",
    "Consumo", "Consumidores", "TempMedia"
]
for c in colunas_obrigatorias:
    if c not in df.columns:
        raise ValueError(f"Coluna obrigatória ausente: {c}")

if "DataBase" in df.columns:
    df["DataRef"] = pd.to_datetime(df["DataBase"], dayfirst=True, errors="coerce")
elif "DataExcel" in df.columns:
    df["DataRef"] = pd.to_datetime(df["DataExcel"], dayfirst=True, errors="coerce")
else:
    raise ValueError("A base precisa conter DataBase ou DataExcel.")

for c in ["Consumo", "Consumidores", "TempMedia"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

n_original = len(df)

print("\nDimensão da base original:")
print(df.shape)
print("\nPeríodo original:")
print(df["DataRef"].min(), "até", df["DataRef"].max())

# Exporta metadados antes das exclusões metodológicas.
metadados = pd.DataFrame({
    "variavel": df.columns,
    "tipo": [str(df[c].dtype) for c in df.columns],
    "nulos": [int(df[c].isna().sum()) for c in df.columns],
    "categorias_unicas": [int(df[c].nunique(dropna=False)) for c in df.columns],
})
metadados.to_csv(PASTA_TABELAS / "01_metadados_variaveis.csv", index=False, sep=";", decimal=",")


# %% In[2]: Preparação mínima e criação de variáveis

# As transformações ln() e Box-Cox exigem valores estritamente positivos.
# Também são removidas datas inválidas ou valores ausentes nas variáveis usadas.
mascara_modelagem = (
    df["DataRef"].notna()
    & df["Consumo"].notna()
    & df["Consumidores"].notna()
    & df["TempMedia"].notna()
    & (df["Consumo"] > 0)
    & (df["Consumidores"] > 0)
)

df_model = df.loc[mascara_modelagem].copy()
df_model = df_model.sort_values("DataRef").reset_index(drop=True)
n_modelagem = len(df_model)

print("\nRegistros removidos por incompatibilidade com a modelagem:")
print(n_original - n_modelagem)

# Variáveis temporais. Tendencia é determinada apenas pela data e não utiliza Y.
data_inicial = df_model["DataRef"].min()
df_model["Ano"] = df_model["DataRef"].dt.year
df_model["Mes"] = df_model["DataRef"].dt.month
# Índice mensal: jan/2004 = 1, fev/2004 = 2, ...
df_model["Tendencia"] = (
    (df_model["DataRef"].dt.year - data_inicial.year) * 12
    + (df_model["DataRef"].dt.month - data_inicial.month)
    + 1
).astype(float)

# Transformações usadas nas especificações finais.
df_model["ln_Consumo"] = np.log(df_model["Consumo"])
df_model["ln_Consumidores"] = np.log(df_model["Consumidores"])

# Grupo contextual do HLM2: Sistema + Classe + TipoConsumidor.
df_model["Grupo_HLM2"] = (
    df_model["Sistema"].astype(str) + " | "
    + df_model["Classe"].astype(str) + " | "
    + df_model["TipoConsumidor"].astype(str)
)

# Variáveis categóricas estruturais.
for c in ["Regiao", "Sistema", "Classe", "TipoConsumidor"]:
    df_model[c] = df_model[c].astype("category")
df_model["Mes"] = pd.Categorical(df_model["Mes"], categories=list(range(1, 13)))

# Salva base histórica tratada, antes do split.
df_model.to_csv(PASTA_TABELAS / "02_base_historica_tratada.csv", index=False, sep=";", decimal=",")


# %% In[3]: Análise exploratória descritiva

# A exploração abaixo é descritiva da base histórica completa. Nenhuma estatística
# calculada nesta seção é usada para estimar parâmetros dos modelos de validação.

estatisticas = df_model[[
    "Consumo", "Consumidores", "TempMedia", "ln_Consumo", "ln_Consumidores"
]].describe().T
estatisticas.to_csv(PASTA_TABELAS / "03_estatisticas_univariadas.csv", sep=";", decimal=",")

# Frequências das variáveis qualitativas.
for col in ["Sistema", "Classe", "TipoConsumidor", "Regiao", "Grupo_HLM2"]:
    freq = pd.DataFrame({
        "contagem": df_model[col].value_counts(dropna=False),
        "percentual": df_model[col].value_counts(dropna=False, normalize=True) * 100,
    })
    freq.to_csv(PASTA_TABELAS / f"04_frequencia_{col}.csv", sep=";", decimal=",")

# Evolução mensal do consumo total.
evolucao = df_model.groupby("DataRef", as_index=False, observed=True).agg(
    Consumo_Total=("Consumo", "sum"),
    Consumidores_Total=("Consumidores", "sum"),
    TempMedia_Media=("TempMedia", "mean"),
)
evolucao.to_csv(PASTA_TABELAS / "05_evolucao_consumo_total.csv", index=False, sep=";", decimal=",")

plt.figure(figsize=(15, 8))
sns.lineplot(data=evolucao, x="DataRef", y="Consumo_Total", linewidth=2)
plt.title("Evolução mensal do consumo total (kWh)", fontsize=18)
plt.xlabel("Data", fontsize=14)
plt.ylabel("Consumo total (kWh)", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "01_evolucao_consumo_total.png", dpi=300)
plt.close()

# Distribuições.
plt.figure(figsize=(15, 8))
sns.histplot(df_model["Consumo"], bins=40, color="steelblue")
plt.title("Distribuição do consumo mensal (kWh)", fontsize=18)
plt.xlabel("Consumo (kWh)", fontsize=14)
plt.ylabel("Frequência", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "02_distribuicao_consumo.png", dpi=300)
plt.close()

plt.figure(figsize=(15, 8))
sns.histplot(df_model["ln_Consumo"], bins=40, color="steelblue")
plt.title("Distribuição de ln(Consumo em kWh)", fontsize=18)
plt.xlabel("ln(Consumo em kWh)", fontsize=14)
plt.ylabel("Frequência", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "03_distribuicao_ln_consumo.png", dpi=300)
plt.close()

# Relações quantitativas com amostra gráfica para desempenho.
df_plot = df_model.sample(n=min(5000, len(df_model)), random_state=42).copy()

plt.figure(figsize=(15, 8))
sns.scatterplot(data=df_plot, x="ln_Consumidores", y="ln_Consumo", alpha=0.35, s=20, color="steelblue")
plt.title("Relação entre ln(número de consumidores) e ln(Consumo em kWh)", fontsize=18)
plt.xlabel("ln(número de consumidores)", fontsize=14)
plt.ylabel("ln(Consumo em kWh)", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "04_ln_consumidores_vs_ln_consumo.png", dpi=300)
plt.close()

plt.figure(figsize=(15, 8))
sns.scatterplot(data=df_plot, x="TempMedia", y="ln_Consumo", alpha=0.35, s=20, color="steelblue")
plt.title("Relação entre temperatura média (°C) e ln(Consumo em kWh)", fontsize=18)
plt.xlabel("Temperatura média (°C)", fontsize=14)
plt.ylabel("ln(Consumo em kWh)", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "05_tempmedia_vs_ln_consumo.png", dpi=300)
plt.close()

# Matriz de correlação.
correlacoes = df_model[[
    "Consumo", "Consumidores", "TempMedia", "ln_Consumo", "ln_Consumidores", "Tendencia"
]].corr()
correlacoes.to_csv(PASTA_TABELAS / "06_matriz_correlacoes.csv", sep=";", decimal=",")

plt.figure(figsize=(12, 8))
sns.heatmap(correlacoes, annot=True, fmt=".3f", vmin=-1, vmax=1)
plt.title("Matriz de correlação das variáveis quantitativas", fontsize=18)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "06_matriz_correlacoes.png", dpi=300)
plt.close()

# Consumo por Sistema.
consumo_sistema = df_model.groupby("Sistema", as_index=False, observed=True).agg(
    consumo_total=("Consumo", "sum"),
    consumo_medio=("Consumo", "mean"),
    consumidores_total=("Consumidores", "sum"),
    qtd_registros=("Consumo", "count"),
)
consumo_sistema["participacao_consumo_pct"] = (
    consumo_sistema["consumo_total"] / consumo_sistema["consumo_total"].sum() * 100
)
consumo_sistema.to_csv(PASTA_TABELAS / "07_consumo_por_sistema.csv", index=False, sep=";", decimal=",")

plt.figure(figsize=(15, 8))
sns.barplot(
    data=consumo_sistema.sort_values("consumo_total", ascending=False),
    x="consumo_total", y="Sistema", color="steelblue"
)
plt.title("Consumo total por Sistema (kWh)", fontsize=18)
plt.xlabel("Consumo total (kWh)", fontsize=14)
plt.ylabel("Sistema", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "07_consumo_total_por_sistema.png", dpi=300)
plt.close()

# Relação Região x Sistema, mantida como diagnóstico.
crosstab_regiao_sistema = pd.crosstab(df_model["Regiao"], df_model["Sistema"])
crosstab_regiao_sistema.to_csv(PASTA_TABELAS / "08_crosstab_regiao_sistema.csv", sep=";", decimal=",")
chi2, p_chi2, _, _ = stats.chi2_contingency(crosstab_regiao_sistema)
n_cramer = crosstab_regiao_sistema.to_numpy().sum()
min_dim = min(crosstab_regiao_sistema.shape) - 1
cramers_v = np.sqrt((chi2 / n_cramer) / min_dim)

pd.DataFrame({
    "teste": ["Qui-quadrado Regiao x Sistema", "Cramer_V Regiao x Sistema"],
    "estatistica": [chi2, cramers_v],
    "p_valor": [p_chi2, np.nan],
}).to_csv(PASTA_TABELAS / "09_teste_associacao_regiao_sistema.csv", index=False, sep=";", decimal=",")


# %% In[4]: Split temporal 80% treino / 20% validação

contagem_mensal = df_model.groupby("DataRef", observed=True).size().sort_index()
acumulado = contagem_mensal.cumsum()
alvo_treino = PERCENTUAL_TREINO * len(df_model)
data_corte = (acumulado - alvo_treino).abs().idxmin()

base_treino = df_model[df_model["DataRef"] <= data_corte].copy()
base_validacao = df_model[df_model["DataRef"] > data_corte].copy()

pct_linhas_treino = len(base_treino) / len(df_model) * 100
pct_linhas_validacao = len(base_validacao) / len(df_model) * 100

split_info = pd.DataFrame({
    "base": ["Treinamento", "Validação temporal"],
    "data_min": [base_treino["DataRef"].min(), base_validacao["DataRef"].min()],
    "data_max": [base_treino["DataRef"].max(), base_validacao["DataRef"].max()],
    "qtd_linhas": [len(base_treino), len(base_validacao)],
    "percentual_linhas": [pct_linhas_treino, pct_linhas_validacao],
    "qtd_meses": [base_treino["DataRef"].nunique(), base_validacao["DataRef"].nunique()],
})
split_info.to_csv(PASTA_TABELAS / "10_split_temporal_80_20.csv", index=False, sep=";", decimal=",")
print("\nSplit temporal utilizado:")
print(split_info)

# Verifica se as categorias fixas da validação estão presentes no treino.
checagem_categorias = []
for col in ["Sistema", "Classe", "TipoConsumidor", "Mes"]:
    treino_vals = set(base_treino[col].astype(str).unique())
    valid_vals = set(base_validacao[col].astype(str).unique())
    novos = sorted(valid_vals - treino_vals)
    checagem_categorias.append({
        "variavel": col,
        "categorias_novas_validacao": " | ".join(novos) if novos else "Nenhuma",
        "qtd_novas": len(novos),
    })
pd.DataFrame(checagem_categorias).to_csv(
    PASTA_TABELAS / "11_checagem_categorias_validacao.csv", index=False, sep=";", decimal="," 
)

# CENTRALIZAÇÃO SEM VAZAMENTO:
# As médias são calculadas exclusivamente no treinamento e aplicadas às duas bases.
medias_treino = {
    "ln_Consumidores": float(base_treino["ln_Consumidores"].mean()),
    "TempMedia": float(base_treino["TempMedia"].mean()),
    "Tendencia": float(base_treino["Tendencia"].mean()),
}

for base in [base_treino, base_validacao]:
    base["ln_Consumidores_c"] = base["ln_Consumidores"] - medias_treino["ln_Consumidores"]
    base["TempMedia_c"] = base["TempMedia"] - medias_treino["TempMedia"]
    base["Tendencia_c"] = base["Tendencia"] - medias_treino["Tendencia"]

# BOX-COX SEM VAZAMENTO:
# Lambda é estimado apenas no treinamento e depois reutilizado na validação.
base_treino["bc_Consumo"], lambda_boxcox_treino = boxcox(base_treino["Consumo"])
base_validacao["bc_Consumo"] = boxcox(base_validacao["Consumo"], lmbda=lambda_boxcox_treino)

print(f"\nLambda Box-Cox estimado somente no treinamento: {lambda_boxcox_treino:.6f}")

# Salva as duas bases usadas efetivamente nos modelos.
base_treino.to_csv(PASTA_TABELAS / "12_base_treinamento.csv", index=False, sep=";", decimal=",")
base_validacao.to_csv(PASTA_TABELAS / "13_base_validacao_temporal.csv", index=False, sep=";", decimal=",")


# %% In[5]: Funções auxiliares de modelagem e avaliação

def salvar_summary(modelo, caminho):
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(str(modelo.summary()))


def calcular_vif(formula, data):
    _, X = dmatrices(formula, data=data, return_type="dataframe")
    X = X.drop(columns=["Intercept"], errors="ignore")
    resultado = pd.DataFrame({
        "variavel": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    resultado["Tolerancia"] = 1 / resultado["VIF"]
    return resultado.sort_values("VIF", ascending=False)


def calcular_metricas(y_real, y_pred, nome_modelo, conjunto):
    """
    Calcula métricas de erro na escala original de Consumo (kWh).

    RMSE, MAE e WMAPE são comparáveis entre os dois modelos porque utilizam
    as mesmas observações e a mesma unidade da variável resposta após a
    retransfomação das predições.

    O R² não é calculado aqui. Para a regressão OLS com Box-Cox, o R² e o
    R² ajustado pertencem ao modelo estimado na escala Box-Cox e são obtidos
    diretamente de modelo_boxcox.rsquared e modelo_boxcox.rsquared_adj.
    Para o MixedLM/HLM2, a implementação e o roteiro multinível usados neste
    trabalho não fornecem um R² convencional diretamente equivalente ao R² do
    OLS. Existem definições específicas de R² para modelos mistos, mas nenhuma
    delas foi adotada nesta análise. Por isso, R² não é usado como métrica comum
    de comparação entre os dois modelos.
    """
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
    mae = float(mean_absolute_error(y_real, y_pred))
    wmape = float(np.sum(np.abs(y_real - y_pred)) / np.sum(np.abs(y_real)) * 100)

    return {
        "modelo": nome_modelo,
        "conjunto": conjunto,
        "N": len(y_real),
        "RMSE_kWh": rmse,
        "MAE_kWh": mae,
        "WMAPE_percentual": wmape,
    }


def lrtest(modelo_restrito, modelo_completo):
    """Teste LR para modelos HLM2 aninhados ajustados por ML (reml=False)."""
    ll0 = float(modelo_restrito.llf)
    ll1 = float(modelo_completo.llf)
    gl = max(int(abs(len(modelo_completo.params) - len(modelo_restrito.params))), 1)
    lr = -2 * (ll0 - ll1)
    p = float(stats.chi2.sf(lr, gl))
    return pd.DataFrame({
        "LL_modelo_restrito": [ll0],
        "LL_modelo_completo": [ll1],
        "LR_statistic": [lr],
        "graus_liberdade": [gl],
        "p_valor": [p],
    })


def extrair_intercepto_aleatorio(valor):
    try:
        return float(valor.iloc[0])
    except Exception:
        return float(np.asarray(valor).reshape(-1)[0])


def prever_hlm2_futuro(resultado, dados):
    """Predição HLM2 fora da amostra.

    MixedLM.predict() retorna a componente de efeitos fixos. Para grupos já
    observados no treinamento, adiciona-se o intercepto aleatório estimado com
    os dados históricos do grupo. Para um grupo totalmente novo, o efeito
    aleatório não é conhecido e é definido como zero (média populacional).
    """
    pred_fixo = np.asarray(resultado.predict(dados), dtype=float)
    mapa_re = {
        str(grupo): extrair_intercepto_aleatorio(valor)
        for grupo, valor in resultado.random_effects.items()
    }
    re = np.array([
        mapa_re.get(str(grupo), 0.0)
        for grupo in dados["Grupo_HLM2"].astype(str)
    ])
    return pred_fixo + re, mapa_re


# %% In[6]: Modelo 1 - Regressão Linear Múltipla com Box-Cox

formula_boxcox = (
    "bc_Consumo ~ ln_Consumidores_c + TempMedia_c + Tendencia_c + "
    "C(Sistema) + C(Classe) + C(TipoConsumidor) + C(Mes)"
)

modelo_boxcox = sm.OLS.from_formula(formula_boxcox, data=base_treino).fit()
salvar_summary(modelo_boxcox, PASTA_MODELOS / "01_summary_boxcox_treino.txt")

# Diagnósticos de ajuste específicos do OLS na escala Box-Cox.
# Estes valores NÃO são usados para comparar o Box-Cox com o HLM2.
r2_boxcox_ols = float(modelo_boxcox.rsquared)
r2aj_boxcox_ols = float(modelo_boxcox.rsquared_adj)
pd.DataFrame({
    "indicador": ["R2_OLS_escala_BoxCox", "R2_ajustado_OLS_escala_BoxCox"],
    "valor": [r2_boxcox_ols, r2aj_boxcox_ols],
}).to_csv(PASTA_TABELAS / "14a_r2_ols_boxcox_treino.csv", index=False, sep=";", decimal=",")

# Predições na escala Box-Cox.
pred_boxcox_treino_bc = np.asarray(modelo_boxcox.predict(base_treino), dtype=float)
pred_boxcox_valid_bc = np.asarray(modelo_boxcox.predict(base_validacao), dtype=float)

# Inversa de Box-Cox. Quando lambda > 0, exige lambda*y + 1 > 0.
# Caso alguma predição extrapole ligeiramente o domínio, ela é limitada ao
# menor valor matematicamente admissível. A quantidade dessas ocorrências é
# registrada explicitamente para não ocultar a limitação.
if lambda_boxcox_treino > 0:
    limite_inferior_bc = (-1 / lambda_boxcox_treino) + 1e-8
    pred_boxcox_treino_bc_inv = np.maximum(pred_boxcox_treino_bc, limite_inferior_bc)
    pred_boxcox_valid_bc_inv = np.maximum(pred_boxcox_valid_bc, limite_inferior_bc)
else:
    limite_inferior_bc = np.nan
    pred_boxcox_treino_bc_inv = pred_boxcox_treino_bc
    pred_boxcox_valid_bc_inv = pred_boxcox_valid_bc

flag_clip_boxcox_treino = pred_boxcox_treino_bc_inv != pred_boxcox_treino_bc
flag_clip_boxcox_valid = pred_boxcox_valid_bc_inv != pred_boxcox_valid_bc
n_clip_treino = int(np.sum(flag_clip_boxcox_treino))
n_clip_valid = int(np.sum(flag_clip_boxcox_valid))

pred_boxcox_treino = inv_boxcox(pred_boxcox_treino_bc_inv, lambda_boxcox_treino)
pred_boxcox_valid = inv_boxcox(pred_boxcox_valid_bc_inv, lambda_boxcox_treino)

# Diagnósticos na base de treinamento.
coef_boxcox = pd.DataFrame({
    "coeficiente": modelo_boxcox.params,
    "erro_padrao": modelo_boxcox.bse,
    "t": modelo_boxcox.tvalues,
    "p_valor": modelo_boxcox.pvalues,
})
coef_boxcox = coef_boxcox.join(
    modelo_boxcox.conf_int(alpha=0.05).rename(columns={0: "IC_2_5", 1: "IC_97_5"})
)
coef_boxcox.to_csv(PASTA_TABELAS / "14_coeficientes_boxcox_treino.csv", sep=";", decimal=",")

vif_boxcox = calcular_vif(formula_boxcox, base_treino)
vif_boxcox.to_csv(PASTA_TABELAS / "15_vif_boxcox_treino.csv", index=False, sep=";", decimal=",")

bp = het_breuschpagan(modelo_boxcox.resid, modelo_boxcox.model.exog)
jarque = stats.jarque_bera(modelo_boxcox.resid)
diagnosticos_boxcox = pd.DataFrame({
    "teste": ["Breusch-Pagan LM", "Breusch-Pagan LM p-valor", "Breusch-Pagan F", "Breusch-Pagan F p-valor", "Jarque-Bera", "Jarque-Bera p-valor"],
    "valor": [bp[0], bp[1], bp[2], bp[3], jarque.statistic, jarque.pvalue],
})
diagnosticos_boxcox.to_csv(PASTA_TABELAS / "16_diagnosticos_boxcox_treino.csv", index=False, sep=";", decimal=",")

pd.DataFrame({
    "indicador": ["lambda_boxcox_treino", "limite_inferior_escala_boxcox", "predicoes_clip_treino", "predicoes_clip_validacao"],
    "valor": [lambda_boxcox_treino, limite_inferior_bc, n_clip_treino, n_clip_valid],
}).to_csv(PASTA_TABELAS / "17_retransformacao_boxcox.csv", index=False, sep=";", decimal=",")


# %% In[7]: Modelo 2 - Modelo Multinível HLM2 final log-linear

formula_hlm2 = "ln_Consumo ~ ln_Consumidores_c + TempMedia_c + Tendencia_c + C(Mes)"

# Modelo nulo somente para LR/ICC na base de treinamento.
modelo_nulo_hlm2 = sm.MixedLM.from_formula(
    formula="ln_Consumo ~ 1",
    groups="Grupo_HLM2",
    re_formula="1",
    data=base_treino,
).fit(method="lbfgs", reml=False, maxiter=500, disp=False)

modelo_hlm2 = sm.MixedLM.from_formula(
    formula=formula_hlm2,
    groups="Grupo_HLM2",
    re_formula="1",
    data=base_treino,
).fit(method="lbfgs", reml=False, maxiter=500, disp=False)

salvar_summary(modelo_nulo_hlm2, PASTA_MODELOS / "02_summary_hlm2_nulo_treino.txt")
salvar_summary(modelo_hlm2, PASTA_MODELOS / "03_summary_hlm2_final_treino.txt")

# Treino: fittedvalues contém efeitos fixos + aleatórios para os grupos observados.
pred_hlm2_treino_log = np.asarray(modelo_hlm2.fittedvalues, dtype=float)
pred_hlm2_treino = np.exp(pred_hlm2_treino_log)

# Validação: efeitos fixos + intercepto aleatório histórico para grupos conhecidos.
pred_hlm2_valid_log, mapa_efeitos_aleatorios = prever_hlm2_futuro(modelo_hlm2, base_validacao)
pred_hlm2_valid = np.exp(pred_hlm2_valid_log)

# Grupos novos no período de validação.
base_validacao["grupo_visto_no_treino"] = base_validacao["Grupo_HLM2"].astype(str).isin(
    set(mapa_efeitos_aleatorios.keys())
)
grupos_novos = (
    base_validacao.loc[~base_validacao["grupo_visto_no_treino"], "Grupo_HLM2"]
    .astype(str)
    .value_counts()
    .rename_axis("Grupo_HLM2")
    .reset_index(name="qtd_linhas_validacao")
)
if len(grupos_novos) > 0:
    grupos_novos["percentual_validacao"] = grupos_novos["qtd_linhas_validacao"] / len(base_validacao) * 100
grupos_novos.to_csv(PASTA_TABELAS / "18_grupos_hlm2_novos_validacao.csv", index=False, sep=";", decimal=",")

# Coeficientes fixos e efeitos aleatórios estimados no treinamento.
coef_hlm2 = pd.DataFrame({
    "coeficiente": modelo_hlm2.fe_params,
    "erro_padrao": modelo_hlm2.bse_fe,
    "z": modelo_hlm2.fe_params / modelo_hlm2.bse_fe,
    "p_valor": modelo_hlm2.pvalues[modelo_hlm2.fe_params.index],
})
coef_hlm2.to_csv(PASTA_TABELAS / "19_coeficientes_hlm2_treino.csv", sep=";", decimal=",")

efeitos_aleatorios = pd.DataFrame([
    {"Grupo_HLM2": str(g), "intercepto_aleatorio": extrair_intercepto_aleatorio(v)}
    for g, v in modelo_hlm2.random_effects.items()
]).sort_values("intercepto_aleatorio", ascending=False)
efeitos_aleatorios.to_csv(PASTA_TABELAS / "20_efeitos_aleatorios_hlm2_treino.csv", index=False, sep=";", decimal=",")

# LR e ICC na base de treinamento.
teste_lr_hlm2 = lrtest(modelo_nulo_hlm2, modelo_hlm2)
teste_lr_hlm2.to_csv(PASTA_TABELAS / "21_lrtest_hlm2_treino.csv", index=False, sep=";", decimal=",")

var_grupo = float(modelo_hlm2.cov_re.iloc[0, 0])
var_resid = float(modelo_hlm2.scale)
icc_hlm2 = var_grupo / (var_grupo + var_resid)
pd.DataFrame({
    "variancia_grupo": [var_grupo],
    "variancia_residual": [var_resid],
    "ICC": [icc_hlm2],
}).to_csv(PASTA_TABELAS / "22_icc_hlm2_treino.csv", index=False, sep=";", decimal=",")


# %% In[8]: Métricas de ajuste (treino) e predição (validação)

metricas = pd.DataFrame([
    calcular_metricas(base_treino["Consumo"], pred_boxcox_treino, "Regressão Linear Múltipla com Box-Cox", "Treinamento - ajuste"),
    calcular_metricas(base_validacao["Consumo"], pred_boxcox_valid, "Regressão Linear Múltipla com Box-Cox", "Validação temporal - predição"),
    calcular_metricas(base_treino["Consumo"], pred_hlm2_treino, "Modelo Multinível HLM2 final log-linear", "Treinamento - ajuste"),
    calcular_metricas(base_validacao["Consumo"], pred_hlm2_valid, "Modelo Multinível HLM2 final log-linear", "Validação temporal - predição"),
])
metricas.to_csv(PASTA_TABELAS / "23_metricas_treino_validacao.csv", index=False, sep=";", decimal=",")

# Métricas estatísticas do espaço de estimação, apenas como diagnóstico do treino.
# ATENÇÃO: AIC/BIC/LogLik NÃO são usados para escolher entre Box-Cox e HLM2,
# pois as variáveis resposta estão em escalas transformadas diferentes.
ajuste_estatistico = pd.DataFrame({
    "modelo": ["Regressão Linear Múltipla com Box-Cox", "Modelo Multinível HLM2 final log-linear"],
    "escala_resposta": ["Box-Cox(Consumo)", "ln(Consumo)"],
    "AIC_treino": [modelo_boxcox.aic, modelo_hlm2.aic],
    "BIC_treino": [modelo_boxcox.bic, modelo_hlm2.bic],
    "LogLik_treino": [modelo_boxcox.llf, modelo_hlm2.llf],
})
ajuste_estatistico.to_csv(PASTA_TABELAS / "24_ajuste_estatistico_treino_nao_comparar_entre_escalas.csv", index=False, sep=";", decimal=",")

# Seleção principal baseada na validação temporal.
metricas_validacao = metricas[metricas["conjunto"] == "Validação temporal - predição"].copy()
melhor_rmse = metricas_validacao.sort_values("RMSE_kWh").iloc[0]["modelo"]
melhor_mae = metricas_validacao.sort_values("MAE_kWh").iloc[0]["modelo"]
melhor_wmape = metricas_validacao.sort_values("WMAPE_percentual").iloc[0]["modelo"]

# Critério principal: menor RMSE na validação. MAE e WMAPE verificam a
# consistência da conclusão. R² não participa da comparação entre os modelos.
melhor_modelo = melhor_rmse

selecao = pd.DataFrame({
    "criterio": ["Menor RMSE validação", "Menor MAE validação", "Menor WMAPE validação", "Modelo final - critério principal RMSE"],
    "modelo": [melhor_rmse, melhor_mae, melhor_wmape, melhor_modelo],
})
selecao.to_csv(PASTA_TABELAS / "25_selecao_modelo_validacao.csv", index=False, sep=";", decimal=",")

print("\nMétricas de treinamento e validação:")
print(metricas.to_string(index=False))
print(f"\nModelo selecionado pela validação temporal (menor RMSE): {melhor_modelo}")


# %% In[9]: Exportação das predições de validação e gráficos preditivos

pred_validacao = base_validacao[[
    "DataRef", "Regiao", "Sistema", "Classe", "TipoConsumidor", "Grupo_HLM2",
    "Consumo", "Consumidores", "TempMedia", "grupo_visto_no_treino"
]].copy()
pred_validacao["pred_boxcox_kWh"] = pred_boxcox_valid
pred_validacao["boxcox_pred_clipped"] = flag_clip_boxcox_valid
pred_validacao["erro_boxcox_kWh"] = pred_validacao["Consumo"] - pred_validacao["pred_boxcox_kWh"]
pred_validacao["pred_hlm2_kWh"] = pred_hlm2_valid
pred_validacao["erro_hlm2_kWh"] = pred_validacao["Consumo"] - pred_validacao["pred_hlm2_kWh"]
pred_validacao.to_csv(PASTA_TABELAS / "26_predicoes_validacao_detalhadas.csv", index=False, sep=";", decimal=",")

# Agregação mensal para visualizar a trajetória do período fora da amostra.
validacao_mensal = pred_validacao.groupby("DataRef", as_index=False).agg(
    Consumo_Real_kWh=("Consumo", "sum"),
    BoxCox_Previsto_kWh=("pred_boxcox_kWh", "sum"),
    HLM2_Previsto_kWh=("pred_hlm2_kWh", "sum"),
)
validacao_mensal.to_csv(PASTA_TABELAS / "27_validacao_mensal_real_previsto.csv", index=False, sep=";", decimal=",")

# Métricas suplementares no total mensal agregado. Não substituem as métricas
# principais por registro, mas ajudam a interpretar o desempenho no consumo
# agregado de cada mês do período de validação.
metricas_mensais = pd.DataFrame([
    calcular_metricas(validacao_mensal["Consumo_Real_kWh"], validacao_mensal["BoxCox_Previsto_kWh"],
                      "Regressão Linear Múltipla com Box-Cox", "Validação - total mensal agregado"),
    calcular_metricas(validacao_mensal["Consumo_Real_kWh"], validacao_mensal["HLM2_Previsto_kWh"],
                      "Modelo Multinível HLM2 final log-linear", "Validação - total mensal agregado"),
])
metricas_mensais.to_csv(PASTA_TABELAS / "28_metricas_validacao_total_mensal.csv", index=False, sep=";", decimal=",")

plt.figure(figsize=(15, 8))
plt.plot(validacao_mensal["DataRef"], validacao_mensal["Consumo_Real_kWh"], label="Real", linewidth=2.5)
plt.plot(validacao_mensal["DataRef"], validacao_mensal["BoxCox_Previsto_kWh"], label="Box-Cox", linewidth=2)
plt.plot(validacao_mensal["DataRef"], validacao_mensal["HLM2_Previsto_kWh"], label="HLM2", linewidth=2)
plt.title("Validação temporal: consumo mensal real x previsto", fontsize=18)
plt.xlabel("Data", fontsize=14)
plt.ylabel("Consumo agregado (kWh)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "08_validacao_temporal_real_previsto.png", dpi=300)
plt.close()

# Real x previsto - Box-Cox.
plt.figure(figsize=(9, 9))
plt.scatter(base_validacao["Consumo"], pred_boxcox_valid, alpha=0.35, s=18)
lim_max = max(float(base_validacao["Consumo"].max()), float(np.max(pred_boxcox_valid)))
plt.plot([0, lim_max], [0, lim_max], linestyle="--")
plt.title("Validação: consumo real x previsto - Box-Cox", fontsize=16)
plt.xlabel("Consumo real (kWh)")
plt.ylabel("Consumo previsto (kWh)")
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "09_validacao_real_vs_previsto_boxcox.png", dpi=300)
plt.close()

# Real x previsto - HLM2.
plt.figure(figsize=(9, 9))
plt.scatter(base_validacao["Consumo"], pred_hlm2_valid, alpha=0.35, s=18)
lim_max = max(float(base_validacao["Consumo"].max()), float(np.max(pred_hlm2_valid)))
plt.plot([0, lim_max], [0, lim_max], linestyle="--")
plt.title("Validação: consumo real x previsto - HLM2", fontsize=16)
plt.xlabel("Consumo real (kWh)")
plt.ylabel("Consumo previsto (kWh)")
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "10_validacao_real_vs_previsto_hlm2.png", dpi=300)
plt.close()

# Comparação das principais métricas de validação em painéis separados para
# evitar misturar escalas incompatíveis no mesmo eixo.
for metrica, rotulo, arquivo in [
    ("RMSE_kWh", "RMSE (kWh)", "11_validacao_rmse.png"),
    ("MAE_kWh", "MAE (kWh)", "12_validacao_mae.png"),
    ("WMAPE_percentual", "WMAPE (%)", "13_validacao_wmape.png"),
]:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metricas_validacao, x="modelo", y=metrica, color="steelblue")
    plt.title(f"Comparação preditiva - {rotulo}", fontsize=16)
    plt.xlabel("Modelo")
    plt.ylabel(rotulo)
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / arquivo, dpi=300)
    plt.close()


# %% In[10]: Reajuste do melhor modelo em 100% dos dados para uso futuro

# Após a comparação fora da amostra, é prática comum reestimar o modelo vencedor
# usando todo o histórico disponível antes de seu uso operacional. Esse reajuste
# NÃO altera nem substitui as métricas de validação reportadas acima.

metadados_modelo_final = {
    "modelo_selecionado_validacao": melhor_modelo,
    "criterio_principal": "menor RMSE na validação temporal",
    "data_inicial_historico": str(df_model["DataRef"].min().date()),
    "data_final_historico": str(df_model["DataRef"].max().date()),
    "observacao": (
        "O modelo final foi escolhido no holdout temporal e depois reajustado em 100% "
        "do histórico. Para prever períodos futuros, as variáveis explicativas futuras "
        "precisam ser conhecidas ou estimadas."
    ),
}

if melhor_modelo == "Regressão Linear Múltipla com Box-Cox":
    base_final = df_model.copy()
    medias_final = {
        "ln_Consumidores": float(base_final["ln_Consumidores"].mean()),
        "TempMedia": float(base_final["TempMedia"].mean()),
        "Tendencia": float(base_final["Tendencia"].mean()),
    }
    base_final["ln_Consumidores_c"] = base_final["ln_Consumidores"] - medias_final["ln_Consumidores"]
    base_final["TempMedia_c"] = base_final["TempMedia"] - medias_final["TempMedia"]
    base_final["Tendencia_c"] = base_final["Tendencia"] - medias_final["Tendencia"]
    base_final["bc_Consumo"], lambda_final = boxcox(base_final["Consumo"])
    modelo_final_operacional = sm.OLS.from_formula(formula_boxcox, data=base_final).fit()
    salvar_summary(modelo_final_operacional, PASTA_MODELOS / "04_modelo_final_reajustado_100pct_boxcox.txt")
    metadados_modelo_final["lambda_boxcox_reajuste_100pct"] = float(lambda_final)
    metadados_modelo_final["medias_centralizacao_reajuste_100pct"] = medias_final
else:
    base_final = df_model.copy()
    medias_final = {
        "ln_Consumidores": float(base_final["ln_Consumidores"].mean()),
        "TempMedia": float(base_final["TempMedia"].mean()),
        "Tendencia": float(base_final["Tendencia"].mean()),
    }
    base_final["ln_Consumidores_c"] = base_final["ln_Consumidores"] - medias_final["ln_Consumidores"]
    base_final["TempMedia_c"] = base_final["TempMedia"] - medias_final["TempMedia"]
    base_final["Tendencia_c"] = base_final["Tendencia"] - medias_final["Tendencia"]
    modelo_final_operacional = sm.MixedLM.from_formula(
        formula=formula_hlm2,
        groups="Grupo_HLM2",
        re_formula="1",
        data=base_final,
    ).fit(method="lbfgs", reml=False, maxiter=500, disp=False)
    salvar_summary(modelo_final_operacional, PASTA_MODELOS / "04_modelo_final_reajustado_100pct_hlm2.txt")
    metadados_modelo_final["medias_centralizacao_reajuste_100pct"] = medias_final

with open(PASTA_MODELOS / "05_metadados_modelo_final.json", "w", encoding="utf-8") as f:
    json.dump(metadados_modelo_final, f, ensure_ascii=False, indent=2)

# Metadados mínimos para reproduzir a execução e conferir se a mesma base foi usada.
try:
    import statsmodels
    import scipy
    import sklearn
    versoes = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "statsmodels": statsmodels.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
except Exception:
    versoes = {}

reprodutibilidade = {
    "arquivo_base": str(ARQUIVO_BASE.resolve()),
    "sha256_base": HASH_BASE,
    "percentual_treino_alvo": PERCENTUAL_TREINO,
    "data_corte_treino": str(data_corte.date()),
    "n_treino": int(len(base_treino)),
    "n_validacao": int(len(base_validacao)),
    "versoes_bibliotecas": versoes,
}
with open(PASTA_RESULTADOS / "00_reprodutibilidade_execucao.json", "w", encoding="utf-8") as f:
    json.dump(reprodutibilidade, f, ensure_ascii=False, indent=2)


# %% In[11]: Relatório 


def linha_metrica(modelo, conjunto):
    return metricas[(metricas["modelo"] == modelo) & (metricas["conjunto"] == conjunto)].iloc[0]

mb_tr = linha_metrica("Regressão Linear Múltipla com Box-Cox", "Treinamento - ajuste")
mb_va = linha_metrica("Regressão Linear Múltipla com Box-Cox", "Validação temporal - predição")
mh_tr = linha_metrica("Modelo Multinível HLM2 final log-linear", "Treinamento - ajuste")
mh_va = linha_metrica("Modelo Multinível HLM2 final log-linear", "Validação temporal - predição")

n_grupos_novos = int(grupos_novos["Grupo_HLM2"].nunique()) if not grupos_novos.empty else 0
n_linhas_grupos_novos = int(grupos_novos["qtd_linhas_validacao"].sum()) if not grupos_novos.empty else 0

relatorio = f"""
# Relatório sintético - validação temporal dos modelos de consumo de energia

## Base e split

- Arquivo: `{ARQUIVO_BASE.name}`
- SHA-256 da base: `{HASH_BASE}`
- Registros originais: {n_original:,}
- Registros usados após filtros metodológicos: {n_modelagem:,}
- Treinamento: {len(base_treino):,} registros ({pct_linhas_treino:.2f}%), de {base_treino['DataRef'].min().date()} a {base_treino['DataRef'].max().date()}
- Validação: {len(base_validacao):,} registros ({pct_linhas_validacao:.2f}%), de {base_validacao['DataRef'].min().date()} a {base_validacao['DataRef'].max().date()}

## Diagnósticos de ajuste específicos

### Regressão Linear Múltipla com Box-Cox

- R² OLS na escala Box-Cox: {r2_boxcox_ols:.4f}
- R² ajustado OLS na escala Box-Cox: {r2aj_boxcox_ols:.4f}
- Esses valores pertencem ao OLS estimado com `bc_Consumo` como resposta e não são comparados com um R² do HLM2.

### HLM2 log-linear

- Log-Likelihood do modelo final: {modelo_hlm2.llf:.4f}
- LR nulo vs. final: {float(teste_lr_hlm2['LR_statistic'].iloc[0]):.4f}
- Graus de liberdade do LR: {int(teste_lr_hlm2['graus_liberdade'].iloc[0])}
- ICC: {icc_hlm2:.4f}
- O script de aula de modelagem multinível não utiliza R² convencional para o HLM2; a avaliação interna usa verossimilhança, teste LR e componentes de variância.

## Métricas comuns na escala original

| Modelo | Conjunto | RMSE (kWh) | MAE (kWh) | WMAPE (%) |
|---|---|---:|---:|---:|
| Box-Cox | Treino | {mb_tr['RMSE_kWh']:.2f} | {mb_tr['MAE_kWh']:.2f} | {mb_tr['WMAPE_percentual']:.2f} |
| Box-Cox | Validação | {mb_va['RMSE_kWh']:.2f} | {mb_va['MAE_kWh']:.2f} | {mb_va['WMAPE_percentual']:.2f} |
| HLM2 | Treino | {mh_tr['RMSE_kWh']:.2f} | {mh_tr['MAE_kWh']:.2f} | {mh_tr['WMAPE_percentual']:.2f} |
| HLM2 | Validação | {mh_va['RMSE_kWh']:.2f} | {mh_va['MAE_kWh']:.2f} | {mh_va['WMAPE_percentual']:.2f} |

## Pontos técnicos

- As métricas comuns são calculadas após a retransfomação direta das predições para kWh.
- Predições Box-Cox que exigiram limitação ao domínio matemático na validação: {n_clip_valid} de {len(base_validacao)}.
- Grupos HLM2 presentes na validação e ausentes no treinamento: {n_grupos_novos}; linhas afetadas: {n_linhas_grupos_novos} de {len(base_validacao)}.
- AIC/BIC/LogLik não são usados para comparar diretamente Box-Cox e HLM2 porque as respostas de estimação estão em escalas diferentes.

## Seleção final

Modelo selecionado pela menor RMSE de validação: **{melhor_modelo}**.

Na validação, o Box-Cox apresentou RMSE = {mb_va['RMSE_kWh']:.2f} kWh, MAE = {mb_va['MAE_kWh']:.2f} kWh e WMAPE = {mb_va['WMAPE_percentual']:.2f}%; o HLM2 apresentou RMSE = {mh_va['RMSE_kWh']:.2f} kWh, MAE = {mh_va['MAE_kWh']:.2f} kWh e WMAPE = {mh_va['WMAPE_percentual']:.2f}%. As três métricas comuns favorecem o Box-Cox nesta divisão temporal.

## Limitação de uso como previsão futura

A validação utiliza valores observados das variáveis explicativas no período de validação. Portanto, mede predição condicional do consumo. Para meses ainda futuros, Consumidores e TempMedia precisam ser conhecidos, previstos ou definidos por cenários.
"""

with open(PASTA_RESULTADOS / "Relatorio_final.md", "w", encoding="utf-8") as f:
    f.write(relatorio)

print("\nArquivos gerados em:")
print(PASTA_RESULTADOS.resolve())
print("\nProcessamento concluído com sucesso.")
