# UNIVERSIDADE DE SÃO PAULO
# MBA DATA SCIENCE & ANALYTICS USP/ESALQ
# NOME: DIEGO TRUGILHO FERRARI
#
# Aplicação: previsão do consumo mensal de energia elétrica
#!/usr/bin/env python
# coding: utf-8


# %% In[0.1]: Importação dos pacotes

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from patsy import dmatrices
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# %% In[0.2]: Configurações iniciais do projeto

ARQUIVO_BASE = "Dados_Consumo_Mensal.csv"
PASTA_RESULTADOS = Path("resultados_final")
PASTA_GRAFICOS = PASTA_RESULTADOS / "graficos"
PASTA_TABELAS = PASTA_RESULTADOS / "tabelas"
PASTA_MODELOS = PASTA_RESULTADOS / "modelos"

for pasta in [PASTA_RESULTADOS, PASTA_GRAFICOS, PASTA_TABELAS, PASTA_MODELOS]:
    pasta.mkdir(parents=True, exist_ok=True)

# %% In[1]: Carregamento da base de dados

# A base final está em CSV separado por ponto e vírgula e com decimal vírgula
# para a variável TempMedia.
df = pd.read_csv(ARQUIVO_BASE, sep=";", decimal=",", encoding="utf-8")

# A base pode vir com o campo temporal chamado DataBase.
# Para manter o restante do código padronizado, cria-se DataRef.
if "DataBase" in df.columns:
    df["DataRef"] = pd.to_datetime(df["DataBase"], dayfirst=True)
elif "DataExcel" in df.columns:
    df["DataRef"] = pd.to_datetime(df["DataExcel"], dayfirst=True)
else:
    raise ValueError("A base precisa conter DataBase ou DataExcel.")

# Visualização inicial da base
print("\nDimensão da base original:")
print(df.shape)
print("\nPrimeiras linhas:")
print(df.head())
print("\nInformações da base:")
print(df.info())
print("\nEstatísticas descritivas iniciais:")
print(df.describe(include="all"))

# Exporta metadados simples
metadados = pd.DataFrame({
    "variavel": df.columns,
    "tipo": [str(df[c].dtype) for c in df.columns],
    "nulos": [df[c].isna().sum() for c in df.columns],
    "categorias_unicas": [df[c].nunique(dropna=False) for c in df.columns]
})
metadados.to_csv(PASTA_TABELAS / "01_metadados_variaveis.csv", index=False, sep=";", decimal=",")

# %% In[2]: Preparação mínima para modelagem

# A transformação logarítmica e Box-Cox exigem variáveis estritamente positivas.
# Portanto, a única tratativa aplicada é a exclusão dos registros incompatíveis
# com ln(Consumo), ln(Consumidores) e Box-Cox(Consumo).
n_original = len(df)
df_model = df.loc[(df["Consumo"] > 0) & (df["Consumidores"] > 0)].copy()
n_modelagem = len(df_model)

print("\nRegistros removidos por Consumo <= 0 ou Consumidores <= 0:")
print(n_original - n_modelagem)

# Ordenação temporal
df_model = df_model.sort_values("DataRef").reset_index(drop=True)

# Variáveis temporais
df_model["Ano"] = df_model["DataRef"].dt.year
df_model["Mes"] = df_model["DataRef"].dt.month.astype("category")
df_model["AnoMes"] = df_model["DataRef"].dt.to_period("M").astype(str)
df_model["Tendencia"] = pd.factorize(df_model["AnoMes"])[0] + 1

# Variáveis transformadas
df_model["ln_Consumo"] = np.log(df_model["Consumo"])
df_model["ln_Consumidores"] = np.log(df_model["Consumidores"])

# Centralização de variáveis quantitativas para facilitar a convergência e a
# interpretação, especialmente no modelo multinível.
df_model["ln_Consumidores_c"] = df_model["ln_Consumidores"] - df_model["ln_Consumidores"].mean()
df_model["TempMedia_c"] = df_model["TempMedia"] - df_model["TempMedia"].mean()
df_model["Tendencia_c"] = df_model["Tendencia"] - df_model["Tendencia"].mean()

# Grupo contextual do modelo HLM2: nível 2.
# Como o foco é setorial, usa-se Sistema em vez de Região.
df_model["Grupo_HLM2"] = (
    df_model["Sistema"].astype(str) + " | " +
    df_model["Classe"].astype(str) + " | " +
    df_model["TipoConsumidor"].astype(str)
)

# Conversão das variáveis qualitativas para category, em linha com a aula.
for col in ["Sistema", "Classe", "TipoConsumidor", "Regiao", "Grupo_HLM2"]:
    df_model[col] = df_model[col].astype("category")

# Transformação de Box-Cox da variável dependente Consumo.
df_model["bc_Consumo"], lambda_boxcox = boxcox(df_model["Consumo"])
print(f"\nLambda estimado para Box-Cox: {lambda_boxcox:.6f}")

# Salva a base de modelagem.
df_model.to_csv(PASTA_TABELAS / "02_base_modelagem.csv", index=False, sep=";", decimal=",")

# Amostra apenas para gráficos com muitos pontos. A modelagem usa todos os dados.
df_plot = df_model.sample(n=min(5000, len(df_model)), random_state=42).copy()

# %% In[3]: Exploração dos dados

#############################################################################
# O objetivo desta etapa é apresentar evidências para justificar os modelos:
# - Consumo é positivo e assimétrico, justificando Box-Cox/log.
# - Observações estão agrupadas por Sistema/Classe/TipoConsumidor, justificando HLM2.
# - Sistema é preservado como variável setorial principal.
#############################################################################

# In[3.1]: Estatísticas univariadas
estatisticas = df_model[["Consumo", "Consumidores", "TempMedia", "ln_Consumo", "ln_Consumidores"]].describe().T
estatisticas.to_csv(PASTA_TABELAS / "03_estatisticas_univariadas.csv", sep=";", decimal=",")
print("\nEstatísticas univariadas da base de modelagem:")
print(estatisticas)

# In[3.2]: Tabelas de frequência das variáveis qualitativas
for col in ["Sistema", "Classe", "TipoConsumidor", "Regiao", "Grupo_HLM2"]:
    freq = pd.concat([
        df_model[col].value_counts(dropna=False),
        df_model[col].value_counts(dropna=False, normalize=True)
    ], axis=1)
    freq.columns = ["contagem", "percentual"]
    freq.to_csv(PASTA_TABELAS / f"04_frequencia_{col}.csv", sep=";", decimal=",")

# In[3.3]: Evolução temporal do consumo total
evolucao = df_model.groupby("DataRef", as_index=False)["Consumo"].sum()
evolucao.to_csv(PASTA_TABELAS / "05_evolucao_consumo_total.csv", index=False, sep=";", decimal=",")

plt.figure(figsize=(15, 8))
sns.lineplot(data=evolucao, x="DataRef", y="Consumo", linewidth=2.5, color="steelblue")
plt.title("Evolução mensal do consumo total (kWh)", fontsize=18)
plt.xlabel("Data", fontsize=14)
plt.ylabel("Consumo total (kWh)", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "01_evolucao_consumo_total.png", dpi=300)
plt.close()

# In[3.4]: Distribuição de Consumo e ln(Consumo)
plt.figure(figsize=(15, 8))
sns.histplot(df_model["Consumo"], kde=False, bins=40, color="steelblue")
plt.title("Distribuição do consumo mensal (kWh)", fontsize=18)
plt.xlabel("Consumo (kWh)", fontsize=14)
plt.ylabel("Frequência", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "02_distribuicao_consumo.png", dpi=300)
plt.close()

plt.figure(figsize=(15, 8))
sns.histplot(df_model["ln_Consumo"], kde=False, bins=40, color="steelblue")
plt.title("Distribuição de ln(Consumo em kWh)", fontsize=18)
plt.xlabel("ln(Consumo em kWh)", fontsize=14)
plt.ylabel("Frequência", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "03_distribuicao_ln_consumo.png", dpi=300)
plt.close()

# In[3.5]: Dispersão ln(Consumidores) x ln(Consumo)
plt.figure(figsize=(15, 8))
sns.scatterplot(data=df_plot, x="ln_Consumidores", y="ln_Consumo", alpha=0.35, s=20)
plt.title("Relação entre ln(número de consumidores) e ln(Consumo em kWh)", fontsize=18)
plt.xlabel("ln(número de consumidores)", fontsize=14)
plt.ylabel("ln(Consumo em kWh)", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "04_regplot_ln_consumidores_ln_consumo.png", dpi=300)
plt.close()

# In[3.6]: Consumo por Sistema
consumo_sistema = df_model.groupby("Sistema", as_index=False).agg(
    consumo_total=("Consumo", "sum"),
    consumo_medio=("Consumo", "mean"),
    consumidores_total=("Consumidores", "sum"),
    qtd_registros=("Consumo", "count")
)
consumo_sistema["participacao_consumo"] = consumo_sistema["consumo_total"] / consumo_sistema["consumo_total"].sum()
consumo_sistema.to_csv(PASTA_TABELAS / "06_consumo_por_sistema.csv", index=False, sep=";", decimal=",")

plt.figure(figsize=(15, 8))
sns.barplot(data=consumo_sistema.sort_values("consumo_total", ascending=False),
            x="consumo_total", y="Sistema", color="steelblue")
plt.title("Consumo total por Sistema (kWh)", fontsize=18)
plt.xlabel("Consumo total (kWh)", fontsize=14)
plt.ylabel("Sistema", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "05_consumo_total_por_sistema.png", dpi=300)
plt.close()

# In[3.7]: Boxplot de ln(Consumo) por Grupo_HLM2, usando os maiores grupos por consumo
principais_grupos = (
    df_model.groupby("Grupo_HLM2")["Consumo"].sum()
    .sort_values(ascending=False)
    .head(15)
    .index
)
plt.figure(figsize=(16, 9))
sns.boxplot(data=df_model[df_model["Grupo_HLM2"].isin(principais_grupos)],
            x="ln_Consumo", y="Grupo_HLM2", color="steelblue")
plt.title("Variação de ln(Consumo em kWh) nos principais grupos HLM2", fontsize=18)
plt.xlabel("ln(Consumo em kWh)", fontsize=14)
plt.ylabel("Grupo_HLM2", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "06_boxplot_ln_consumo_grupo_hlm2.png", dpi=300)
plt.close()

# In[3.8]: Temperatura média e consumo
plt.figure(figsize=(15, 8))
sns.scatterplot(data=df_plot, x="TempMedia", y="ln_Consumo", alpha=0.35, s=20)
plt.title("Relação entre temperatura média (°C) e ln(Consumo em kWh)", fontsize=18)
plt.xlabel("Temperatura média (°C)", fontsize=14)
plt.ylabel("ln(Consumo em kWh)", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "07_regplot_tempmedia_ln_consumo.png", dpi=300)
plt.close()

# In[3.9]: Matriz de correlação das variáveis quantitativas
correlacoes = df_model[["Consumo", "Consumidores", "TempMedia", "ln_Consumo", "ln_Consumidores", "Tendencia"]].corr()
correlacoes.to_csv(PASTA_TABELAS / "07_matriz_correlacoes.csv", sep=";", decimal=",")

plt.figure(figsize=(12, 8))
sns.heatmap(correlacoes, annot=True, fmt=".3f", vmin=-1, vmax=1)
plt.title("Matriz de correlação das variáveis quantitativas", fontsize=18)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "08_matriz_correlacoes.png", dpi=300)
plt.close()

# In[3.10]: Diagnóstico simples da relação Regiao x Sistema
# Região não entra no modelo final, mas é mantida para evidenciar que o uso
# simultâneo de Região e Sistema pode carregar informação redundante.
crosstab_regiao_sistema = pd.crosstab(df_model["Regiao"], df_model["Sistema"])
crosstab_regiao_sistema.to_csv(PASTA_TABELAS / "08_crosstab_regiao_sistema.csv", sep=";", decimal=",")

chi2, p_chi2, _, _ = stats.chi2_contingency(crosstab_regiao_sistema)
n = crosstab_regiao_sistema.to_numpy().sum()
min_dim = min(crosstab_regiao_sistema.shape) - 1
cramers_v = np.sqrt((chi2 / n) / min_dim)

pd.DataFrame({
    "teste": ["Qui-quadrado Regiao x Sistema", "Cramer_V Regiao x Sistema"],
    "estatistica": [chi2, cramers_v],
    "p_valor": [p_chi2, np.nan]
}).to_csv(PASTA_TABELAS / "09_teste_associacao_regiao_sistema.csv", index=False, sep=";", decimal=",")

print(f"\nCramer's V entre Regiao e Sistema: {cramers_v:.4f}")

# %% In[4]: Funções auxiliares para testes e métricas

def calcular_metricas(y_real, y_pred, nome_modelo):
    """Calcula métricas em escala original."""
    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    wmape = np.sum(np.abs(y_real - y_pred)) / np.sum(np.abs(y_real)) * 100
    return {
        "modelo": nome_modelo,
        "R2_escala_original": r2,
        "RMSE_escala_original": rmse,
        "MAE_escala_original": mae,
        "MAPE_percentual": mape,
        "WMAPE_percentual": wmape
    }


def salvar_summary(modelo, caminho):
    """Salva o summary textual do statsmodels."""
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(str(modelo.summary()))


def teste_breusch_pagan(modelo):
    """Teste de Breusch-Pagan para heterocedasticidade."""
    bp = het_breuschpagan(modelo.resid, modelo.model.exog)
    return pd.DataFrame({
        "estatistica": ["LM statistic", "LM p-value", "F statistic", "F p-value"],
        "valor": list(bp)
    })


def teste_normalidade_residuos(residuos, nome_modelo):
    """Aplica Jarque-Bera e, se disponível, Shapiro-Francia."""
    resultados = []

    jb_stat, jb_p = stats.jarque_bera(residuos)
    resultados.append({
        "modelo": nome_modelo,
        "teste": "Jarque-Bera",
        "estatistica": jb_stat,
        "p_valor": jb_p
    })

    try:
        from statstests.tests import shapiro_francia
        teste_sf = shapiro_francia(residuos)
        method, statistics_W, statistics_z, p = teste_sf.items()
        resultados.append({
            "modelo": nome_modelo,
            "teste": "Shapiro-Francia",
            "estatistica": statistics_W[1],
            "p_valor": p[1]
        })
    except Exception:
        resultados.append({
            "modelo": nome_modelo,
            "teste": "Shapiro-Francia",
            "estatistica": np.nan,
            "p_valor": np.nan
        })

    return pd.DataFrame(resultados)


def calcular_vif(formula, data):
    """Calcula VIF a partir de uma fórmula statsmodels/patsy."""
    y, X = dmatrices(formula, data=data, return_type="dataframe")
    X = X.drop(columns=["Intercept"], errors="ignore")
    vif = pd.DataFrame({
        "variavel": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    vif["Tolerancia"] = 1 / vif["VIF"]
    return vif.sort_values("VIF", ascending=False)


def lrtest(modelo_restrito, modelo_completo, graus_liberdade=1):
    """Teste de razão de verossimilhança, conforme lógica da aula multinível."""
    ll_restrito = modelo_restrito.llf
    ll_completo = modelo_completo.llf
    lr_statistic = -2 * (ll_restrito - ll_completo)
    p_val = stats.chi2.sf(lr_statistic, graus_liberdade)
    return pd.DataFrame({
        "LL_modelo_restrito": [ll_restrito],
        "LL_modelo_completo": [ll_completo],
        "LR_statistic": [lr_statistic],
        "graus_liberdade": [graus_liberdade],
        "p_valor": [p_val]
    })

# %% In[5]: Modelo 1 - Regressão Linear Múltipla com Box-Cox

#############################################################################
# Modelo 1: Regressão Linear Múltipla com Box-Cox
# - Y transformado por Box-Cox: bc_Consumo
# - X quantitativas: ln_Consumidores, TempMedia e Tendencia
# - X qualitativas como dummies n-1: Sistema, Classe, TipoConsumidor e Mes
#############################################################################

formula_boxcox = (
    "bc_Consumo ~ ln_Consumidores_c + TempMedia_c + Tendencia_c + "
    "C(Sistema) + C(Classe) + C(TipoConsumidor) + C(Mes)"
)

modelo_boxcox = sm.OLS.from_formula(formula_boxcox, data=df_model).fit()
print("\nResumo do Modelo 1 - Regressão Linear Múltipla com Box-Cox")
print(modelo_boxcox.summary())

salvar_summary(modelo_boxcox, PASTA_MODELOS / "01_summary_modelo_boxcox.txt")

# Fitted values no espaço Box-Cox e retorno para a escala original.
df_model["yhat_boxcox_bc"] = modelo_boxcox.fittedvalues
# Na inversão de Box-Cox, é necessário garantir que lambda*y + 1 > 0.
# Pequenos valores ajustados fora do domínio são truncados ao limite matemático
# para permitir a comparação em escala original.
limite_inferior_bc = (-1 / lambda_boxcox) + 1e-8
df_model["yhat_boxcox_bc_clip"] = np.maximum(df_model["yhat_boxcox_bc"], limite_inferior_bc)
df_model["yhat_boxcox_original"] = inv_boxcox(df_model["yhat_boxcox_bc_clip"], lambda_boxcox)
df_model["resid_boxcox_bc"] = modelo_boxcox.resid
df_model["resid_boxcox_original"] = df_model["Consumo"] - df_model["yhat_boxcox_original"]

# Coeficientes e intervalos de confiança.
coef_boxcox = pd.DataFrame({
    "coeficiente": modelo_boxcox.params,
    "erro_padrao": modelo_boxcox.bse,
    "t": modelo_boxcox.tvalues,
    "p_valor": modelo_boxcox.pvalues
})
coef_boxcox = coef_boxcox.join(modelo_boxcox.conf_int(alpha=0.05).rename(columns={0: "IC_2_5", 1: "IC_97_5"}))
coef_boxcox.to_csv(PASTA_TABELAS / "10_coeficientes_modelo_boxcox.csv", sep=";", decimal=",")

# VIF e tolerância do modelo Box-Cox.
vif_boxcox = calcular_vif(formula_boxcox, df_model)
vif_boxcox.to_csv(PASTA_TABELAS / "11_vif_modelo_boxcox.csv", index=False, sep=";", decimal=",")
print("\nMaiores VIFs do Modelo Box-Cox:")
print(vif_boxcox.head(15))

# Testes dos resíduos do modelo Box-Cox.
teste_bp_boxcox = teste_breusch_pagan(modelo_boxcox)
teste_bp_boxcox.to_csv(PASTA_TABELAS / "12_breusch_pagan_modelo_boxcox.csv", index=False, sep=";", decimal=",")

teste_norm_boxcox = teste_normalidade_residuos(modelo_boxcox.resid, "Modelo Box-Cox")
teste_norm_boxcox.to_csv(PASTA_TABELAS / "13_normalidade_residuos_modelo_boxcox.csv", index=False, sep=";", decimal=",")

# Gráficos do modelo Box-Cox.
plt.figure(figsize=(15, 8))
df_plot_boxcox = df_model.sample(n=min(5000, len(df_model)), random_state=42)
sns.scatterplot(data=df_plot_boxcox, x="Consumo", y="yhat_boxcox_original", alpha=0.35, s=20)
plt.plot([df_model["Consumo"].min(), df_model["Consumo"].max()],
         [df_model["Consumo"].min(), df_model["Consumo"].max()],
         linestyle="--")
plt.title("Modelo Box-Cox: consumo observado x consumo ajustado (kWh)", fontsize=18)
plt.xlabel("Consumo observado (kWh)", fontsize=14)
plt.ylabel("Consumo ajustado (kWh)", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "09_boxcox_real_vs_fitted.png", dpi=300)
plt.close()

plt.figure(figsize=(15, 8))
resid_plot_boxcox = pd.DataFrame({"fitted": modelo_boxcox.fittedvalues, "resid": modelo_boxcox.resid}).sample(n=min(5000, len(df_model)), random_state=42)
sns.scatterplot(data=resid_plot_boxcox, x="fitted", y="resid", alpha=0.35, s=20)
plt.axhline(0, linestyle="--")
plt.title("Modelo Box-Cox: resíduos x valores ajustados na escala transformada", fontsize=18)
plt.xlabel("Valores ajustados na escala Box-Cox", fontsize=14)
plt.ylabel("Resíduos na escala Box-Cox", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "10_boxcox_residuos_vs_fitted.png", dpi=300)
plt.close()

plt.figure(figsize=(15, 8))
sns.histplot(modelo_boxcox.resid, kde=False, bins=40, color="steelblue")
plt.title("Modelo Box-Cox: distribuição dos resíduos na escala transformada", fontsize=18)
plt.xlabel("Resíduos na escala Box-Cox", fontsize=14)
plt.ylabel("Frequência", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "11_boxcox_histograma_residuos.png", dpi=300)
plt.close()

# %% In[6]: Modelo 2 - Modelo Multinível HLM2 final log-linear

#############################################################################
# Modelo 2: Modelo Multinível HLM2 final log-linear
# - Nível 1: observações mensais
# - Nível 2: Grupo_HLM2 = Sistema + Classe + TipoConsumidor
# - Y: ln_Consumo
# - Efeitos fixos: ln_Consumidores, TempMedia, Tendencia e Mes
# - Efeitos aleatórios: intercepto aleatório por grupo
#############################################################################

formula_hlm2 = "ln_Consumo ~ ln_Consumidores_c + TempMedia_c + Tendencia_c + C(Mes)"

# O modelo nulo é estimado apenas como referência auxiliar para o teste de razão
# de verossimilhança, seguindo a prática da aula multinível.
modelo_nulo_hlm2 = sm.MixedLM.from_formula(
    formula="ln_Consumo ~ 1",
    groups="Grupo_HLM2",
    re_formula="1",
    data=df_model
).fit(method="lbfgs", reml=False, maxiter=500, disp=False)

# Modelo HLM2 final.
modelo_hlm2_final = sm.MixedLM.from_formula(
    formula=formula_hlm2,
    groups="Grupo_HLM2",
    re_formula="1",
    data=df_model
).fit(method="lbfgs", reml=False, maxiter=500, disp=False)

print("\nResumo do Modelo 2 - HLM2 final log-linear")
print(modelo_hlm2_final.summary())

salvar_summary(modelo_nulo_hlm2, PASTA_MODELOS / "02_summary_modelo_nulo_hlm2_auxiliar.txt")
salvar_summary(modelo_hlm2_final, PASTA_MODELOS / "03_summary_modelo_hlm2_final.txt")

# Teste de razão de verossimilhança: HLM2 nulo auxiliar vs HLM2 final.
# Graus de liberdade aproximados: diferença no número de parâmetros estimados.
gl_lr = int(abs(len(modelo_hlm2_final.params) - len(modelo_nulo_hlm2.params)))
gl_lr = max(gl_lr, 1)
teste_lr_hlm2 = lrtest(modelo_nulo_hlm2, modelo_hlm2_final, graus_liberdade=gl_lr)
teste_lr_hlm2.to_csv(PASTA_TABELAS / "14_lrtest_hlm2_nulo_vs_final.csv", index=False, sep=";", decimal=",")
print("\nTeste de razão de verossimilhança - HLM2 nulo auxiliar vs HLM2 final:")
print(teste_lr_hlm2)

# Fitted values do HLM2. Em MixedLM, fittedvalues considera os efeitos fixos e
# aleatórios estimados para os grupos observados.
df_model["yhat_hlm2_log"] = modelo_hlm2_final.fittedvalues
df_model["yhat_hlm2_original"] = np.exp(df_model["yhat_hlm2_log"])
df_model["resid_hlm2_log"] = modelo_hlm2_final.resid
df_model["resid_hlm2_original"] = df_model["Consumo"] - df_model["yhat_hlm2_original"]

# Coeficientes fixos do HLM2.
coef_hlm2 = pd.DataFrame({
    "coeficiente": modelo_hlm2_final.fe_params,
    "erro_padrao": modelo_hlm2_final.bse_fe,
    "z": modelo_hlm2_final.fe_params / modelo_hlm2_final.bse_fe,
    "p_valor": modelo_hlm2_final.pvalues[modelo_hlm2_final.fe_params.index]
})
coef_hlm2.to_csv(PASTA_TABELAS / "15_coeficientes_fixos_modelo_hlm2_final.csv", sep=";", decimal=",")

# Efeitos aleatórios do HLM2.
efeitos_aleatorios = pd.DataFrame(modelo_hlm2_final.random_effects).T.reset_index()
efeitos_aleatorios = efeitos_aleatorios.rename(columns={"index": "Grupo_HLM2"})
efeitos_aleatorios.to_csv(PASTA_TABELAS / "16_efeitos_aleatorios_modelo_hlm2_final.csv", index=False, sep=";", decimal=",")

# Teste de normalidade dos resíduos do HLM2 em escala log.
teste_norm_hlm2 = teste_normalidade_residuos(modelo_hlm2_final.resid, "Modelo HLM2 final")
teste_norm_hlm2.to_csv(PASTA_TABELAS / "17_normalidade_residuos_modelo_hlm2_final.csv", index=False, sep=";", decimal=",")

# Gráficos do HLM2.
plt.figure(figsize=(15, 8))
df_plot_hlm2 = df_model.sample(n=min(5000, len(df_model)), random_state=42)
sns.scatterplot(data=df_plot_hlm2, x="Consumo", y="yhat_hlm2_original", alpha=0.35, s=20)
plt.plot([df_model["Consumo"].min(), df_model["Consumo"].max()],
         [df_model["Consumo"].min(), df_model["Consumo"].max()],
         linestyle="--")
plt.title("HLM2 final: consumo observado x consumo ajustado (kWh)", fontsize=18)
plt.xlabel("Consumo observado (kWh)", fontsize=14)
plt.ylabel("Consumo ajustado (kWh)", fontsize=14)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "12_hlm2_real_vs_fitted.png", dpi=300)
plt.close()



# %% In[7]: Comparação final dos dois modelos

print("\nIniciando comparação final dos dois modelos...")
metricas = pd.DataFrame([
    calcular_metricas(df_model["Consumo"], df_model["yhat_boxcox_original"], "Regressão Linear Múltipla com Box-Cox"),
    calcular_metricas(df_model["Consumo"], df_model["yhat_hlm2_original"], "Modelo Multinível HLM2 final log-linear")
])

# Métricas adicionais do espaço de estimação.
metricas["AIC"] = [modelo_boxcox.aic, modelo_hlm2_final.aic]
metricas["BIC"] = [modelo_boxcox.bic, modelo_hlm2_final.bic]
metricas["LogLik"] = [modelo_boxcox.llf, modelo_hlm2_final.llf]
metricas["N"] = [int(modelo_boxcox.nobs), int(modelo_hlm2_final.nobs)]
metricas["lambda_boxcox"] = [lambda_boxcox, np.nan]

print("Métricas calculadas. Salvando tabela de comparação...")
metricas.to_csv(PASTA_TABELAS / "18_comparacao_final_dois_modelos.csv", index=False, sep=";", decimal=",")
print("\nComparação final dos dois modelos:")
print(metricas)

print("Tabela de comparação salva. Gerando gráfico simples de comparação...")
# Gráfico de comparação de métricas principais.
metricas_plot = metricas.melt(
    id_vars="modelo",
    value_vars=["R2_escala_original", "RMSE_escala_original", "MAE_escala_original", "WMAPE_percentual"],
    var_name="metrica",
    value_name="valor"
)

# Rótulos mais claros para apresentação dos gráficos.
# RMSE e MAE estão em kWh; WMAPE está em percentual; R² é adimensional.
rotulos_metricas = {
    "R2_escala_original": "R² (adimensional)",
    "RMSE_escala_original": "RMSE (kWh)",
    "MAE_escala_original": "MAE (kWh)",
    "WMAPE_percentual": "WMAPE (%)"
}
metricas_plot["metrica"] = metricas_plot["metrica"].replace(rotulos_metricas)

plt.figure(figsize=(16, 9))
sns.barplot(data=metricas_plot, x="metrica", y="valor", hue="modelo", palette="viridis")
plt.title("Comparação final dos dois modelos por métrica", fontsize=18)
plt.xlabel("Métrica de desempenho", fontsize=14)
plt.ylabel("Valor da métrica (R², kWh ou %)", fontsize=14)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "16_comparacao_final_modelos.png", dpi=300)
plt.close()

# Tabela com fitted values e resíduos.
colunas_saida = [
    "DataRef", "Regiao", "Sistema", "Classe", "TipoConsumidor",
    "Consumo", "Consumidores", "TempMedia", "Grupo_HLM2",
    "yhat_boxcox_original", "resid_boxcox_original",
    "yhat_hlm2_original", "resid_hlm2_original"
]
print("Salvando fitted values e resíduos...")
df_model[colunas_saida].to_csv(PASTA_TABELAS / "19_fitted_values_residuos_modelos_finais.csv", index=False, sep=";", decimal=",")
print("Fitted values e resíduos salvos.")

# Observação: os modelos possuem variáveis dependentes em escalas diferentes,
# portanto os coeficientes não devem ser comparados diretamente.
with open(PASTA_MODELOS / "04_observacao_comparacao_coeficientes.txt", "w", encoding="utf-8") as f:
    f.write("Os modelos Box-Cox e HLM2 foram estimados em escalas diferentes. "
            "A comparação final deve priorizar métricas na escala original de Consumo, "
            "AIC/BIC/LogLik e os diagnósticos de resíduos salvos nas pastas de resultados.")

print("Gerando relatório sintético...")
# %% In[8]: Relatório sintético automático

melhor_rmse = metricas.sort_values("RMSE_escala_original").iloc[0]["modelo"]
melhor_mae = metricas.sort_values("MAE_escala_original").iloc[0]["modelo"]
melhor_wmape = metricas.sort_values("WMAPE_percentual").iloc[0]["modelo"]

relatorio = f"""
# Relatório sintético - modelos finais de previsão de consumo de energia

## Base utilizada

Arquivo carregado: `{ARQUIVO_BASE}`

Registros na base original: {n_original:,}
Registros usados na modelagem: {n_modelagem:,}
Registros removidos por `Consumo <= 0` ou `Consumidores <= 0`: {n_original - n_modelagem:,}

A exclusão foi necessária porque os dois modelos finais utilizam transformações que exigem valores positivos: `ln(Consumo)`, `ln(Consumidores)` e `Box-Cox(Consumo)`.

## Modelos desenvolvidos

### 1º modelo: Regressão Linear Múltipla com Box-Cox

Fórmula estimada:

`{formula_boxcox}`

Lambda estimado para Box-Cox: {lambda_boxcox:.6f}

Esse modelo segue a rota de regressão linear da aula: variável dependente transformada por Box-Cox, variáveis quantitativas e variáveis qualitativas representadas por dummies n-1.

### 2º modelo: Modelo Multinível HLM2 final log-linear

Fórmula dos efeitos fixos:

`{formula_hlm2}`

Grupo de nível 2:

`Grupo_HLM2 = Sistema + Classe + TipoConsumidor`

Esse modelo segue a rota de modelagem multinível da aula: observações mensais no nível 1 e grupos contextuais no nível 2, com intercepto aleatório por grupo.

## Diagnóstico Região x Sistema

Cramer's V entre `Regiao` e `Sistema`: {cramers_v:.4f}

A variável `Regiao` foi mantida apenas para diagnóstico. Como o foco da aplicação é o setor energético, o modelo final usa `Sistema` como variável setorial principal.

## Comparação dos modelos

{metricas.to_markdown(index=False)}

Melhor modelo por RMSE: **{melhor_rmse}**

Melhor modelo por MAE: **{melhor_mae}**

Melhor modelo por WMAPE: **{melhor_wmape}**

## Observação metodológica

Os dois modelos usam variáveis dependentes em escalas diferentes no processo de estimação. Por isso, a comparação operacional foi feita após retornar os fitted values para a escala original de `Consumo`.

O modelo Box-Cox tende a ser a melhor escolha quando o foco é regressão linear clássica com transformação da variável resposta. O HLM2 tende a ser a melhor escolha quando o foco é representar a estrutura agrupada dos dados por sistema elétrico, classe e tipo de consumidor.
"""

with open(PASTA_RESULTADOS / "relatorio_sintetico_modelos_finais.md", "w", encoding="utf-8") as f:
    f.write(relatorio)

print("\nArquivos gerados em:")
print(PASTA_RESULTADOS.resolve())
print("\nProcessamento concluído com sucesso.")
