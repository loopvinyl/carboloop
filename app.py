# IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS (mesmo do script anterior)
import requests
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
from joblib import Parallel, delayed
import warnings
from matplotlib.ticker import FuncFormatter
from SALib.sample.sobol import sample
from SALib.analyze.sobol import analyze
import yfinance as yf

# Semente fixa e configurações da página
np.random.seed(50)
st.set_page_config(
    page_title="Comparação de Tecnologias de Compostagem para Créditos de Carbono",
    layout="wide"
)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# =============================================================================
# PARÂMETROS (mesmos do script anterior)
# =============================================================================
CAPTURE_FRACTION_BASELINE = 0.6
MCF_BASELINE = 1.0
OX_BASELINE = 0.1
PHI_BASELINE = 0.85

TOC = 0.436
TN = 0.0142
F_CH4_THERMO = 0.0060
F_N2O_THERMO = 0.0196

EF_CH4_WINDROW = 0.002
EF_N2O_WINDROW = 0.0005

# Classe GHGEmissionCalculator (exatamente igual à última versão funcional)
class GHGEmissionCalculator:
    def __init__(self):
        self.MCF = MCF_BASELINE
        self.F = 0.5
        self.OX = OX_BASELINE
        self.Ri = 0.0
        self.TOC = TOC
        self.TN = TN
        self.f_CH4_thermo = F_CH4_THERMO
        self.f_N2O_thermo = F_N2O_THERMO
        self.EF_CH4_windrow = EF_CH4_WINDROW
        self.EF_N2O_windrow = EF_N2O_WINDROW
        self.COMPOSTING_DAYS = 50
        self.GWP_CH4_20 = 79.7
        self.GWP_N2O_20 = 273
        self._load_profiles()
        self._setup_pre_disposal()

    def _load_profiles(self):
        self.profile_ch4 = np.array([
            0.02,0.02,0.02,0.03,0.03,0.04,0.04,0.05,0.05,0.06,
            0.07,0.08,0.09,0.10,0.09,0.08,0.07,0.06,0.05,0.04,
            0.03,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.01,0.01,
            0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,
            0.002,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.001,0.001
        ])
        self.profile_ch4 /= self.profile_ch4.sum()
        self.profile_n2o = np.array([
            0.10,0.08,0.15,0.05,0.03,0.04,0.05,0.07,0.10,0.12,
            0.15,0.18,0.20,0.18,0.15,0.12,0.10,0.08,0.06,0.05,
            0.04,0.03,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.01,
            0.005,0.005,0.005,0.005,0.005,0.002,0.002,0.002,0.002,0.002,
            0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001
        ])
        self.profile_n2o /= self.profile_n2o.sum()
        self.profile_n2o_landfill = {1:0.10,2:0.30,3:0.40,4:0.15,5:0.05}

    def _setup_pre_disposal(self):
        CH4_pre_ugC_per_kg_h = 2.78
        self.CH4_pre_kg_per_kg_day = CH4_pre_ugC_per_kg_h * (16/12) * 24 / 1_000_000_000
        N2O_pre_mgN_per_kg_total = 20.26
        self.N2O_pre_kg_per_kg_total = N2O_pre_mgN_per_kg_total * (44/28) / 1_000_000
        self.profile_n2o_pre = {1:0.8623,2:0.10,3:0.0377}

    def calculate_landfill_emissions(self, w_kg_day, k, temp, doc, umid, years=20,
                                     phi=PHI_BASELINE, capt=CAPTURE_FRACTION_BASELINE):
        days = years*365
        docf = 0.0147*temp + 0.28
        ch4_pot_kg = (doc * docf * self.MCF * self.F * (16/12) * (1-self.Ri) * (1-self.OX)) * w_kg_day
        t = np.arange(1, days+1, dtype=float)
        kernel = np.exp(-k*(t-1)/365.0) - np.exp(-k*t/365.0)
        ch4 = np.convolve(np.ones(days), kernel, mode='full')[:days] * ch4_pot_kg
        ch4 = ch4 * phi * (1 - capt)
        opening_factor = min(1.0, (100/w_kg_day)*(8/24))
        E_avg = opening_factor*1.91 + (1-opening_factor)*2.15
        moisture_factor = (1-umid)/(1-0.55)
        daily_n2o_kg = (E_avg * moisture_factor * (44/28) / 1_000_000) * w_kg_day
        kernel_n2o = np.array([self.profile_n2o_landfill.get(d,0) for d in range(1,6)])
        n2o = np.convolve(np.full(days, daily_n2o_kg), kernel_n2o, mode='full')[:days]
        ch4_pre, n2o_pre = self._pre_disposal(w_kg_day, days)
        return ch4 + ch4_pre, n2o + n2o_pre

    def _pre_disposal(self, w_kg_day, days):
        ch4 = np.full(days, w_kg_day * self.CH4_pre_kg_per_kg_day)
        n2o = np.zeros(days)
        for e in range(days):
            for dd, frac in self.profile_n2o_pre.items():
                idx = e + dd - 1
                if idx < days:
                    n2o[idx] += w_kg_day * self.N2O_pre_kg_per_kg_total * frac
        return ch4, n2o

    def calculate_thermophilic_emissions(self, w_kg_day, umid, years=20):
        days = years*365
        dry = 1 - umid
        ch4_batch = w_kg_day * self.TOC * self.f_CH4_thermo * (16/12) * dry
        n2o_batch = w_kg_day * self.TN * self.f_N2O_thermo * (44/28) * dry
        ch4 = np.zeros(days)
        n2o = np.zeros(days)
        for e in range(days):
            for d in range(self.COMPOSTING_DAYS):
                ed = e + d
                if ed < days:
                    ch4[ed] += ch4_batch * self.profile_ch4[d]
                    n2o[ed] += n2o_batch * self.profile_n2o[d]
        return ch4, n2o

    def calculate_windrow_emissions(self, w_kg_day, umid, years=20):
        days = years*365
        total_t = (w_kg_day * days) / 1000.0
        total_ch4_t = total_t * self.EF_CH4_windrow
        total_n2o_t = total_t * self.EF_N2O_windrow
        ch4_per_kg = self.EF_CH4_windrow / 1000.0
        n2o_per_kg = self.EF_N2O_windrow / 1000.0
        ch4_batch_kg = w_kg_day * ch4_per_kg
        n2o_batch_kg = w_kg_day * n2o_per_kg
        ch4 = np.zeros(days)
        n2o = np.zeros(days)
        for e in range(days):
            for d in range(self.COMPOSTING_DAYS):
                ed = e + d
                if ed < days:
                    ch4[ed] += ch4_batch_kg * self.profile_ch4[d]
                    n2o[ed] += n2o_batch_kg * self.profile_n2o[d]
        return ch4, n2o

    def calculate_avoided_emissions(self, w_kg_day, k, temp, doc, umid, years):
        ch4_l, n2o_l = self.calculate_landfill_emissions(w_kg_day, k, temp, doc, umid, years)
        ch4_t, n2o_t = self.calculate_thermophilic_emissions(w_kg_day, umid, years)
        ch4_w, n2o_w = self.calculate_windrow_emissions(w_kg_day, umid, years)
        base = (ch4_l*self.GWP_CH4_20 + n2o_l*self.GWP_N2O_20)/1000
        thermo = (ch4_t*self.GWP_CH4_20 + n2o_t*self.GWP_N2O_20)/1000
        wind = (ch4_w*self.GWP_CH4_20 + n2o_w*self.GWP_N2O_20)/1000
        return {
            'baseline': base.sum(),
            'thermo_avoided': base.sum() - thermo.sum(),
            'wind_avoided': base.sum() - wind.sum(),
            'base_series': base, 'thermo_series': thermo, 'wind_series': wind
        }


# =============================================================================
# FUNÇÕES DE COTAÇÃO, FORMATAÇÃO E INTERFACE (mantidas iguais)
# =============================================================================
def obter_cotacao_carbono():
    try:
        ticker = yf.Ticker("CO2.L")
        data = ticker.history(period="1d")
        if not data.empty:
            preco = data['Close'].iloc[-1]
            if 10 < preco < 200:
                return preco, "€", "Carbon Futures (CO2.L)", True, "Yahoo Finance (CO2.L)"
        return 85.50, "€", "Carbon Emissions (Referência)", False, "Referência"
    except Exception:
        return 85.50, "€", "Carbon Emissions (Referência)", False, "Referência"

def obter_cotacao_euro_real():
    try:
        url = "https://economia.awesomeapi.com.br/last/EUR-BRL"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return float(data['EURBRL']['bid']), "R$", True, "AwesomeAPI"
    except:
        pass
    try:
        url = "https://api.exchangerate-api.com/v4/latest/EUR"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data['rates']['BRL'], "R$", True, "ExchangeRate-API"
    except:
        pass
    return 5.50, "R$", False, "Referência"

def calcular_valor_creditos(emissoes_evitadas_tco2eq, preco_carbono_por_tonelada, moeda, taxa_cambio=1):
    return emissoes_evitadas_tco2eq * preco_carbono_por_tonelada * taxa_cambio

def formatar_br(numero):
    if pd.isna(numero):
        return "N/A"
    numero = round(numero, 2)
    return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def br_format(x, pos):
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        return f"{x:.1e}".replace(".", ",")
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def exibir_cotacao_carbono():
    st.sidebar.header("💰 Mercado de Carbono e Câmbio")
    if not st.session_state.get('cotacao_carregada', False):
        st.session_state.mostrar_atualizacao = True
        st.session_state.cotacao_carregada = True

    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        if st.button("🔄 Atualizar Cotações", key="atualizar_cotacoes"):
            st.session_state.cotacao_atualizada = True
            st.session_state.mostrar_atualizacao = True

    if st.session_state.get('mostrar_atualizacao', False):
        st.sidebar.info("🔄 Atualizando cotações...")
        preco_carbono, moeda, _, _, fonte_carbono = obter_cotacao_carbono()
        preco_euro, moeda_real, _, _ = obter_cotacao_euro_real()
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        st.session_state.fonte_cotacao = fonte_carbono
        st.session_state.mostrar_atualizacao = False
        st.session_state.cotacao_atualizada = False
        st.rerun()

    st.sidebar.metric(
        label="Preço do Carbono (tCO₂eq)",
        value=f"{st.session_state.moeda_carbono} {formatar_br(st.session_state.preco_carbono)}",
        help=f"Fonte: {st.session_state.fonte_cotacao}"
    )
    st.sidebar.metric(
        label="Euro (EUR/BRL)",
        value=f"{st.session_state.moeda_real} {formatar_br(st.session_state.taxa_cambio)}",
        help="Cotação do Euro em Reais Brasileiros"
    )
    preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
    st.sidebar.metric(
        label="Carbono em Reais (tCO₂eq)",
        value=f"R$ {formatar_br(preco_carbono_reais)}",
        help="Preço do carbono convertido para Reais Brasileiros"
    )

def inicializar_session_state():
    if 'preco_carbono' not in st.session_state:
        p, m, _, _, f = obter_cotacao_carbono()
        st.session_state.preco_carbono = p
        st.session_state.moeda_carbono = m
        st.session_state.fonte_cotacao = f
    if 'taxa_cambio' not in st.session_state:
        euro, real, _, _ = obter_cotacao_euro_real()
        st.session_state.taxa_cambio = euro
        st.session_state.moeda_real = real
    if 'moeda_real' not in st.session_state:
        st.session_state.moeda_real = "R$"
    if 'cotacao_atualizada' not in st.session_state:
        st.session_state.cotacao_atualizada = False
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    if 'mostrar_atualizacao' not in st.session_state:
        st.session_state.mostrar_atualizacao = False
    if 'cotacao_carregada' not in st.session_state:
        st.session_state.cotacao_carregada = False
    if 'k_ano' not in st.session_state:
        st.session_state.k_ano = 0.06

inicializar_session_state()

# INTERFACE PRINCIPAL
st.title("Comparação de Tecnologias de Compostagem para Créditos de Carbono")
st.markdown("""
Esta ferramenta compara **duas tecnologias de compostagem** (termofílica e em leiras) com o **cenário baseline (aterro sanitário)** calibrado para Ribeirão Preto (aterro CGR Guatapará com captura de biogás).  
**Estatísticas de diferença significativa** entre as emissões evitadas são calculadas via Monte Carlo.

**Metodologias:**  
- **Baseline:** A6.4‑AMT‑003 (MCF=1,0; captura=60%; φ=0,85)  
- **Termofílica:** Yang et al. (2017) – CH₄=0,0060 t/tC; N₂O=0,0196 t/tN  
- **Leiras:** TOOL13 (2017) – CH₄=0,002 t/t úmido; N₂O=0,0005 t/t úmido
""")

exibir_cotacao_carbono()

with st.sidebar:
    st.header("⚙️ Parâmetros")
    residuos_kg_dia = st.slider("Resíduos (kg/dia)", 10, 1000, 100, 10)
    opcao_k = st.selectbox("k (ano⁻¹)", ["0,06 (lento)", "0,40 (rápido)"], index=0)
    k_ano = 0.40 if "0,40" in opcao_k else 0.06
    st.session_state.k_ano = k_ano
    T = st.slider("Temperatura (°C)", 20, 40, 25, 1)
    DOC = st.slider("DOC (fração)", 0.10, 0.25, 0.15, 0.01)
    umidade_valor = st.slider("Umidade (%)", 50, 95, 85, 1)
    umidade = umidade_valor / 100.0
    anos_simulacao = st.slider("Anos de simulação", 5, 50, 20, 5)
    n_simulations = st.slider("Monte Carlo (n)", 50, 1000, 100, 50)
    n_samples = st.slider("Sobol (amostras)", 32, 256, 64, 16)
    if st.button("🚀 Executar Simulação", type="primary"):
        st.session_state.run_simulation = True

# =============================================================================
# FUNÇÕES PARA SOBOL E MONTE CARLO (serão chamadas após os resultados rápidos)
# =============================================================================
def sobol_thermo(params, gwp_ch4, gwp_n2o):
    k, temp, doc = params
    np.random.seed(50)
    calc = GHGEmissionCalculator()
    calc.GWP_CH4_20 = gwp_ch4
    calc.GWP_N2O_20 = gwp_n2o
    res = calc.calculate_avoided_emissions(residuos_kg_dia, k, temp, doc, umidade, anos_simulacao)
    return res['thermo_avoided']

def sobol_windrow(params, gwp_ch4, gwp_n2o):
    k, temp, doc = params
    np.random.seed(50)
    calc = GHGEmissionCalculator()
    calc.GWP_CH4_20 = gwp_ch4
    calc.GWP_N2O_20 = gwp_n2o
    res = calc.calculate_avoided_emissions(residuos_kg_dia, k, temp, doc, umidade, anos_simulacao)
    return res['wind_avoided']

def gerar_parametros_mc(n):
    np.random.seed(50)
    u = np.random.uniform(0.75, 0.90, n)
    t = np.random.normal(25, 3, n)
    d = np.random.triangular(0.12, 0.15, 0.18, n)
    return u, t, d


# =============================================================================
# EXECUÇÃO PRINCIPAL (com ordem de exibição progressiva)
# =============================================================================
if st.session_state.get('run_simulation', False):

    # -------------------- 1. RESULTADOS DETERMINÍSTICOS (rápidos) --------------------
    with st.spinner("Calculando resultados determinísticos..."):
        calc = GHGEmissionCalculator()
        # Usar GWP-20 para os gráficos principais
        calc.GWP_CH4_20, calc.GWP_N2O_20 = (79.7, 273)
        res_det = calc.calculate_avoided_emissions(residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao)
        evitado_thermo = res_det['thermo_avoided']
        evitado_windrow = res_det['wind_avoided']

        # Séries diárias para gráficos
        base_series = res_det['base_series']
        thermo_series = res_det['thermo_series']
        wind_series = res_det['wind_series']

        # Cálculo anual para gráfico de barras
        dias_total = len(base_series)
        datas = pd.date_range(start=datetime.now(), periods=dias_total, freq='D')
        df_dia = pd.DataFrame({'Data': datas, 'base': base_series, 'thermo': thermo_series, 'wind': wind_series})
        df_dia['Year'] = df_dia['Data'].dt.year
        df_anual = df_dia.groupby('Year').agg({'base':'sum','thermo':'sum','wind':'sum'}).reset_index()
        df_anual['Evitado_Thermo'] = df_anual['base'] - df_anual['thermo']
        df_anual['Evitado_Wind'] = df_anual['base'] - df_anual['wind']

    # Exibição imediata dos resultados rápidos
    st.header("📈 Resultados da Simulação (GWP-20)")
    st.info(f"""
    **Parâmetros calibrados para Ribeirão Preto:**  
    - k = {formatar_br(k_ano)} ano⁻¹, T = {formatar_br(T)} °C, DOC = {formatar_br(DOC)}, Umidade = {formatar_br(umidade_valor)}%  
    - Resíduos totais: {formatar_br(residuos_kg_dia * 365 * anos_simulacao / 1000)} t  
    - **Aterro:** MCF = 1,0; captura = 60%; φ = 0,85  
    - **Termofílica:** Yang et al. (2017)  
    - **Leiras:** TOOL13 (0,002 t CH₄/t; 0,0005 t N₂O/t)
    """)

    st.subheader("💰 Valor Financeiro (Cenário Otimista)")
    preco = st.session_state.preco_carbono
    moeda = st.session_state.moeda_carbono
    cambio = st.session_state.taxa_cambio
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Termofílica - Evitado", f"{formatar_br(evitado_thermo)} tCO₂eq")
        st.metric("Valor (Euro)", f"{moeda} {formatar_br(evitado_thermo * preco)}")
        st.metric("Valor (R$)", f"R$ {formatar_br(evitado_thermo * preco * cambio)}")
    with col2:
        st.metric("Leiras - Evitado", f"{formatar_br(evitado_windrow)} tCO₂eq")
        st.metric("Valor (Euro)", f"{moeda} {formatar_br(evitado_windrow * preco)}")
        st.metric("Valor (R$)", f"R$ {formatar_br(evitado_windrow * preco * cambio)}")

    st.subheader("📊 Comparação Anual das Emissões Evitadas")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_anual['Year']))
    width = 0.35
    ax.bar(x - width/2, df_anual['Evitado_Thermo'], width, label='Termofílica', edgecolor='black', color='orange')
    ax.bar(x + width/2, df_anual['Evitado_Wind'], width, label='Leiras (TOOL13)', edgecolor='black', color='green', hatch='//')
    for i, (v1, v2) in enumerate(zip(df_anual['Evitado_Thermo'], df_anual['Evitado_Wind'])):
        ax.text(i - width/2, v1 + max(v1,v2)*0.01, formatar_br(v1), ha='center', fontsize=9, fontweight='bold')
        ax.text(i + width/2, v2 + max(v1,v2)*0.01, formatar_br(v2), ha='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Emissões Evitadas (t CO₂eq)')
    ax.set_title('Comparação Anual: Termofílica vs Leiras')
    ax.set_xticks(x)
    ax.set_xticklabels(df_anual['Year'], fontsize=8)
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(br_format))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📉 Redução de Emissões Acumulada")
    base_acum = np.cumsum(base_series)
    thermo_acum = np.cumsum(thermo_series)
    wind_acum = np.cumsum(wind_series)
    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.plot(datas, base_acum, 'r-', label='Baseline (Aterro)', linewidth=2)
    ax2.plot(datas, thermo_acum, 'orange', label='Termofílica', linewidth=2)
    ax2.plot(datas, wind_acum, 'green', label='Leiras (TOOL13)', linewidth=2)
    ax2.fill_between(datas, thermo_acum, wind_acum, color='gray', alpha=0.3, label='Diferença entre tecnologias')
    ax2.set_title(f'Redução de Emissões em {anos_simulacao} anos (k = {formatar_br(k_ano)} ano⁻¹)')
    ax2.set_xlabel('Data')
    ax2.set_ylabel('tCO₂eq Acumulado')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.yaxis.set_major_formatter(FuncFormatter(br_format))
    st.pyplot(fig2)
    plt.close(fig2)

    # -------------------- 2. TABELA COMPARATIVA DOS TRÊS GWPs (rápido) --------------------
    st.subheader("📊 Comparação entre Cenários de GWP")
    gwps = {
        "Otimista (GWP-20)": (79.7, 273),
        "Realista (GWP-100)": (27.0, 273),
        "Pessimista (GWP-500)": (7.2, 130)
    }
    comparacao = []
    for nome, (gwp_ch4, gwp_n2o) in gwps.items():
        calc_temp = GHGEmissionCalculator()
        calc_temp.GWP_CH4_20 = gwp_ch4
        calc_temp.GWP_N2O_20 = gwp_n2o
        r = calc_temp.calculate_avoided_emissions(residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao)
        comparacao.append({
            "Cenário": nome,
            "Termofílica (tCO₂eq)": r['thermo_avoided'],
            "Leiras (tCO₂eq)": r['wind_avoided']
        })
    df_gwp = pd.DataFrame(comparacao)
    st.dataframe(df_gwp.style.format({c: lambda x: formatar_br(x) for c in df_gwp.columns if c != "Cenário"}))

    # -------------------- 3. ANÁLISE SOBOL (mais pesada) --------------------
    st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - GWP-20")
    problem = {'num_vars':3, 'names':['k','T','DOC'], 'bounds':[[0.06,0.40],[20,40],[0.10,0.25]]}
    param_values = sample(problem, n_samples, seed=50)
    g20_ch4, g20_n2o = gwps["Otimista (GWP-20)"]
    with st.spinner("Executando Sobol para termofílica..."):
        res_t = Parallel(n_jobs=1)(delayed(sobol_thermo)(p, g20_ch4, g20_n2o) for p in param_values)
        Si_t = analyze(problem, np.array(res_t), print_to_console=False)
    with st.spinner("Executando Sobol para leiras..."):
        res_w = Parallel(n_jobs=1)(delayed(sobol_windrow)(p, g20_ch4, g20_n2o) for p in param_values)
        Si_w = analyze(problem, np.array(res_w), print_to_console=False)
    df_sens = pd.DataFrame({
        'Parâmetro': ['k','T','DOC'],
        'S1_Termofílica': Si_t['S1'], 'ST_Termofílica': Si_t['ST'],
        'S1_Leiras': Si_w['S1'], 'ST_Leiras': Si_w['ST']
    })
    st.dataframe(df_sens.style.format({c:'{:.4f}' for c in df_sens.columns if c != 'Parâmetro'}))

    # -------------------- 4. MONTE CARLO E ESTATÍSTICAS (mais pesado) --------------------
    st.subheader("🎲 Análise de Incerteza (Monte Carlo) e Comparação Estatística")
    with st.spinner("Executando simulações Monte Carlo..."):
        u_mc, t_mc, d_mc = gerar_parametros_mc(n_simulations)
        arr_thermo_mc = []
        arr_wind_mc = []
        for i in range(n_simulations):
            calc_mc = GHGEmissionCalculator()
            calc_mc.GWP_CH4_20, calc_mc.GWP_N2O_20 = g20_ch4, g20_n2o
            r_mc = calc_mc.calculate_avoided_emissions(
                residuos_kg_dia, k_ano, t_mc[i], d_mc[i], u_mc[i], anos_simulacao
            )
            arr_thermo_mc.append(r_mc['thermo_avoided'])
            arr_wind_mc.append(r_mc['wind_avoided'])
        arr_thermo_mc = np.array(arr_thermo_mc)
        arr_wind_mc = np.array(arr_wind_mc)
        diff = arr_thermo_mc - arr_wind_mc

        shapiro_stat, shapiro_p = stats.shapiro(diff)
        t_stat, t_p = stats.ttest_rel(arr_thermo_mc, arr_wind_mc)
        w_stat, w_p = stats.wilcoxon(arr_thermo_mc, arr_wind_mc)

    st.write(f"**Teste de normalidade (Shapiro-Wilk) da diferença:** estatística = {shapiro_stat:.5f}, p = {shapiro_p:.5f}")
    st.write(f"**Teste t pareado:** t = {t_stat:.5f}, p = {t_p:.5f}")
    st.write(f"**Teste de Wilcoxon:** estatística = {w_stat:.5f}, p = {w_p:.5f}")

    stats_df = pd.DataFrame([
        {"Tecnologia": "Termofílica", "Média": np.mean(arr_thermo_mc), "Mediana": np.median(arr_thermo_mc),
         "Desvio Padrão": np.std(arr_thermo_mc), "IC 95% Inf": np.percentile(arr_thermo_mc,2.5),
         "IC 95% Sup": np.percentile(arr_thermo_mc,97.5)},
        {"Tecnologia": "Leiras", "Média": np.mean(arr_wind_mc), "Mediana": np.median(arr_wind_mc),
         "Desvio Padrão": np.std(arr_wind_mc), "IC 95% Inf": np.percentile(arr_wind_mc,2.5),
         "IC 95% Sup": np.percentile(arr_wind_mc,97.5)}
    ])
    st.dataframe(stats_df.style.format({c: lambda x: formatar_br(x) for c in stats_df.columns if c != "Tecnologia"}))

    # Distribuição das emissões evitadas (gráfico KDE)
    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.kdeplot(arr_thermo_mc, label="Termofílica", linewidth=2, ax=ax3)
    sns.kdeplot(arr_wind_mc, label="Leiras (TOOL13)", linewidth=2, ax=ax3)
    ax3.set_title("Distribuição das Emissões Evitadas (Monte Carlo)")
    ax3.set_xlabel("tCO₂eq")
    ax3.set_ylabel("Densidade")
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.xaxis.set_major_formatter(FuncFormatter(br_format))
    st.pyplot(fig3)
    plt.close(fig3)

    # Tabela anual detalhada (já calculada)
    st.subheader("📋 Resultados Anuais (Cenário Otimista)")
    df_anual_fmt = df_anual[['Year', 'base', 'thermo', 'wind', 'Evitado_Thermo', 'Evitado_Wind']].copy()
    df_anual_fmt.columns = ['Year', 'Baseline (tCO₂eq)', 'Termofílica (tCO₂eq)', 'Leiras (tCO₂eq)', 'Redução Termofílica', 'Redução Leiras']
    for col in df_anual_fmt.columns:
        if col != 'Year':
            df_anual_fmt[col] = df_anual_fmt[col].apply(formatar_br)
    st.dataframe(df_anual_fmt)

    st.session_state.run_simulation = False

else:
    st.info("💡 Ajuste os parâmetros na barra lateral e clique em 'Executar Simulação'.")

st.markdown("---")
st.markdown("""
**📚 Referências:**  
- **AMS‑III.F (v12.0)** – *Avoidance of methane emissions through composting* (UNFCCC, 2016)  
- **TOOL13 (v02.0)** – *Project and leakage emissions from composting* (UNFCCC, 2017)  
- **A6.4‑AMT‑003 (v01.0)** – *Emissions from solid waste disposal sites* (UNFCCC, 2024)  
- **Yang et al. (2017)** – *Waste Management*, 66, 44-51 (DOI: 10.1016/j.wasman.2017.04.033)  
- **GWP-20** – Forster et al. (2021) IPCC AR6  
- **Aterro CGR Guatapará (Ribeirão Preto):** usina de biogás com captura estimada de 60% do metano gerado.
""")
