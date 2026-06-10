# IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS
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
# PARÂMETROS GLOBAIS (constantes)
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

COMPOSTING_DAYS = 50
GWP_CH4_20 = 79.7
GWP_N2O_20 = 273

# =============================================================================
# PERFIS DE EMISSÃO (carregados uma única vez - otimização)
# =============================================================================
profile_ch4 = np.array([
    0.02,0.02,0.02,0.03,0.03,0.04,0.04,0.05,0.05,0.06,
    0.07,0.08,0.09,0.10,0.09,0.08,0.07,0.06,0.05,0.04,
    0.03,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.01,0.01,
    0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,
    0.002,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.001,0.001
])
profile_ch4 /= profile_ch4.sum()

profile_n2o = np.array([
    0.10,0.08,0.15,0.05,0.03,0.04,0.05,0.07,0.10,0.12,
    0.15,0.18,0.20,0.18,0.15,0.12,0.10,0.08,0.06,0.05,
    0.04,0.03,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.01,
    0.005,0.005,0.005,0.005,0.005,0.002,0.002,0.002,0.002,0.002,
    0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001
])
profile_n2o /= profile_n2o.sum()

profile_n2o_landfill = {1:0.10,2:0.30,3:0.40,4:0.15,5:0.05}
profile_n2o_pre = {1:0.8623,2:0.10,3:0.0377}

# Emissões de pré-descarte (constantes)
CH4_pre_ugC_per_kg_h = 2.78
CH4_pre_kg_per_kg_day = CH4_pre_ugC_per_kg_h * (16/12) * 24 / 1_000_000_000
N2O_pre_mgN_per_kg_total = 20.26
N2O_pre_kg_per_kg_total = N2O_pre_mgN_per_kg_total * (44/28) / 1_000_000

# =============================================================================
# CLASSE DE CÁLCULO (otimizada)
# =============================================================================
class GHGEmissionCalculator:
    # Atributos estáticos (compartilhados entre instâncias)
    profile_ch4 = profile_ch4
    profile_n2o = profile_n2o
    profile_n2o_landfill = profile_n2o_landfill
    profile_n2o_pre = profile_n2o_pre
    CH4_pre_kg_per_kg_day = CH4_pre_kg_per_kg_day
    N2O_pre_kg_per_kg_total = N2O_pre_kg_per_kg_total

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
        self.COMPOSTING_DAYS = COMPOSTING_DAYS
        self.GWP_CH4_20 = GWP_CH4_20
        self.GWP_N2O_20 = GWP_N2O_20

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

        # Pré-descarte
        ch4_pre = np.full(days, w_kg_day * self.CH4_pre_kg_per_kg_day)
        n2o_pre = np.zeros(days)
        for e in range(days):
            for dd, frac in self.profile_n2o_pre.items():
                idx = e + dd - 1
                if idx < days:
                    n2o_pre[idx] += w_kg_day * self.N2O_pre_kg_per_kg_total * frac
        return ch4 + ch4_pre, n2o + n2o_pre

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

    # Método completo (com séries) para gráficos determinísticos
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

    # Método rápido (apenas totais) para Monte Carlo e Sobol
    def calculate_avoided_emissions_fast(self, w_kg_day, k, temp, doc, umid, years):
        ch4_l, n2o_l = self.calculate_landfill_emissions(w_kg_day, k, temp, doc, umid, years)
        ch4_t, n2o_t = self.calculate_thermophilic_emissions(w_kg_day, umid, years)
        ch4_w, n2o_w = self.calculate_windrow_emissions(w_kg_day, umid, years)
        base = (ch4_l*self.GWP_CH4_20 + n2o_l*self.GWP_N2O_20)/1000
        thermo = (ch4_t*self.GWP_CH4_20 + n2o_t*self.GWP_N2O_20)/1000
        wind = (ch4_w*self.GWP_CH4_20 + n2o_w*self.GWP_N2O_20)/1000
        return base.sum() - thermo.sum(), base.sum() - wind.sum()


# =============================================================================
# FUNÇÕES DE COTAÇÃO, FORMATAÇÃO E INTERFACE
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
Esta ferramenta compara **duas tecnologias de compostagem**: **Termofílica** (fatores de Yang et al., 2017) e **Convencional em Leiras** (fatores do **TOOL13** – ferramenta da metodologia **AMS‑III.F** da UNFCCC) com o **cenário baseline (aterro sanitário)** calibrado para Ribeirão Preto (aterro CGR Guatapará com captura de biogás).  
**Estatísticas de diferença significativa** entre as emissões evitadas são calculadas via Monte Carlo.

**Metodologias:**  
- **Baseline:** A6.4‑AMT‑003 (UNFCCC, 2024) – modelo FOD do IPCC  
- **Compostagem em Leiras (TOOL13):** parte integrante da metodologia AMS‑III.F (UNFCCC, 2016) – fatores padrão: CH₄ = 0,002 t/t úmido; N₂O = 0,0005 t/t úmido  
- **Compostagem Termofílica (Yang et al., 2017):** apenas para comparação científica (não segue a AMS‑III.F) – CH₄ = 0,0060 t/tC; N₂O = 0,0196 t/tN
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
# FUNÇÕES COM CACHE PARA ANÁLISES PESADAS (otimização crítica)
# =============================================================================
@st.cache_data(show_spinner=False)
def cached_sobol(n_samples, residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao, gwp_ch4, gwp_n2o):
    """Executa análise de sensibilidade Sobol com cache."""
    problem = {'num_vars':3, 'names':['k','T','DOC'], 'bounds':[[0.06,0.40],[20,40],[0.10,0.25]]}
    param_values = sample(problem, n_samples, seed=50)
    calc = GHGEmissionCalculator()
    calc.GWP_CH4_20 = gwp_ch4
    calc.GWP_N2O_20 = gwp_n2o
    
    # Paralelização total
    res_t = Parallel(n_jobs=-1)(
        delayed(calc.calculate_avoided_emissions_fast)(residuos_kg_dia, p[0], p[1], p[2], umidade, anos_simulacao)[0]
        for p in param_values
    )
    res_w = Parallel(n_jobs=-1)(
        delayed(calc.calculate_avoided_emissions_fast)(residuos_kg_dia, p[0], p[1], p[2], umidade, anos_simulacao)[1]
        for p in param_values
    )
    Si_t = analyze(problem, np.array(res_t), print_to_console=False)
    Si_w = analyze(problem, np.array(res_w), print_to_console=False)
    return Si_t, Si_w

@st.cache_data(show_spinner=False)
def cached_montecarlo(n_simulations, residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao, gwp_ch4, gwp_n2o):
    """Executa Monte Carlo com cache."""
    np.random.seed(50)
    u = np.random.uniform(0.75, 0.90, n_simulations)
    t = np.random.normal(25, 3, n_simulations)
    d = np.random.triangular(0.12, 0.15, 0.18, n_simulations)
    
    calc = GHGEmissionCalculator()
    calc.GWP_CH4_20 = gwp_ch4
    calc.GWP_N2O_20 = gwp_n2o
    
    def run_one(i):
        np.random.seed(50 + i)
        thermo, wind = calc.calculate_avoided_emissions_fast(
            residuos_kg_dia, k_ano, t[i], d[i], u[i], anos_simulacao
        )
        return thermo, wind
    
    resultados = Parallel(n_jobs=-1)(delayed(run_one)(i) for i in range(n_simulations))
    arr_thermo = np.array([r[0] for r in resultados])
    arr_wind = np.array([r[1] for r in resultados])
    return arr_thermo, arr_wind

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if st.session_state.get('run_simulation', False):

    # -------------------- 1. RESULTADOS DETERMINÍSTICOS (rápidos) --------------------
    with st.spinner("Calculando resultados determinísticos..."):
        calc = GHGEmissionCalculator()
        calc.GWP_CH4_20, calc.GWP_N2O_20 = (79.7, 273)
        res_det = calc.calculate_avoided_emissions(residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao)
        evitado_thermo = res_det['thermo_avoided']
        evitado_windrow = res_det['wind_avoided']

        base_series = res_det['base_series']
        thermo_series = res_det['thermo_series']
        wind_series = res_det['wind_series']

        dias_total = len(base_series)
        datas = pd.date_range(start=datetime.now(), periods=dias_total, freq='D')
        df_dia = pd.DataFrame({'Data': datas, 'base': base_series, 'thermo': thermo_series, 'wind': wind_series})
        df_dia['Year'] = df_dia['Data'].dt.year
        df_anual = df_dia.groupby('Year').agg({'base':'sum','thermo':'sum','wind':'sum'}).reset_index()
        df_anual['Evitado_Thermo'] = df_anual['base'] - df_anual['thermo']
        df_anual['Evitado_Wind'] = df_anual['base'] - df_anual['wind']

    st.header("📈 Resultados da Simulação (GWP-20)")
    st.info(f"""
    **Parâmetros calibrados para Ribeirão Preto:**  
    - k = {formatar_br(k_ano)} ano⁻¹, T = {formatar_br(T)} °C, DOC = {formatar_br(DOC)}, Umidade = {formatar_br(umidade_valor)}%  
    - Resíduos totais: {formatar_br(residuos_kg_dia * 365 * anos_simulacao / 1000)} t  
    - **Aterro (baseline):** MCF = 1,0; captura de metano = 60%; φ = 0,85 (A6.4‑AMT‑003)  
    - **Compostagem em Leiras (TOOL13):** fatores padrão da metodologia AMS‑III.F – CH₄ = 0,002 t/t úmido; N₂O = 0,0005 t/t úmido  
    - **Compostagem Termofílica (Yang et al., 2017):** apenas para comparação – CH₄ = 0,0060 t/tC; N₂O = 0,0196 t/tN
    """)

    st.subheader("💰 Valor Financeiro (Cenário Otimista)")
    preco = st.session_state.preco_carbono
    moeda = st.session_state.moeda_carbono
    cambio = st.session_state.taxa_cambio
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Termofílica (Yang et al., 2017) - Evitado", f"{formatar_br(evitado_thermo)} tCO₂eq")
        st.metric("Valor (Euro)", f"{moeda} {formatar_br(evitado_thermo * preco)}")
        st.metric("Valor (R$)", f"R$ {formatar_br(evitado_thermo * preco * cambio)}")
    with col2:
        st.metric("Leiras (TOOL13) - Evitado", f"{formatar_br(evitado_windrow)} tCO₂eq")
        st.metric("Valor (Euro)", f"{moeda} {formatar_br(evitado_windrow * preco)}")
        st.metric("Valor (R$)", f"R$ {formatar_br(evitado_windrow * preco * cambio)}")

    st.subheader("📊 Comparação Anual das Emissões Evitadas por Tecnologia")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_anual['Year']))
    width = 0.35
    ax.bar(x - width/2, df_anual['Evitado_Thermo'], width, label='Termofílica (Yang et al., 2017)', edgecolor='black', color='orange')
    ax.bar(x + width/2, df_anual['Evitado_Wind'], width, label='Leiras (TOOL13) / AMS‑III.F', edgecolor='black', color='green', hatch='//')
    for i, (v1, v2) in enumerate(zip(df_anual['Evitado_Thermo'], df_anual['Evitado_Wind'])):
        ax.text(i - width/2, v1 + max(v1,v2)*0.01, formatar_br(v1), ha='center', fontsize=9, fontweight='bold')
        ax.text(i + width/2, v2 + max(v1,v2)*0.01, formatar_br(v2), ha='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Ano', fontsize=12)
    ax.set_ylabel('Emissões Evitadas (t CO₂eq)', fontsize=12)
    ax.set_title('Comparação Anual: Emissões Evitadas pela Termofílica (Yang et al., 2017) vs Leiras (TOOL13 / AMS‑III.F)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(df_anual['Year'], fontsize=8)
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(br_format))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📉 Emissões Acumuladas ao Longo do Tempo: Baseline vs Tecnologias")
    base_acum = np.cumsum(base_series)
    thermo_acum = np.cumsum(thermo_series)
    wind_acum = np.cumsum(wind_series)
    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.plot(datas, base_acum, 'r-', label='Baseline (Aterro Sanitário)', linewidth=2)
    ax2.plot(datas, thermo_acum, 'orange', label='Termofílica (Yang et al., 2017)', linewidth=2)
    ax2.plot(datas, wind_acum, 'green', label='Leiras (TOOL13 / AMS‑III.F)', linewidth=2)
    ax2.fill_between(datas, thermo_acum, wind_acum, color='gray', alpha=0.3, label='Diferença entre as duas tecnologias de compostagem')
    ax2.set_title(f'Emissões Acumuladas (tCO₂eq) – {anos_simulacao} anos de simulação (k = {formatar_br(k_ano)} ano⁻¹)', fontsize=13)
    ax2.set_xlabel('Data', fontsize=12)
    ax2.set_ylabel('tCO₂eq Acumulado', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.yaxis.set_major_formatter(FuncFormatter(br_format))
    st.pyplot(fig2)
    plt.close(fig2)

    # -------------------- 2. TABELA COMPARATIVA DOS TRÊS GWPs --------------------
    st.subheader("📊 Comparação entre Cenários de GWP – Emissões Evitadas (tCO₂eq)")
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
        thermo_avoided, wind_avoided = calc_temp.calculate_avoided_emissions_fast(
            residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao
        )
        comparacao.append({
            "Cenário": nome,
            "Termofílica (Yang et al., 2017)": thermo_avoided,
            "Leiras (TOOL13) / AMS‑III.F": wind_avoided
        })
    df_gwp = pd.DataFrame(comparacao)
    st.dataframe(df_gwp.style.format({c: lambda x: formatar_br(x) for c in df_gwp.columns if c != "Cenário"}))

    # -------------------- 3. ANÁLISE SOBOL (com cache e paralelismo total) --------------------
    st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - GWP-20")
    with st.spinner("Executando análise Sobol (paralelizada com cache)..."):
        g20_ch4, g20_n2o = gwps["Otimista (GWP-20)"]
        Si_t, Si_w = cached_sobol(
            n_samples, residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao, g20_ch4, g20_n2o
        )
    df_sens = pd.DataFrame({
        'Parâmetro': ['k','T','DOC'],
        'S1_Termofílica (Yang et al., 2017)': Si_t['S1'], 'ST_Termofílica (Yang et al., 2017)': Si_t['ST'],
        'S1_Leiras (TOOL13 / AMS‑III.F)': Si_w['S1'], 'ST_Leiras (TOOL13 / AMS‑III.F)': Si_w['ST']
    })
    st.dataframe(df_sens.style.format({c:'{:.4f}' for c in df_sens.columns if c != 'Parâmetro'}))

    # -------------------- 4. MONTE CARLO (com cache e paralelismo total) --------------------
    st.subheader("🎲 Análise de Incerteza (Monte Carlo) e Comparação Estatística")
    with st.spinner("Executando simulações Monte Carlo (paralelizado com cache)..."):
        arr_thermo_mc, arr_wind_mc = cached_montecarlo(
            n_simulations, residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao, g20_ch4, g20_n2o
        )
        diff = arr_thermo_mc - arr_wind_mc

        shapiro_stat, shapiro_p = stats.shapiro(diff)
        t_stat, t_p = stats.ttest_rel(arr_thermo_mc, arr_wind_mc)
        w_stat, w_p = stats.wilcoxon(arr_thermo_mc, arr_wind_mc)

    st.write(f"**Teste de normalidade (Shapiro-Wilk) da diferença:** estatística = {shapiro_stat:.5f}, p = {shapiro_p:.5f}")
    st.write(f"**Teste t pareado:** t = {t_stat:.5f}, p = {t_p:.5f}")
    st.write(f"**Teste de Wilcoxon:** estatística = {w_stat:.5f}, p = {w_p:.5f}")

    stats_df = pd.DataFrame([
        {"Tecnologia": "Termofílica (Yang et al., 2017)", "Média": np.mean(arr_thermo_mc), "Mediana": np.median(arr_thermo_mc),
         "Desvio Padrão": np.std(arr_thermo_mc), "IC 95% Inf": np.percentile(arr_thermo_mc,2.5),
         "IC 95% Sup": np.percentile(arr_thermo_mc,97.5)},
        {"Tecnologia": "Leiras (TOOL13 / AMS‑III.F)", "Média": np.mean(arr_wind_mc), "Mediana": np.median(arr_wind_mc),
         "Desvio Padrão": np.std(arr_wind_mc), "IC 95% Inf": np.percentile(arr_wind_mc,2.5),
         "IC 95% Sup": np.percentile(arr_wind_mc,97.5)}
    ])
    st.dataframe(stats_df.style.format({c: lambda x: formatar_br(x) for c in stats_df.columns if c != "Tecnologia"}))

    # Distribuição das emissões evitadas
    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.kdeplot(arr_thermo_mc, label="Termofílica (Yang et al., 2017)", linewidth=2, ax=ax3)
    sns.kdeplot(arr_wind_mc, label="Leiras (TOOL13 / AMS‑III.F)", linewidth=2, ax=ax3)
    ax3.set_title("Distribuição de Probabilidade das Emissões Evitadas – Monte Carlo", fontsize=13)
    ax3.set_xlabel("Emissões Evitadas (tCO₂eq)", fontsize=12)
    ax3.set_ylabel("Densidade de Probabilidade", fontsize=12)
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.xaxis.set_major_formatter(FuncFormatter(br_format))
    st.pyplot(fig3)
    plt.close(fig3)

    # Tabela anual detalhada
    st.subheader("📋 Resultados Anuais (Cenário Otimista GWP-20)")
    df_anual_fmt = df_anual[['Year', 'base', 'thermo', 'wind', 'Evitado_Thermo', 'Evitado_Wind']].copy()
    df_anual_fmt.columns = ['Year', 'Baseline (tCO₂eq)', 'Termofílica (Yang et al., 2017) (tCO₂eq)', 'Leiras (TOOL13 / AMS‑III.F) (tCO₂eq)', 'Emissões Evitadas Termofílica (Yang et al., 2017)', 'Emissões Evitadas Leiras (TOOL13 / AMS‑III.F)']
    for col in df_anual_fmt.columns:
        if col != 'Year':
            df_anual_fmt[col] = df_anual_fmt[col].apply(formatar_br)
    st.dataframe(df_anual_fmt)

    st.session_state.run_simulation = False

else:
    st.info("💡 Ajuste os parâmetros na barra lateral e clique em 'Executar Simulação'.")

st.markdown("---")
st.markdown("""
**📚 Referências metodológicas:**  
- **AMS‑III.F (v12.0)** – *Avoidance of methane emissions through composting* (UNFCCC, 2016) – metodologia guarda‑chuva para projetos de compostagem.  
- **TOOL13 (v02.0)** – *Project and leakage emissions from composting* (UNFCCC, 2017) – ferramenta obrigatória para cálculo das emissões do projeto no âmbito da **AMS‑III.F**.  
- **A6.4‑AMT‑003 (v01.0)** – *Emissions from solid waste disposal sites* (UNFCCC, 2024) – usada para o baseline (aterro).  

**Tecnologias comparadas:**  
- **Compostagem em Leiras (TOOL13 / AMS‑III.F):** fatores padrão da ferramenta (CH₄ = 0,002 t/t úmido; N₂O = 0,0005 t/t úmido). **Tecnologia alinhada à metodologia da ONU.**  
- **Compostagem Termofílica (Yang et al., 2017):** fatores experimentais (CH₄ = 0,0060 t/tC; N₂O = 0,0196 t/tN) – **não segue a AMS‑III.F**, apenas para comparação científica.  

**Outras referências:**  
- GWP-20: Forster et al. (2021) IPCC AR6  
- Aterro CGR Guatapará (Ribeirão Preto): usina de biogás com captura estimada de 60% do metano gerado.
""")
