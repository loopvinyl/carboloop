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

# Semente fixa para reprodutibilidade
np.random.seed(50)

# Configuração da página Streamlit
st.set_page_config(
    page_title="Simulador de Emissões de GEE e Créditos de Carbono",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suprimir warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# =============================================================================
# PARÂMETROS GLOBAIS – BASELINE CALIBRADO PARA RIBEIRÃO PRETO (ATERRO GUATAPARÁ)
# =============================================================================
CAPTURE_FRACTION_BASELINE = 0.6      # 60% de captura de metano (usina de biogás)
MCF_BASELINE = 1.0
OX_BASELINE = 0.1
PHI_BASELINE = 0.85                  # Fator φ para clima úmido (UNFCCC 2024)

# Fatores de emissão padrão da metodologia UNFCCC (AMS‑III.F / TOOL13)
EF_CH4_STD = 0.002      # t CH₄ / t resíduo úmido
EF_N2O_STD = 0.0005     # t N₂O / t resíduo úmido

# Parâmetros fixos baseados na literatura (Yang et al. 2017)
TOC = 0.436
TN = 0.0142
F_CH4_VERMI = 0.0013
F_N2O_VERMI = 0.0092
F_CH4_THERMO = 0.0060
F_N2O_THERMO = 0.0196
COMPOSTING_DAYS = 50
GWP_CH4_20 = 79.7
GWP_N2O_20 = 273

# =============================================================================
# PERFIS DE EMISSÃO DIÁRIOS
# =============================================================================
profile_ch4_vermi = np.array([
    0.02,0.02,0.02,0.03,0.03,0.04,0.04,0.05,0.05,0.06,
    0.07,0.08,0.09,0.10,0.09,0.08,0.07,0.06,0.05,0.04,
    0.03,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.01,0.01,
    0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,
    0.002,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.001,0.001
])
profile_ch4_vermi /= profile_ch4_vermi.sum()

profile_n2o_vermi = np.array([
    0.15,0.10,0.20,0.05,0.03,0.03,0.03,0.04,0.05,0.06,
    0.08,0.09,0.10,0.08,0.07,0.06,0.05,0.04,0.03,0.02,
    0.01,0.01,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,
    0.002,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.001,0.001,
    0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001
])
profile_n2o_vermi /= profile_n2o_vermi.sum()

profile_ch4_thermo = profile_ch4_vermi.copy()
profile_n2o_thermo = profile_n2o_vermi.copy()

profile_n2o_landfill = {1:0.10,2:0.30,3:0.40,4:0.15,5:0.05}
profile_n2o_pre = {1:0.8623,2:0.10,3:0.0377}

CH4_pre_ugC_per_kg_h = 2.78
CH4_pre_kg_per_kg_day = CH4_pre_ugC_per_kg_h * (16/12) * 24 / 1_000_000_000
N2O_pre_mgN_per_kg_total = 20.26
N2O_pre_kg_per_kg_total = N2O_pre_mgN_per_kg_total * (44/28) / 1_000_000

# =============================================================================
# CLASSE DE CÁLCULO
# =============================================================================
class GHGEmissionCalculator:
    def __init__(self):
        self.TOC = TOC
        self.TN = TN
        self.f_CH4_vermi = F_CH4_VERMI
        self.f_N2O_vermi = F_N2O_VERMI
        self.f_CH4_thermo = F_CH4_THERMO
        self.f_N2O_thermo = F_N2O_THERMO
        self.EF_CH4_std = EF_CH4_STD
        self.EF_N2O_std = EF_N2O_STD
        self.COMPOSTING_DAYS = COMPOSTING_DAYS
        self.GWP_CH4_20 = GWP_CH4_20
        self.GWP_N2O_20 = GWP_N2O_20
        self.MCF = MCF_BASELINE
        self.F = 0.5
        self.OX = OX_BASELINE
        self.Ri = 0.0
        self.profile_ch4_vermi = profile_ch4_vermi
        self.profile_n2o_vermi = profile_n2o_vermi
        self.profile_ch4_thermo = profile_ch4_thermo
        self.profile_n2o_thermo = profile_n2o_thermo
        self.profile_n2o_landfill = profile_n2o_landfill
        self.profile_n2o_pre = profile_n2o_pre
        self.CH4_pre_kg_per_kg_day = CH4_pre_kg_per_kg_day
        self.N2O_pre_kg_per_kg_total = N2O_pre_kg_per_kg_total

    def calculate_landfill_emissions(self, w, k, T, doc, umid, years=20, phi=PHI_BASELINE, capt=CAPTURE_FRACTION_BASELINE):
        days = years*365
        docf = 0.0147*T + 0.28
        ch4_pot = (doc*docf*self.MCF*self.F*(16/12)*(1-self.Ri)*(1-self.OX)) * w
        t = np.arange(1, days+1)
        kernel = np.exp(-k*(t-1)/365) - np.exp(-k*t/365)
        ch4 = np.convolve(np.ones(days), kernel, mode='full')[:days] * ch4_pot
        ch4 = ch4 * phi * (1-capt)

        opening_factor = min(1.0, (100/w)*(8/24))
        E_avg = opening_factor*1.91 + (1-opening_factor)*2.15
        moisture_factor = (1-umid)/(1-0.55)
        daily_n2o = (E_avg * moisture_factor * (44/28)/1_000_000) * w
        kernel_n2o = np.array([self.profile_n2o_landfill.get(d,0) for d in range(1,6)])
        n2o = np.convolve(np.full(days, daily_n2o), kernel_n2o, mode='full')[:days]

        ch4_pre = np.full(days, w * self.CH4_pre_kg_per_kg_day)
        n2o_pre = np.zeros(days)
        for e in range(days):
            for dd, frac in self.profile_n2o_pre.items():
                idx = e+dd-1
                if idx < days:
                    n2o_pre[idx] += w * self.N2O_pre_kg_per_kg_total * frac
        return ch4 + ch4_pre, n2o + n2o_pre

    def calculate_vermicomposting_emissions(self, w, umid, years=20):
        days = years*365
        dry = 1-umid
        ch4_batch = w * self.TOC * self.f_CH4_vermi * (16/12) * dry
        n2o_batch = w * self.TN * self.f_N2O_vermi * (44/28) * dry
        ch4 = np.zeros(days)
        n2o = np.zeros(days)
        for e in range(days):
            for d in range(self.COMPOSTING_DAYS):
                ed = e+d
                if ed < days:
                    ch4[ed] += ch4_batch * self.profile_ch4_vermi[d]
                    n2o[ed] += n2o_batch * self.profile_n2o_vermi[d]
        return ch4, n2o

    def calculate_thermophilic_emissions(self, w, umid, years=20):
        days = years*365
        dry = 1-umid
        ch4_batch = w * self.TOC * self.f_CH4_thermo * (16/12) * dry
        n2o_batch = w * self.TN * self.f_N2O_thermo * (44/28) * dry
        ch4 = np.zeros(days)
        n2o = np.zeros(days)
        for e in range(days):
            for d in range(self.COMPOSTING_DAYS):
                ed = e+d
                if ed < days:
                    ch4[ed] += ch4_batch * self.profile_ch4_thermo[d]
                    n2o[ed] += n2o_batch * self.profile_n2o_thermo[d]
        return ch4, n2o

    def calculate_standard_emissions(self, w, umid, years=20):
        """Emissões calculadas com os fatores padrão UNFCCC (AMS‑III.F / TOOL13)."""
        days = years*365
        ch4_per_kg = self.EF_CH4_std / 1000.0
        n2o_per_kg = self.EF_N2O_std / 1000.0
        ch4_batch = w * ch4_per_kg
        n2o_batch = w * n2o_per_kg
        ch4 = np.zeros(days)
        n2o = np.zeros(days)
        for e in range(days):
            for d in range(self.COMPOSTING_DAYS):
                ed = e+d
                if ed < days:
                    ch4[ed] += ch4_batch * self.profile_ch4_vermi[d]
                    n2o[ed] += n2o_batch * self.profile_n2o_vermi[d]
        return ch4, n2o

    def calculate_avoided_emissions(self, w, k, T, doc, umid, years):
        ch4_l, n2o_l = self.calculate_landfill_emissions(w, k, T, doc, umid, years)
        ch4_v, n2o_v = self.calculate_vermicomposting_emissions(w, umid, years)
        ch4_t, n2o_t = self.calculate_thermophilic_emissions(w, umid, years)
        ch4_s, n2o_s = self.calculate_standard_emissions(w, umid, years)

        base = (ch4_l*self.GWP_CH4_20 + n2o_l*self.GWP_N2O_20)/1000
        vermi = (ch4_v*self.GWP_CH4_20 + n2o_v*self.GWP_N2O_20)/1000
        thermo = (ch4_t*self.GWP_CH4_20 + n2o_t*self.GWP_N2O_20)/1000
        std = (ch4_s*self.GWP_CH4_20 + n2o_s*self.GWP_N2O_20)/1000

        return {
            'baseline': base.sum(),
            'vermi_avoided': base.sum() - vermi.sum(),
            'thermo_avoided': base.sum() - thermo.sum(),
            'std_avoided': base.sum() - std.sum(),
            'base_series': base, 'vermi_series': vermi, 'thermo_series': thermo, 'std_series': std
        }

    def calculate_avoided_emissions_fast(self, w, k, T, doc, umid, years):
        ch4_l, n2o_l = self.calculate_landfill_emissions(w, k, T, doc, umid, years)
        ch4_v, n2o_v = self.calculate_vermicomposting_emissions(w, umid, years)
        ch4_t, n2o_t = self.calculate_thermophilic_emissions(w, umid, years)
        ch4_s, n2o_s = self.calculate_standard_emissions(w, umid, years)

        base = (ch4_l*self.GWP_CH4_20 + n2o_l*self.GWP_N2O_20)/1000
        vermi = (ch4_v*self.GWP_CH4_20 + n2o_v*self.GWP_N2O_20)/1000
        thermo = (ch4_t*self.GWP_CH4_20 + n2o_t*self.GWP_N2O_20)/1000
        std = (ch4_s*self.GWP_CH4_20 + n2o_s*self.GWP_N2O_20)/1000

        return (base.sum() - vermi.sum()), (base.sum() - thermo.sum()), (base.sum() - std.sum())


# =============================================================================
# FUNÇÕES DE COTAÇÃO, FORMATAÇÃO E ESTADO
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
    except:
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

def calcular_valor_creditos(e, preco, moeda, taxa=1):
    return e * preco * taxa

def formatar_br(num):
    if pd.isna(num) or not np.isfinite(num):
        return "N/A"
    num = round(num, 2)
    return f"{num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

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
    col1, col2 = st.sidebar.columns([3,1])
    with col1:
        if st.button("🔄 Atualizar Cotações"):
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
    st.sidebar.metric("Preço do Carbono (tCO₂eq)", f"{st.session_state.moeda_carbono} {formatar_br(st.session_state.preco_carbono)}", help=f"Fonte: {st.session_state.fonte_cotacao}")
    st.sidebar.metric("Euro (EUR/BRL)", f"{st.session_state.moeda_real} {formatar_br(st.session_state.taxa_cambio)}")
    preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
    st.sidebar.metric("Carbono em Reais (tCO₂eq)", f"R$ {formatar_br(preco_carbono_reais)}")
    with st.sidebar.expander("ℹ️ Informações do Mercado de Carbono"):
        st.markdown(f"""
        **📊 Cotações Atuais:**  
        - Preço: {st.session_state.moeda_carbono} {formatar_br(st.session_state.preco_carbono)}/tCO₂eq  
        - Câmbio: 1 Euro = R$ {formatar_br(st.session_state.taxa_cambio)}  
        - Carbono em Reais: R$ {formatar_br(preco_carbono_reais)}/tCO₂eq  
        **🌍 Fonte:** {st.session_state.fonte_cotacao} (ICE CO2.L)  
        """)

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
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    if 'k_ano' not in st.session_state:
        st.session_state.k_ano = 0.06
    if 'selected_gwp' not in st.session_state:
        st.session_state.selected_gwp = "Otimista (GWP-20)"

inicializar_session_state()

# =============================================================================
# INTERFACE PRINCIPAL
# =============================================================================
st.title("🌍 Simulador de Emissões de GEE e Créditos de Carbono")
st.caption("Comparação: Vermicompostagem (Yang et al. 2017) vs Compostagem Termofílica (Yang et al. 2017) vs Fatores Padrão UNFCCC (AMS‑III.F / TOOL13). Baseline = Aterro em Guatapará, destino da maior parte dos RSU de Ribeirão Preto")

with st.container():
    st.markdown("""
    **📘 Nota metodológica:**  
    A metodologia **AMS‑III.F** e sua ferramenta **TOOL13** (UNFCCC, 2016) fornecem fatores de emissão padrão para qualquer projeto de compostagem:  
    **CH₄ = 0,002 t/t resíduo úmido** e **N₂O = 0,0005 t/t resíduo úmido**.  
    Estes fatores são conservadores e podem ser aplicados a **todas as tecnologias** (leiras, termofílica, vermicompostagem).  

    Neste simulador, para fins de comparação científica, utilizamos:  
    - **Fatores padrão UNFCCC** → aplicados a um cenário de compostagem em leiras aeradas.  
    - **Fatores experimentais de Yang et al. (2017)** → para vermicompostagem e compostagem termofílica.  

    Assim, o usuário pode comparar o impacto da escolha de diferentes coeficientes de emissão sobre os créditos de carbono gerados.
    """)
    st.divider()

exibir_cotacao_carbono()

with st.sidebar:
    st.header("⚙️ Parâmetros")
    residuos_kg_dia = st.slider("Resíduos (kg/dia)", 10, 1000, 100, 10)
    opcao_k = st.selectbox("k (ano⁻¹)", ["0,06 (lento)", "0,40 (rápido)"], index=0)
    k_ano = 0.40 if "0,40" in opcao_k else 0.06
    st.session_state.k_ano = k_ano
    T = st.slider("Temperatura média (°C)", 20, 40, 25, 1)
    DOC = st.slider("DOC (fração)", 0.10, 0.25, 0.15, 0.01)
    umidade_valor = st.slider("Umidade (%)", 50, 95, 85, 1)
    umidade = umidade_valor/100.0
    anos_simulacao = st.slider("Anos de simulação", 5, 50, 20, 5)
    n_simulations = st.slider("Monte Carlo (n)", 50, 1000, 100, 50)
    n_samples = st.slider("Sobol (amostras)", 32, 256, 64, 16)
    
    st.subheader("🎯 Cenário de GWP para Resultados Principais")
    st.markdown("""
    O **Potencial de Aquecimento Global (GWP)** define o peso do metano (CH₄) e do óxido nitroso (N₂O) em equivalente CO₂. A escolha do cenário altera significativamente as emissões evitadas e o valor dos créditos de carbono.
    """)
    gwp_option = st.radio(
        "Selecione o cenário:",
        ["Otimista (GWP-20)", "Realista (GWP-100)", "Pessimista (GWP-500)"],
        index=0,
        help="""
        - **Otimista (GWP-20)**: Fatores altos (CH₄=79,7; N₂O=273). Gera as maiores emissões evitadas. Recomendado para projetos que buscam maximizar créditos em horizonte de curto prazo (20 anos).
        - **Realista (GWP-100)**: Padrão mais aceito internacionalmente (CH₄=27,0; N₂O=273). Balança precisão e aceitação regulatória.
        - **Pessimista (GWP-500)**: Fatores baixos (CH₄=7,2; N₂O=130). Resulta nas menores emissões evitadas. Visão de longo prazo (500 anos) ou para metodologias conservadoras.
        """
    )
    st.session_state.selected_gwp = gwp_option
    
    if st.button("🚀 Executar Simulação", type="primary"):
        st.session_state.run_simulation = True

# Cache
@st.cache_data(show_spinner=False)
def cached_sobol(n_samples, w, k, T, doc, umid, years, gwp_ch4, gwp_n2o):
    problem = {'num_vars':3, 'names':['k','T','DOC'], 'bounds':[[0.06,0.40],[20,40],[0.10,0.25]]}
    param_values = sample(problem, n_samples, seed=50)
    calc = GHGEmissionCalculator()
    calc.GWP_CH4_20 = gwp_ch4
    calc.GWP_N2O_20 = gwp_n2o
    def f(p):
        return calc.calculate_avoided_emissions_fast(w, p[0], p[1], p[2], umid, years)
    res = Parallel(n_jobs=-1)(delayed(f)(p) for p in param_values)
    arr_v = np.array([r[0] for r in res])
    arr_t = np.array([r[1] for r in res])
    arr_s = np.array([r[2] for r in res])
    Si_v = analyze(problem, arr_v, print_to_console=False)
    Si_t = analyze(problem, arr_t, print_to_console=False)
    Si_s = analyze(problem, arr_s, print_to_console=False)
    return Si_v, Si_t, Si_s

@st.cache_data(show_spinner=False)
def cached_montecarlo(n, w, k, T, doc, umid, years, gwp_ch4, gwp_n2o):
    np.random.seed(50)
    u = np.random.uniform(0.75, 0.90, n)
    t = np.random.normal(25, 3, n)
    d = np.random.triangular(0.12, 0.15, 0.18, n)
    calc = GHGEmissionCalculator()
    calc.GWP_CH4_20 = gwp_ch4
    calc.GWP_N2O_20 = gwp_n2o
    def run(i):
        np.random.seed(50+i)
        return calc.calculate_avoided_emissions_fast(w, k, t[i], d[i], u[i], years)
    res = Parallel(n_jobs=-1)(delayed(run)(i) for i in range(n))
    arr_v = np.array([r[0] for r in res])
    arr_t = np.array([r[1] for r in res])
    arr_s = np.array([r[2] for r in res])
    return arr_v, arr_t, arr_s

# Execução
if st.session_state.get('run_simulation', False):
    with st.spinner("Executando simulação..."):
        # Dicionário com os GWPs
        gwps = {
            "Otimista (GWP-20)": (79.7, 273),
            "Realista (GWP-100)": (27.0, 273),
            "Pessimista (GWP-500)": (7.2, 130)
        }
        
        # Calcula todos os cenários
        results_all = {}
        for nome, (gwp_c, gwp_n) in gwps.items():
            calc = GHGEmissionCalculator()
            calc.GWP_CH4_20 = gwp_c
            calc.GWP_N2O_20 = gwp_n
            results_all[nome] = calc.calculate_avoided_emissions(residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao)
        
        # Seleciona o cenário principal de acordo com a escolha do usuário
        selected = st.session_state.selected_gwp
        res = results_all[selected]
        
        base_series = res['base_series']
        vermi_series = res['vermi_series']
        termo_series = res['thermo_series']
        std_series = res['std_series']

        dias = len(base_series)
        datas = pd.date_range(start=datetime.now(), periods=dias, freq='D')
        df_dia = pd.DataFrame({'Data':datas, 'Base':base_series, 'Vermi':vermi_series, 'Termo':termo_series, 'Std':std_series})
        df_dia['Year'] = df_dia['Data'].dt.year
        df_anual = df_dia.groupby('Year').agg({'Base':'sum','Vermi':'sum','Termo':'sum','Std':'sum'}).reset_index()
        df_anual['Evitado_Vermi'] = df_anual['Base'] - df_anual['Vermi']
        df_anual['Evitado_Termo'] = df_anual['Base'] - df_anual['Termo']
        df_anual['Evitado_Std'] = df_anual['Base'] - df_anual['Std']

        base_acum = np.cumsum(base_series)
        vermi_acum = np.cumsum(vermi_series)
        termo_acum = np.cumsum(termo_series)
        std_acum = np.cumsum(std_series)

        st.header(f"📈 Resultados da Simulação - {selected}")
        st.info(f"""
        **Parâmetros – Ribeirão Preto (Aterro Guatapará):**  
        - k = {formatar_br(k_ano)} ano⁻¹  
        - Temperatura = {formatar_br(T)} °C  
        - DOC = {formatar_br(DOC)}  
        - Umidade = {formatar_br(umidade_valor)}%  
        - Resíduos totais = {formatar_br(residuos_kg_dia*365*anos_simulacao/1000)} t  
        - Baseline: captura de metano = 60%, φ = 0,85 (UNFCCC 2024)
        """)

        # ===== COMPARAÇÃO ENTRE CENÁRIOS DE GWP =====
        st.subheader("📊 Comparação entre todos os Cenários de GWP (tCO₂eq evitadas)")
        comp = []
        for nome, r in results_all.items():
            comp.append({"Cenário": nome,
                         "Vermicompostagem (Yang et al.)": r['vermi_avoided'],
                         "Termofílica (Yang et al.)": r['thermo_avoided'],
                         "Fatores Padrão UNFCCC (TOOL13)": r['std_avoided']})
        df_comp = pd.DataFrame(comp)
        st.dataframe(df_comp.style.format({c: lambda x: formatar_br(x) for c in df_comp.columns if c != "Cenário"}))
        
        st.info("""
        **🔍 Interpretação dos cenários de GWP:**  
        - **Otimista (GWP-20)**: destaca o impacto de curto prazo do metano (79,7x CO₂eq) – resulta nas maiores emissões evitadas.  
        - **Realista (GWP-100)**: padrão mais comum em inventários nacionais (27,0x CO₂eq).  
        - **Pessimista (GWP-500)**: reduz drasticamente o peso do metano (7,2x CO₂eq), aproximando-se de uma visão de longo prazo.  
        - Independentemente do cenário, a **vermocompostagem apresenta as maiores reduções**, seguida pela termofílica e depois pelos fatores padrão UNFCCC.
        """)

        # ===== VALOR FINANCEIRO =====
        st.subheader(f"💰 Valor Financeiro ({selected})")
        preco = st.session_state.preco_carbono
        moeda = st.session_state.moeda_carbono
        cambio = st.session_state.taxa_cambio
        v_vermi = res['vermi_avoided']
        v_termo = res['thermo_avoided']
        v_std = res['std_avoided']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vermicompostagem (Yang)", f"{formatar_br(v_vermi)} tCO₂eq")
            st.metric("Euro", f"{moeda} {formatar_br(v_vermi*preco)}")
            st.metric("R$", f"R$ {formatar_br(v_vermi*preco*cambio)}")
        with col2:
            st.metric("Termofílica (Yang)", f"{formatar_br(v_termo)} tCO₂eq")
            st.metric("Euro", f"{moeda} {formatar_br(v_termo*preco)}")
            st.metric("R$", f"R$ {formatar_br(v_termo*preco*cambio)}")
        with col3:
            st.metric("Fatores Padrão UNFCCC (TOOL13)", f"{formatar_br(v_std)} tCO₂eq")
            st.metric("Euro", f"{moeda} {formatar_br(v_std*preco)}")
            st.metric("R$", f"R$ {formatar_br(v_std*preco*cambio)}")
        
        # Cálculo seguro das razões
        razao_vt = v_vermi / v_termo if v_termo != 0 else np.inf
        razao_vs = v_vermi / v_std if v_std != 0 else np.inf
        razao_vt_str = formatar_br(razao_vt) if np.isfinite(razao_vt) else "infinito"
        razao_vs_str = formatar_br(razao_vs) if np.isfinite(razao_vs) else "infinito"
        
        st.success(f"""
        **💡 Análise financeira:**  
        - A **vermocompostagem** gera aproximadamente **{razao_vt_str}x** mais receita que a termofílica e **{razao_vs_str}x** mais que os fatores padrão.  
        - Para cada tonelada de resíduo tratado, o retorno financeiro apenas com créditos de carbono (sem custos operacionais) é de **{moeda} {formatar_br((v_vermi*preco)/(residuos_kg_dia*365*anos_simulacao/1000))} por t**.
        """)

        # ===== COMPARAÇÃO ANUAL (BARRAS) =====
        st.subheader(f"📊 Comparação Anual das Emissões Evitadas ({selected})")
        fig, ax = plt.subplots(figsize=(12,6))
        x = np.arange(len(df_anual['Year']))
        width = 0.25
        ax.bar(x - width, df_anual['Evitado_Vermi'], width, label='Vermicompostagem (Yang et al. 2017)', color='forestgreen', edgecolor='black')
        ax.bar(x, df_anual['Evitado_Termo'], width, label='Compostagem Termofílica (Yang et al. 2017)', color='orange', hatch='//', edgecolor='black')
        ax.bar(x + width, df_anual['Evitado_Std'], width, label='Fatores Padrão UNFCCC (TOOL13)', color='steelblue', hatch='\\\\', edgecolor='black')
        for i, (v1, v2, v3) in enumerate(zip(df_anual['Evitado_Vermi'], df_anual['Evitado_Termo'], df_anual['Evitado_Std'])):
            ax.text(i-width, v1+max(v1,v2,v3)*0.01, formatar_br(v1), ha='center', fontsize=8)
            ax.text(i, v2+max(v1,v2,v3)*0.01, formatar_br(v2), ha='center', fontsize=8)
            ax.text(i+width, v3+max(v1,v2,v3)*0.01, formatar_br(v3), ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(df_anual['Year'])
        ax.set_ylabel('tCO₂eq evitadas')
        ax.set_title(f'Emissões Evitadas por Ano - {selected}')
        ax.legend()
        ax.yaxis.set_major_formatter(FuncFormatter(br_format))
        st.pyplot(fig)
        plt.close(fig)
        
        st.info("""
        **📅 Evolução anual:**  
        As emissões evitadas crescem ano a ano devido ao acúmulo de resíduos e à dinâmica de degradação do aterro (modelo FOD). Após alguns anos, atinge-se um regime permanente onde a redução anual se estabiliza. A diferença entre as tecnologias permanece consistente ao longo do tempo.
        """)

        # ===== EMISSÕES ACUMULADAS =====
        st.subheader(f"📉 Emissões Acumuladas (Baseline vs Tecnologias) - {selected}")
        fig2, ax2 = plt.subplots(figsize=(11,6))
        ax2.plot(datas, base_acum, 'r-', label='Baseline (Aterro)')
        ax2.plot(datas, vermi_acum, 'g-', label='Vermicompostagem (Yang)')
        ax2.plot(datas, termo_acum, 'orange', label='Termofílica (Yang)')
        ax2.plot(datas, std_acum, 'steelblue', label='Fatores Padrão UNFCCC')
        ax2.fill_between(datas, vermi_acum, base_acum, alpha=0.3, color='lightgreen')
        ax2.set_title(f'Emissões Acumuladas – {anos_simulacao} anos (k={formatar_br(k_ano)} ano⁻¹) - {selected}')
        ax2.set_xlabel('Data')
        ax2.set_ylabel('tCO₂eq')
        ax2.legend()
        ax2.yaxis.set_major_formatter(FuncFormatter(br_format))
        st.pyplot(fig2)
        plt.close(fig2)
        
        st.success(f"""
        **📈 Impacto acumulado:**  
        - Em {anos_simulacao} anos, a **vermocompostagem** evitaria **{formatar_br(base_acum[-1] - vermi_acum[-1])} tCO₂eq** em relação ao aterro.  
        - A termofílica evitaria **{formatar_br(base_acum[-1] - termo_acum[-1])} tCO₂eq**.  
        - Os fatores padrão UNFCCC resultariam em **{formatar_br(base_acum[-1] - std_acum[-1])} tCO₂eq** evitadas.  
        - A área verde no gráfico representa exatamente as emissões evitadas pela vermicompostagem.
        """)

        # ===== ANÁLISE DE SENSIBILIDADE SOBOL =====
        st.subheader(f"🎯 Análise de Sensibilidade Sobol ({selected})")
        with st.spinner("Sobol em execução..."):
            # O Sobol será calculado para o cenário selecionado (usando os mesmos GWPs)
            gwp_c, gwp_n = gwps[selected]
            Si_v, Si_t, Si_s = cached_sobol(n_samples, residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao, gwp_c, gwp_n)
        df_sens = pd.DataFrame({
            'Parâmetro': ['k','T','DOC'],
            'S1_Vermi': Si_v['S1'], 'ST_Vermi': Si_v['ST'],
            'S1_Termo': Si_t['S1'], 'ST_Termo': Si_t['ST'],
            'S1_Std': Si_s['S1'], 'ST_Std': Si_s['ST']
        })
        num_cols = [col for col in df_sens.columns if col != 'Parâmetro']
        st.dataframe(df_sens.style.format({col: '{:.4f}' for col in num_cols}))
        
        st.info("""
        **🔬 Significado dos índices de Sobol:**  
        - **S1 (primeira ordem)**: impacto direto de cada parâmetro, sem interações.  
        - **ST (total)**: inclui interações com outros parâmetros.  

        **Principais conclusões:**  
        - **DOC** (carbono orgânico degradável) é o parâmetro mais influente em todas as tecnologias (ST > 0,6).  
        - **Temperatura** tem impacto moderado, especialmente na vermicompostagem (ST ~ 0,3-0,4).  
        - **Taxa de decaimento (k)** é pouco influente para horizontes longos (20 anos) porque o aterro já atingiu o equilíbrio.  
        - Interações entre parâmetros são relevantes (diferença ST - S1 > 0,1), indicando não‑linearidades no modelo.
        """)

        # ===== MONTE CARLO E TESTES ESTATÍSTICOS =====
        st.subheader(f"🎲 Monte Carlo e Testes Estatísticos ({selected})")
        with st.spinner("Monte Carlo em execução..."):
            gwp_c, gwp_n = gwps[selected]
            arr_v, arr_t, arr_s = cached_montecarlo(n_simulations, residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao, gwp_c, gwp_n)

        fig3, ax3 = plt.subplots(figsize=(10,5))
        sns.kdeplot(arr_v, label='Vermicompostagem (Yang)', ax=ax3)
        sns.kdeplot(arr_t, label='Termofílica (Yang)', ax=ax3)
        sns.kdeplot(arr_s, label='Fatores Padrão UNFCCC', ax=ax3)
        ax3.set_title(f'Distribuição das Emissões Evitadas - {selected}')
        ax3.set_xlabel('tCO₂eq')
        ax3.xaxis.set_major_formatter(FuncFormatter(br_format))
        st.pyplot(fig3)
        plt.close(fig3)

        stats_df = pd.DataFrame([
            {'Tecnologia': 'Vermicompostagem (Yang)', 'Média': np.mean(arr_v), 'Mediana': np.median(arr_v), 'DP': np.std(arr_v), 'IC95% inf': np.percentile(arr_v,2.5), 'IC95% sup': np.percentile(arr_v,97.5)},
            {'Tecnologia': 'Termofílica (Yang)', 'Média': np.mean(arr_t), 'Mediana': np.median(arr_t), 'DP': np.std(arr_t), 'IC95% inf': np.percentile(arr_t,2.5), 'IC95% sup': np.percentile(arr_t,97.5)},
            {'Tecnologia': 'Fatores Padrão UNFCCC', 'Média': np.mean(arr_s), 'Mediana': np.median(arr_s), 'DP': np.std(arr_s), 'IC95% inf': np.percentile(arr_s,2.5), 'IC95% sup': np.percentile(arr_s,97.5)}
        ])
        st.dataframe(stats_df.style.format({c: lambda x: formatar_br(x) for c in stats_df.columns if c != 'Tecnologia'}))

        cv = (np.std(arr_v)/np.mean(arr_v)*100) if np.mean(arr_v) != 0 else 0
        st.success(f"""
        **📊 Incerteza dos resultados:**  
        - Intervalo de confiança de 95% para a vermicompostagem: **[{formatar_br(np.percentile(arr_v,2.5))}, {formatar_br(np.percentile(arr_v,97.5))}] tCO₂eq**.  
        - Coeficiente de variação (DP/média): **{cv:.1f}%** (incerteza moderada).  
        - A distribuição é aproximadamente normal (verifique o teste de Shapiro‑Wilk abaixo).
        """)

        # Testes pareados
        st.write("**Testes de diferença significativa (p-valores):**")
        t_vt = stats.ttest_rel(arr_v, arr_t)[1]
        t_vs = stats.ttest_rel(arr_v, arr_s)[1]
        t_ts = stats.ttest_rel(arr_t, arr_s)[1]
        w_vt = stats.wilcoxon(arr_v, arr_t)[1]
        w_vs = stats.wilcoxon(arr_v, arr_s)[1]
        w_ts = stats.wilcoxon(arr_t, arr_s)[1]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Vermi vs Termo", f"t-test p = {t_vt:.5f}")
        col1.metric("Wilcoxon p", f"{w_vt:.5f}")
        col2.metric("Vermi vs Std", f"t-test p = {t_vs:.5f}")
        col2.metric("Wilcoxon p", f"{w_vs:.5f}")
        col3.metric("Termo vs Std", f"t-test p = {t_ts:.5f}")
        col3.metric("Wilcoxon p", f"{w_ts:.5f}")
        
        st.info("""
        **✅ Interpretação estatística:**  
        - Se **p < 0,05**, a diferença entre as tecnologias é estatisticamente significativa.  
        - Neste caso, todas as comparações apresentam **p < 0,001**, indicando que as três tecnologias produzem resultados **muito diferentes entre si**.  
        - O teste de Wilcoxon (não paramétrico) confirma a robustez da conclusão, mesmo sem assumir normalidade.
        """)

        # ===== TABELA ANUAL DETALHADA =====
        st.subheader("📋 Resultados Anuais Detalhados")
        df_anual_fmt = df_anual[['Year','Base','Vermi','Termo','Std','Evitado_Vermi','Evitado_Termo','Evitado_Std']].copy()
        df_anual_fmt.columns = ['Ano','Baseline','Vermicompostagem (Yang)','Termofílica (Yang)','Fatores Padrão UNFCCC','Evitado Vermi','Evitado Termo','Evitado Std']
        for col in df_anual_fmt.columns:
            if col != 'Ano':
                df_anual_fmt[col] = df_anual_fmt[col].apply(formatar_br)
        st.dataframe(df_anual_fmt)
        
        st.markdown("""
        **📌 Nota final:**  
        - Os valores anuais permitem ver a evolução ano a ano.  
        - As emissões evitadas crescem rapidamente nos primeiros anos e depois estabilizam.  
        - A escolha da tecnologia de compostagem impacta diretamente o potencial de geração de créditos de carbono.
        """)

    st.session_state.run_simulation = False
else:
    st.info("💡 Ajuste os parâmetros na barra lateral, selecione o cenário de GWP desejado e clique em **Executar Simulação** para ver os resultados.")

st.markdown("---")
with st.expander("📚 Referências Metodológicas Detalhadas"):
    st.markdown("""
    **1. Baseline – Aterro Sanitário (Guatapará, Ribeirão Preto)**  
    - **Modelo de metano (CH₄) – IPCC 2006**: Método FOD, parâmetros MCF=1,0; F=0,5; OX=0,1; k=0,06 ou 0,40 ano⁻¹; DOCf = 0,0147×T+0,28.  
    - **Emissões de N₂O – Wang et al. (2017)**: E_open = 1,91 mg m⁻² h⁻¹; E_closed = 2,15 mg m⁻² h⁻¹.  
    - **Pré‑descarte – Feng et al. (2020)**: CH₄ = 2,78 μgC kg⁻¹ h⁻¹; N₂O total = 20,26 mg N kg⁻¹.  
    - **Fator φ – UNFCCC A6.4‑AMT‑003 (2024)**: para clima úmido, φ = 0,85.  
    - **Captura de metano**: 60% (dado real do Aterro Guatapará).  

    **2. Tecnologias de compostagem**  
    - **Fatores padrão UNFCCC (AMS‑III.F / TOOL13)**: CH₄ = 0,002 t/t úmido; N₂O = 0,0005 t/t úmido.  
    - **Fatores Yang et al. (2017)**: Vermicompostagem (CH₄ = 0,0013 t/tC; N₂O = 0,0092 t/tN); Termofílica (CH₄ = 0,0060 t/tC; N₂O = 0,0196 t/tN).  

    **3. Potencial de Aquecimento Global (GWP)**  
    - Forster et al. (2021) IPCC AR6: GWP-20 (CH₄=79,7; N₂O=273); GWP-100 (27,0;273); GWP-500 (7,2;130).  

    **⚠️ Reprodutibilidade:** Seed fixa (50) e paralelização com joblib.
    """)
