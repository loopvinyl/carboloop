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

st.set_page_config(
    page_title="Simulador de Créditos de Carbono - Comparação entre Compostagem Termofílica e Leiras",
    layout="wide"
)

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# =============================================================================
# PARÂMETROS ESPECÍFICOS PARA O PROJETO EM RIBEIRÃO PRETO
# =============================================================================
# Aterro CGR Guatapará (destino dos RSU):
#   - Aterro sanitário com usina de biogás desde 2014.
#   - MCF = 1,0 (aterro anaeróbio gerenciado) – A6.4-AMT-003 Tabela 8
#   - Captura de metano = 60% (capture_fraction = 0,6)
#   - Fator φ (model correction) para baseline em clima úmido: 0,85 (Tabela 5, Application B)
CAPTURE_FRACTION_BASELINE = 0.6
MCF_BASELINE = 1.0
OX_BASELINE = 0.1
PHI_BASELINE = 0.85

# Fatores para compostagem termofílica (Yang et al. 2017)
# Yang, F., Li, G., Zuo, X., & Yang, H. (2017). Waste Management, 66, 44-51.
TOC = 0.436
TN = 0.0142
F_CH4_THERMO = 0.0060      # t CH₄ / t C orgânico
F_N2O_THERMO = 0.0196      # t N₂O / t N

# Fatores de emissão padrão para compostagem em leiras (TOOL13, v02.0, seção 6.3)
EF_CH4_WINDROW = 0.002     # t CH₄ / t resíduo úmido
EF_N2O_WINDROW = 0.0005    # t N₂O / t resíduo úmido

# CLASSE PARA CÁLCULO DE EMISSÕES DE GEE

class GHGEmissionCalculator:
    """
    Calcula emissões de CH₄ e N₂O para:
    - Aterro sanitário (baseline) – calibrado para Ribeirão Preto (A6.4-AMT-003).
    - Compostagem termofílica (Yang et al. 2017).
    - Compostagem em leiras (TOOL13, conforme AMS-III.F).
    """

    def __init__(self):
        # Baseline
        self.MCF = MCF_BASELINE
        self.F = 0.5
        self.OX = OX_BASELINE
        self.Ri = 0.0

        # Termofílica
        self.TOC = TOC
        self.TN = TN
        self.f_CH4_thermo = F_CH4_THERMO
        self.f_N2O_thermo = F_N2O_THERMO

        # Leiras (TOOL13)
        self.EF_CH4_windrow = EF_CH4_WINDROW
        self.EF_N2O_windrow = EF_N2O_WINDROW

        self.COMPOSTING_DAYS = 50
        self.GWP_CH4_20 = 79.7
        self.GWP_N2O_20 = 273

        self._load_emission_profiles()
        self._setup_pre_disposal_emissions()

    def _load_emission_profiles(self):
        """Perfis diários para distribuição temporal (mesmo perfil para ambas)."""
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

    def _setup_pre_disposal_emissions(self):
        CH4_pre_ugC_per_kg_h = 2.78
        self.CH4_pre_kg_per_kg_day = CH4_pre_ugC_per_kg_h * (16/12) * 24 / 1_000_000_000
        N2O_pre_mgN_per_kg_total = 20.26
        self.N2O_pre_kg_per_kg_total = N2O_pre_mgN_per_kg_total * (44/28) / 1_000_000
        self.profile_n2o_pre = {1:0.8623,2:0.10,3:0.0377}

    def calculate_landfill_emissions(self, waste_kg_day, k_year, temperature_C,
                                     doc_fraction, moisture_fraction, years=20,
                                     phi=PHI_BASELINE, capture_fraction=CAPTURE_FRACTION_BASELINE):
        days = years * 365
        docf = 0.0147 * temperature_C + 0.28
        ch4_pot_per_kg = (doc_fraction * docf * self.MCF * self.F * (16/12) *
                          (1 - self.Ri) * (1 - self.OX))
        ch4_pot_daily = waste_kg_day * ch4_pot_per_kg
        t = np.arange(1, days+1, dtype=float)
        kernel = np.exp(-k_year*(t-1)/365.0) - np.exp(-k_year*t/365.0)
        ch4 = np.convolve(np.ones(days), kernel, mode='full')[:days]
        ch4 *= ch4_pot_daily
        ch4 = ch4 * phi * (1 - capture_fraction)

        # N2O do aterro (Wang et al. 2017)
        opening_factor = min(1.0, (100/waste_kg_day)*(8/24))
        E_avg = opening_factor*1.91 + (1-opening_factor)*2.15
        moisture_factor = (1-moisture_fraction)/(1-0.55)
        daily_n2o_kg = (E_avg * moisture_factor * (44/28) / 1_000_000) * waste_kg_day
        kernel_n2o = np.array([self.profile_n2o_landfill.get(d,0) for d in range(1,6)])
        n2o = np.convolve(np.full(days, daily_n2o_kg), kernel_n2o, mode='full')[:days]

        ch4_pre, n2o_pre = self._calculate_pre_disposal(waste_kg_day, days)
        return ch4 + ch4_pre, n2o + n2o_pre

    def _calculate_pre_disposal(self, waste_kg_day, days):
        ch4 = np.full(days, waste_kg_day * self.CH4_pre_kg_per_kg_day)
        n2o = np.zeros(days)
        for entry in range(days):
            for d_after, frac in self.profile_n2o_pre.items():
                idx = entry + d_after - 1
                if idx < days:
                    n2o[idx] += waste_kg_day * self.N2O_pre_kg_per_kg_total * frac
        return ch4, n2o

    def calculate_thermophilic_emissions(self, waste_kg_day, moisture_fraction, years=20):
        days = years * 365
        dry_frac = 1 - moisture_fraction
        ch4_per_batch = waste_kg_day * self.TOC * self.f_CH4_thermo * (16/12) * dry_frac
        n2o_per_batch = waste_kg_day * self.TN * self.f_N2O_thermo * (44/28) * dry_frac
        ch4 = np.zeros(days)
        n2o = np.zeros(days)
        for entry in range(days):
            for d in range(self.COMPOSTING_DAYS):
                em_day = entry + d
                if em_day < days:
                    ch4[em_day] += ch4_per_batch * self.profile_ch4[d]
                    n2o[em_day] += n2o_per_batch * self.profile_n2o[d]
        return ch4, n2o

    def calculate_windrow_emissions(self, waste_kg_day, moisture_fraction, years=20):
        days = years * 365
        total_waste_t = (waste_kg_day * days) / 1000.0
        total_ch4_t = total_waste_t * self.EF_CH4_windrow
        total_n2o_t = total_waste_t * self.EF_N2O_windrow
        ch4_per_kg = self.EF_CH4_windrow / 1000.0
        n2o_per_kg = self.EF_N2O_windrow / 1000.0
        ch4_per_batch_kg = waste_kg_day * ch4_per_kg
        n2o_per_batch_kg = waste_kg_day * n2o_per_kg
        ch4 = np.zeros(days)
        n2o = np.zeros(days)
        for entry in range(days):
            for d in range(self.COMPOSTING_DAYS):
                em_day = entry + d
                if em_day < days:
                    ch4[em_day] += ch4_per_batch_kg * self.profile_ch4[d]
                    n2o[em_day] += n2o_per_batch_kg * self.profile_n2o[d]
        return ch4, n2o

    def calculate_avoided_emissions(self, waste_kg_day, k_year, temperature_C,
                                    doc_fraction, moisture_fraction, years=20):
        ch4_land, n2o_land = self.calculate_landfill_emissions(
            waste_kg_day, k_year, temperature_C, doc_fraction, moisture_fraction, years)
        ch4_thermo, n2o_thermo = self.calculate_thermophilic_emissions(waste_kg_day, moisture_fraction, years)
        ch4_wind, n2o_wind = self.calculate_windrow_emissions(waste_kg_day, moisture_fraction, years)

        base_co2 = (ch4_land * self.GWP_CH4_20 + n2o_land * self.GWP_N2O_20) / 1000
        thermo_co2 = (ch4_thermo * self.GWP_CH4_20 + n2o_thermo * self.GWP_N2O_20) / 1000
        wind_co2 = (ch4_wind * self.GWP_CH4_20 + n2o_wind * self.GWP_N2O_20) / 1000

        return {
            'baseline': {'co2eq_t': base_co2.sum()},
            'thermophilic': {'avoided_co2eq_t': base_co2.sum() - thermo_co2.sum()},
            'windrow': {'avoided_co2eq_t': base_co2.sum() - wind_co2.sum()}
        }


# FUNÇÕES DE COTAÇÃO, FORMATAÇÃO E INTERFACE (mantidas iguais às versões anteriores)
# (Aqui vou inserir as mesmas funções que já estavam no script final com ambas as tecnologias)

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

def formatar_br_dec(numero, decimais=2):
    if pd.isna(numero):
        return "N/A"
    numero = round(numero, decimais)
    return f"{numero:,.{decimais}f}".replace(",", "X").replace(".", ",").replace("X", ".")

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
    with st.sidebar.expander("ℹ️ Informações do Mercado de Carbono"):
        st.markdown(f"""
        **📊 Cotações Atuais:**
        - **Fonte do Carbono:** {st.session_state.fonte_cotacao}
        - **Preço Atual:** {st.session_state.moeda_carbono} {formatar_br(st.session_state.preco_carbono)}/tCO₂eq
        - **Câmbio EUR/BRL:** 1 Euro = R$ {formatar_br(st.session_state.taxa_cambio)}
        - **Carbono em Reais:** R$ {formatar_br(preco_carbono_reais)}/tCO₂eq
        **🌍 Mercado de Referência:** EU ETS (ICE CO2.L)
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
Esta ferramenta compara **duas tecnologias de compostagem** (termofílica e em leiras) com o **cenário baseline (aterro sanitário)** calibrado para a realidade de Ribeirão Preto (aterro CGR Guatapará com captura de biogás).  
Também realiza análise estatística da diferença significativa entre as emissões evitadas pelas duas tecnologias.

**Metodologias:**  
- **Baseline (aterro):** A6.4‑AMT‑003 (UNFCCC, 2024) – modelo FOD, MCF=1,0; captura=60%; φ=0,85.  
- **Compostagem termofílica:** Yang et al. (2017) – fatores CH₄=0,0060 t/t C, N₂O=0,0196 t/t N.  
- **Compostagem em leiras:** TOOL13 (UNFCCC, 2017) – fatores CH₄=0,002 t/t úmido, N₂O=0,0005 t/t úmido.  
""")

exibir_cotacao_carbono()

with st.sidebar:
    st.header("⚙️ Parâmetros de Entrada")
    residuos_kg_dia = st.slider("Resíduos (kg/dia)", 10, 1000, 100, 10)
    st.subheader("📊 Parâmetros da Análise Sobol")
    opcao_k = st.selectbox("k (ano⁻¹)", ["0,06 (lento)", "0,40 (rápido)"], index=0)
    k_ano = 0.40 if "0,40" in opcao_k else 0.06
    st.session_state.k_ano = k_ano
    T = st.slider("Temperatura (°C)", 20, 40, 25, 1)
    DOC = st.slider("DOC (fração)", 0.10, 0.25, 0.15, 0.01)
    umidade_valor = st.slider("Umidade (%)", 50, 95, 85, 1)
    umidade = umidade_valor / 100.0

    st.subheader("🎯 Configuração de Simulação")
    anos_simulacao = st.slider("Anos de simulação", 5, 50, 20, 5)
    n_simulations = st.slider("Monte Carlo (n)", 50, 1000, 100, 50)
    n_samples = st.slider("Sobol (amostras)", 32, 256, 64, 16)

    if st.button("🚀 Executar Simulação", type="primary"):
        st.session_state.run_simulation = True


# FUNÇÕES AUXILIARES PARA SIMULAÇÃO

def compute_results_for_gwp(gwp_ch4, gwp_n2o, w, k, T, doc, umid, years):
    calc = GHGEmissionCalculator()
    calc.GWP_CH4_20 = gwp_ch4
    calc.GWP_N2O_20 = gwp_n2o
    return calc.calculate_avoided_emissions(w, k, T, doc, umid, years)

def sobol_thermo(params, gwp_ch4, gwp_n2o):
    k, T, doc = params
    np.random.seed(50)
    calc = GHGEmissionCalculator()
    calc.GWP_CH4_20 = gwp_ch4
    calc.GWP_N2O_20 = gwp_n2o
    res = calc.calculate_avoided_emissions(residuos_kg_dia, k, T, doc, umidade, anos_simulacao)
    return res['thermophilic']['avoided_co2eq_t']

def sobol_windrow(params, gwp_ch4, gwp_n2o):
    k, T, doc = params
    np.random.seed(50)
    calc = GHGEmissionCalculator()
    calc.GWP_CH4_20 = gwp_ch4
    calc.GWP_N2O_20 = gwp_n2o
    res = calc.calculate_avoided_emissions(residuos_kg_dia, k, T, doc, umidade, anos_simulacao)
    return res['windrow']['avoided_co2eq_t']

def gerar_parametros_mc(n):
    np.random.seed(50)
    u = np.random.uniform(0.75, 0.90, n)
    t = np.random.normal(25, 3, n)
    d = np.random.triangular(0.12, 0.15, 0.18, n)
    return u, t, d


# EXECUÇÃO PRINCIPAL

if st.session_state.get('run_simulation', False):
    with st.spinner('Simulando...'):
        gwps = {
            "Otimista (GWP-20)": (79.7, 273),
            "Realista (GWP-100)": (27.0, 273),
            "Pessimista (GWP-500)": (7.2, 130)
        }

        # Resultados determinísticos para cada GWP
        results_all = {}
        for nome, (gwp_ch4, gwp_n2o) in gwps.items():
            results_all[nome] = compute_results_for_gwp(
                gwp_ch4, gwp_n2o, residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao
            )

        # Exibir resultados principais (cenário otimista)
        res_otimista = results_all["Otimista (GWP-20)"]
        evitado_thermo = res_otimista['thermophilic']['avoided_co2eq_t']
        evitado_windrow = res_otimista['windrow']['avoided_co2eq_t']

        st.header("📈 Resultados da Simulação (GWP-20)")
        st.info(f"""
        **Parâmetros calibrados para Ribeirão Preto:**
        - k = {formatar_br(k_ano)} ano⁻¹, T = {formatar_br(T)} °C, DOC = {formatar_br(DOC)}, Umidade = {formatar_br(umidade_valor)}%
        - Resíduos totais: {formatar_br(residuos_kg_dia * 365 * anos_simulacao / 1000)} t
        - **Aterro:** MCF = 1,0; captura = 60%; φ = 0,85
        - **Compostagem termofílica:** Yang et al. (2017)
        - **Compostagem em leiras:** TOOL13 (0,002 t CH₄/t; 0,0005 t N₂O/t)
        """)

        st.subheader("📊 Emissões Evitadas (tCO₂eq) – Comparação entre Cenários GWP")
        df_gwp = pd.DataFrame([{
            "Cenário": nome,
            "Termofílica": res['thermophilic']['avoided_co2eq_t'],
            "Leiras (TOOL13)": res['windrow']['avoided_co2eq_t']
        } for nome, res in results_all.items()])
        st.dataframe(df_gwp.style.format({c: lambda x: formatar_br(x) for c in df_gwp.columns if c != "Cenário"}))

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

        # Análise de sensibilidade Sobol para cada tecnologia (GWP-20)
        st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - GWP-20")
        problem = {'num_vars':3, 'names':['k','T','DOC'], 'bounds':[[0.06,0.40],[20,40],[0.10,0.25]]}
        param_values = sample(problem, n_samples, seed=50)
        g20_ch4, g20_n2o = gwps["Otimista (GWP-20)"]

        with st.spinner("Calculando Sobol para termofílica..."):
            res_thermo = Parallel(n_jobs=1)(delayed(sobol_thermo)(p, g20_ch4, g20_n2o) for p in param_values)
            Si_thermo = analyze(problem, np.array(res_thermo), print_to_console=False)
        with st.spinner("Calculando Sobol para leiras..."):
            res_wind = Parallel(n_jobs=1)(delayed(sobol_windrow)(p, g20_ch4, g20_n2o) for p in param_values)
            Si_wind = analyze(problem, np.array(res_wind), print_to_console=False)

        df_sens = pd.DataFrame({
            'Parâmetro': ['k','T','DOC'],
            'S1_Termofílica': Si_thermo['S1'], 'ST_Termofílica': Si_thermo['ST'],
            'S1_Leiras': Si_wind['S1'], 'ST_Leiras': Si_wind['ST']
        })
        st.dataframe(df_sens.style.format({c:'{:.4f}' for c in df_sens.columns if c != 'Parâmetro'}))

        # Monte Carlo e estatísticas de diferença significativa
        st.subheader("🎲 Análise de Incerteza (Monte Carlo) e Comparação Estatística")
        u_mc, t_mc, d_mc = gerar_parametros_mc(n_simulations)
        arr_thermo = []
        arr_windrow = []
        for i in range(n_simulations):
            calc_mc = GHGEmissionCalculator()
            calc_mc.GWP_CH4_20, calc_mc.GWP_N2O_20 = g20_ch4, g20_n2o
            res_mc = calc_mc.calculate_avoided_emissions(
                residuos_kg_dia, k_ano, t_mc[i], d_mc[i], u_mc[i], anos_simulacao
            )
            arr_thermo.append(res_mc['thermophilic']['avoided_co2eq_t'])
            arr_windrow.append(res_mc['windrow']['avoided_co2eq_t'])
        arr_thermo = np.array(arr_thermo)
        arr_windrow = np.array(arr_windrow)
        diff = arr_thermo - arr_windrow

        # Testes estatísticos
        shapiro_stat, shapiro_p = stats.shapiro(diff)
        t_stat, t_p = stats.ttest_rel(arr_thermo, arr_windrow)
        w_stat, w_p = stats.wilcoxon(arr_thermo, arr_windrow)

        st.write(f"**Teste de normalidade (Shapiro-Wilk) da diferença:** estatística = {shapiro_stat:.5f}, p = {shapiro_p:.5f}")
        st.write(f"**Teste t pareado:** t = {t_stat:.5f}, p = {t_p:.5f}")
        st.write(f"**Teste de Wilcoxon:** estatística = {w_stat:.5f}, p = {w_p:.5f}")

        st.subheader("📊 Estatísticas Descritivas das Emissões Evitadas (Monte Carlo)")
        stats_df = pd.DataFrame([
            {"Tecnologia": "Termofílica", "Média": np.mean(arr_thermo), "Mediana": np.median(arr_thermo),
             "Desvio Padrão": np.std(arr_thermo), "IC 95% Inf": np.percentile(arr_thermo,2.5),
             "IC 95% Sup": np.percentile(arr_thermo,97.5)},
            {"Tecnologia": "Leiras", "Média": np.mean(arr_windrow), "Mediana": np.median(arr_windrow),
             "Desvio Padrão": np.std(arr_windrow), "IC 95% Inf": np.percentile(arr_windrow,2.5),
             "IC 95% Sup": np.percentile(arr_windrow,97.5)}
        ])
        st.dataframe(stats_df.style.format({c: lambda x: formatar_br(x) for c in stats_df.columns if c != "Tecnologia"}))

        # Gráfico de distribuições
        fig, ax = plt.subplots(figsize=(10,6))
        sns.kdeplot(arr_thermo, label="Termofílica", linewidth=2, ax=ax)
        sns.kdeplot(arr_windrow, label="Leiras (TOOL13)", linewidth=2, ax=ax)
        ax.set_title("Distribuição das Emissões Evitadas (Monte Carlo)")
        ax.set_xlabel("tCO₂eq")
        ax.set_ylabel("Densidade")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(br_format))
        st.pyplot(fig)

        # Gráfico de redução acumulada (série temporal determinística)
        st.subheader("📉 Evolução da Redução de Emissões Acumulada (Cenário Otimista)")
        # Recalcular séries diárias para o gráfico
        calc = GHGEmissionCalculator()
        calc.GWP_CH4_20, calc.GWP_N2O_20 = g20_ch4, g20_n2o
        ch4_land, n2o_land = calc.calculate_landfill_emissions(residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao)
        ch4_thermo, n2o_thermo = calc.calculate_thermophilic_emissions(residuos_kg_dia, umidade, anos_simulacao)
        ch4_wind, n2o_wind = calc.calculate_windrow_emissions(residuos_kg_dia, umidade, anos_simulacao)
        base_acum = (ch4_land * g20_ch4 + n2o_land * g20_n2o).cumsum() / 1000
        thermo_acum = (ch4_thermo * g20_ch4 + n2o_thermo * g20_n2o).cumsum() / 1000
        wind_acum = (ch4_wind * g20_ch4 + n2o_wind * g20_n2o).cumsum() / 1000
        dias = np.arange(len(base_acum))
        datas = pd.date_range(start=datetime.now(), periods=len(base_acum), freq='D')
        fig2, ax2 = plt.subplots(figsize=(10,6))
        ax2.plot(datas, base_acum, 'r-', label='Baseline (Aterro)', linewidth=2)
        ax2.plot(datas, thermo_acum, 'orange', label='Termofílica', linewidth=2)
        ax2.plot(datas, wind_acum, 'green', label='Leiras (TOOL13)', linewidth=2)
        ax2.fill_between(datas, thermo_acum, wind_acum, color='gray', alpha=0.3, label='Diferença entre tecnologias')
        ax2.set_title('Emissões Acumuladas (tCO₂eq)')
        ax2.set_xlabel('Data')
        ax2.set_ylabel('tCO₂eq')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(FuncFormatter(br_format))
        st.pyplot(fig2)

        # Tabela anual comparativa
        st.subheader("📋 Resultados Anuais (Cenário Otimista)")
        df_anual = pd.DataFrame({
            'Year': datas.year.unique()[:anos_simulacao],
            'Baseline (tCO₂eq)': [base_acum[d*365+364] - (base_acum[d*365-1] if d>0 else 0) for d in range(anos_simulacao)],
            'Termofílica (tCO₂eq)': [thermo_acum[d*365+364] - (thermo_acum[d*365-1] if d>0 else 0) for d in range(anos_simulacao)],
            'Leiras (tCO₂eq)': [wind_acum[d*365+364] - (wind_acum[d*365-1] if d>0 else 0) for d in range(anos_simulacao)]
        })
        df_anual['Redução Termofílica'] = df_anual['Baseline (tCO₂eq)'] - df_anual['Termofílica (tCO₂eq)']
        df_anual['Redução Leiras'] = df_anual['Baseline (tCO₂eq)'] - df_anual['Leiras (tCO₂eq)']
        df_anual_fmt = df_anual.copy()
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
- **Dados operacionais do aterro CGR Guatapará (Ribeirão Preto):** usina de biogás com captura estimada de 60% do metano gerado.
""")
