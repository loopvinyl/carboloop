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
import yfinance as yf  # para obter cotação do carbono

# Semente fixa para reprodutibilidade
np.random.seed(50)

# Configuração da página Streamlit
st.set_page_config(
    page_title="Simulador de Emissões de GEE e Créditos de Carbono: Yang et al. 2017 vs TOOL13/AMS‑III.F com Baseline Aterro (Ribeirão Preto)",
    layout="wide"
)

# Suprimir warnings futuros e ajustar formatação
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')

# Configurações de estilo para gráficos
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

# Fatores de emissão para compostagem convencional em leiras (TOOL13 / AMS‑III.F)
EF_CH4_WINDROW = 0.002               # t CH₄ / t resíduo úmido
EF_N2O_WINDROW = 0.0005              # t N₂O / t resíduo úmido

# Parâmetros fixos baseados na literatura (Yang et al. 2017 para vermi e termo)
TOC = 0.436                # Carbono orgânico total
TN = 0.0142                # Nitrogênio total
F_CH4_VERMI = 0.0013       # Fração de CH4 na vermicompostagem (Yang et al. 2017)
F_N2O_VERMI = 0.0092       # Fração de N2O na vermicompostagem (Yang et al. 2017)
F_CH4_THERMO = 0.0060      # Fração de CH4 na compostagem termofílica (Yang et al. 2017)
F_N2O_THERMO = 0.0196      # Fração de N2O na compostagem termofílica (Yang et al. 2017)
COMPOSTING_DAYS = 50       # Duração do processo de compostagem (dias)
GWP_CH4_20 = 79.7          # GWP-20 para CH4 (Forster et al. 2021)
GWP_N2O_20 = 273           # GWP-20 para N2O (Forster et al. 2021)

# =============================================================================
# PERFIS DE EMISSÃO DIÁRIOS (carregados uma única vez)
# =============================================================================
profile_ch4_vermi = np.array([
    0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06,
    0.07, 0.08, 0.09, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04,
    0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
    0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001
])
profile_ch4_vermi /= profile_ch4_vermi.sum()

profile_n2o_vermi = np.array([
    0.15, 0.10, 0.20, 0.05, 0.03, 0.03, 0.03, 0.04, 0.05, 0.06,
    0.08, 0.09, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02,
    0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
    0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001,
    0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
])
profile_n2o_vermi /= profile_n2o_vermi.sum()

# Os mesmos perfis são usados para termofílica e windrow
profile_ch4_thermo = profile_ch4_vermi.copy()
profile_n2o_thermo = profile_n2o_vermi.copy()

profile_n2o_landfill = {1: 0.10, 2: 0.30, 3: 0.40, 4: 0.15, 5: 0.05}
profile_n2o_pre = {1: 0.8623, 2: 0.10, 3: 0.0377}

# Emissões de pré-descarte (constantes)
CH4_pre_ugC_per_kg_h = 2.78
CH4_pre_kg_per_kg_day = CH4_pre_ugC_per_kg_h * (16/12) * 24 / 1_000_000_000
N2O_pre_mgN_per_kg_total = 20.26
N2O_pre_kg_per_kg_total = N2O_pre_mgN_per_kg_total * (44/28) / 1_000_000

# =============================================================================
# CLASSE PARA CÁLCULO DE EMISSÕES DE GEE (ATUALIZADA COM WINDROW)
# =============================================================================
class GHGEmissionCalculator:
    """
    Calcula emissões de CH₄ e N₂O para:
    - Aterro sanitário (baseline, método FOD do IPCC) calibrado para Ribeirão Preto
    - Vermicompostagem (Yang et al. 2017)
    - Compostagem termofílica (Yang et al. 2017)
    - Compostagem convencional em leiras (windrow) - TOOL13/AMS‑III.F
    Inclui correção φ (UNFCCC 2024) e fator de captura de metano.
    """

    def __init__(self):
        # Parâmetros fixos baseados na literatura
        self.TOC = TOC
        self.TN = TN
        self.f_CH4_vermi = F_CH4_VERMI
        self.f_N2O_vermi = F_N2O_VERMI
        self.f_CH4_thermo = F_CH4_THERMO
        self.f_N2O_thermo = F_N2O_THERMO
        self.EF_CH4_windrow = EF_CH4_WINDROW
        self.EF_N2O_windrow = EF_N2O_WINDROW
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

    def calculate_landfill_emissions(self, waste_kg_day, k_year, temperature_C,
                                     doc_fraction, moisture_fraction, years=20,
                                     phi=PHI_BASELINE, capture_fraction=CAPTURE_FRACTION_BASELINE):
        """
        Emissões do aterro sanitário (método FOD do IPCC).
        Parâmetros calibrados para Ribeirão Preto: phi=0.85, capture_fraction=0.6.
        """
        days = years * 365
        docf = 0.0147 * temperature_C + 0.28
        ch4_potential_per_kg = (doc_fraction * docf * self.MCF * self.F * (16/12) *
                                (1 - self.Ri) * (1 - self.OX))
        ch4_potential_daily = waste_kg_day * ch4_potential_per_kg

        t = np.arange(1, days + 1, dtype=float)
        kernel_ch4 = np.exp(-k_year * (t - 1) / 365.0) - np.exp(-k_year * t / 365.0)
        daily_inputs = np.ones(days, dtype=float)
        ch4_emissions = np.convolve(daily_inputs, kernel_ch4, mode='full')[:days]
        ch4_emissions *= ch4_potential_daily
        ch4_emissions = ch4_emissions * phi * (1 - capture_fraction)

        # Emissões de N2O (Wang et al. 2017)
        exposed_mass = 100
        exposed_hours = 8
        opening_factor = (exposed_mass / waste_kg_day) * (exposed_hours / 24)
        opening_factor = np.clip(opening_factor, 0.0, 1.0)
        E_open = 1.91
        E_closed = 2.15
        E_avg = opening_factor * E_open + (1 - opening_factor) * E_closed
        moisture_factor = (1 - moisture_fraction) / (1 - 0.55)
        E_avg_adjusted = E_avg * moisture_factor
        daily_n2o_kg = (E_avg_adjusted * (44/28) / 1_000_000) * waste_kg_day

        kernel_n2o = np.array([self.profile_n2o_landfill.get(d, 0) for d in range(1, 6)], dtype=float)
        n2o_emissions = np.convolve(np.full(days, daily_n2o_kg), kernel_n2o, mode='full')[:days]

        # Pré-descarte
        ch4_pre = np.full(days, waste_kg_day * self.CH4_pre_kg_per_kg_day)
        n2o_pre = np.zeros(days)
        for entry_day in range(days):
            for days_after, fraction in self.profile_n2o_pre.items():
                emission_day = entry_day + days_after - 1
                if emission_day < days:
                    n2o_pre[emission_day] += waste_kg_day * self.N2O_pre_kg_per_kg_total * fraction
        return ch4_emissions + ch4_pre, n2o_emissions + n2o_pre

    def calculate_vermicomposting_emissions(self, waste_kg_day, moisture_fraction, years=20):
        """Emissões da vermicompostagem (Yang et al. 2017)."""
        days = years * 365
        dry_fraction = 1 - moisture_fraction
        ch4_per_batch = (waste_kg_day * self.TOC * self.f_CH4_vermi * (16/12) * dry_fraction)
        n2o_per_batch = (waste_kg_day * self.TN * self.f_N2O_vermi * (44/28) * dry_fraction)
        ch4_emissions = np.zeros(days)
        n2o_emissions = np.zeros(days)
        for entry_day in range(days):
            for compost_day in range(self.COMPOSTING_DAYS):
                emission_day = entry_day + compost_day
                if emission_day < days:
                    ch4_emissions[emission_day] += ch4_per_batch * self.profile_ch4_vermi[compost_day]
                    n2o_emissions[emission_day] += n2o_per_batch * self.profile_n2o_vermi[compost_day]
        return ch4_emissions, n2o_emissions

    def calculate_thermophilic_emissions(self, waste_kg_day, moisture_fraction, years=20):
        """Emissões da compostagem termofílica (Yang et al. 2017)."""
        days = years * 365
        dry_fraction = 1 - moisture_fraction
        ch4_per_batch = (waste_kg_day * self.TOC * self.f_CH4_thermo * (16/12) * dry_fraction)
        n2o_per_batch = (waste_kg_day * self.TN * self.f_N2O_thermo * (44/28) * dry_fraction)
        ch4_emissions = np.zeros(days)
        n2o_emissions = np.zeros(days)
        for entry_day in range(days):
            for compost_day in range(self.COMPOSTING_DAYS):
                emission_day = entry_day + compost_day
                if emission_day < days:
                    ch4_emissions[emission_day] += ch4_per_batch * self.profile_ch4_thermo[compost_day]
                    n2o_emissions[emission_day] += n2o_per_batch * self.profile_n2o_thermo[compost_day]
        return ch4_emissions, n2o_emissions

    def calculate_windrow_emissions(self, waste_kg_day, moisture_fraction, years=20):
        """Emissões da compostagem convencional em leiras (TOOL13 / AMS‑III.F)."""
        days = years * 365
        # Fatores de emissão em t por t úmido -> converter para kg por kg
        ch4_per_kg = self.EF_CH4_windrow / 1000.0
        n2o_per_kg = self.EF_N2O_windrow / 1000.0
        ch4_batch_kg = waste_kg_day * ch4_per_kg
        n2o_batch_kg = waste_kg_day * n2o_per_kg
        ch4_emissions = np.zeros(days)
        n2o_emissions = np.zeros(days)
        for entry_day in range(days):
            for compost_day in range(self.COMPOSTING_DAYS):
                emission_day = entry_day + compost_day
                if emission_day < days:
                    ch4_emissions[emission_day] += ch4_batch_kg * self.profile_ch4_vermi[compost_day]
                    n2o_emissions[emission_day] += n2o_batch_kg * self.profile_n2o_vermi[compost_day]
        return ch4_emissions, n2o_emissions

    def calculate_avoided_emissions(self, waste_kg_day, k_year, temperature_C,
                                    doc_fraction, moisture_fraction, years=20,
                                    phi_baseline=PHI_BASELINE, capture_fraction=CAPTURE_FRACTION_BASELINE):
        """Calcula emissões evitadas (tCO₂eq) para as três tecnologias."""
        ch4_landfill, n2o_landfill = self.calculate_landfill_emissions(
            waste_kg_day, k_year, temperature_C, doc_fraction, moisture_fraction, years,
            phi=phi_baseline, capture_fraction=capture_fraction
        )
        ch4_vermi, n2o_vermi = self.calculate_vermicomposting_emissions(waste_kg_day, moisture_fraction, years)
        ch4_thermo, n2o_thermo = self.calculate_thermophilic_emissions(waste_kg_day, moisture_fraction, years)
        ch4_wind, n2o_wind = self.calculate_windrow_emissions(waste_kg_day, moisture_fraction, years)

        baseline_co2eq = (ch4_landfill * self.GWP_CH4_20 + n2o_landfill * self.GWP_N2O_20) / 1000
        vermi_co2eq = (ch4_vermi * self.GWP_CH4_20 + n2o_vermi * self.GWP_N2O_20) / 1000
        thermo_co2eq = (ch4_thermo * self.GWP_CH4_20 + n2o_thermo * self.GWP_N2O_20) / 1000
        wind_co2eq = (ch4_wind * self.GWP_CH4_20 + n2o_wind * self.GWP_N2O_20) / 1000

        avoided_vermi = baseline_co2eq.sum() - vermi_co2eq.sum()
        avoided_thermo = baseline_co2eq.sum() - thermo_co2eq.sum()
        avoided_wind = baseline_co2eq.sum() - wind_co2eq.sum()

        results = {
            'baseline': {
                'ch4_kg': ch4_landfill.sum(),
                'n2o_kg': n2o_landfill.sum(),
                'co2eq_t': baseline_co2eq.sum()
            },
            'vermicomposting': {
                'ch4_kg': ch4_vermi.sum(),
                'n2o_kg': n2o_vermi.sum(),
                'co2eq_t': vermi_co2eq.sum(),
                'avoided_co2eq_t': avoided_vermi
            },
            'composting': {   # termofílica
                'ch4_kg': ch4_thermo.sum(),
                'n2o_kg': n2o_thermo.sum(),
                'co2eq_t': thermo_co2eq.sum(),
                'avoided_co2eq_t': avoided_thermo
            },
            'windrow': {      # convencional em leiras (TOOL13)
                'ch4_kg': ch4_wind.sum(),
                'n2o_kg': n2o_wind.sum(),
                'co2eq_t': wind_co2eq.sum(),
                'avoided_co2eq_t': avoided_wind
            },
            'comparison': {
                'difference_vermi_thermo': avoided_vermi - avoided_thermo,
                'difference_vermi_wind': avoided_vermi - avoided_wind,
                'difference_thermo_wind': avoided_thermo - avoided_wind
            },
            'annual_averages': {
                'baseline_tco2eq_year': baseline_co2eq.sum() / years,
                'vermi_avoided_year': avoided_vermi / years,
                'thermo_avoided_year': avoided_thermo / years,
                'wind_avoided_year': avoided_wind / years
            }
        }
        return results

    # Método rápido (apenas totais) para Monte Carlo e Sobol – retorna evitado para as três
    def calculate_avoided_emissions_fast(self, waste_kg_day, k_year, temperature_C,
                                         doc_fraction, moisture_fraction, years):
        ch4_l, n2o_l = self.calculate_landfill_emissions(waste_kg_day, k_year, temperature_C,
                                                         doc_fraction, moisture_fraction, years)
        ch4_v, n2o_v = self.calculate_vermicomposting_emissions(waste_kg_day, moisture_fraction, years)
        ch4_t, n2o_t = self.calculate_thermophilic_emissions(waste_kg_day, moisture_fraction, years)
        ch4_w, n2o_w = self.calculate_windrow_emissions(waste_kg_day, moisture_fraction, years)

        base = (ch4_l*self.GWP_CH4_20 + n2o_l*self.GWP_N2O_20)/1000
        vermi = (ch4_v*self.GWP_CH4_20 + n2o_v*self.GWP_N2O_20)/1000
        thermo = (ch4_t*self.GWP_CH4_20 + n2o_t*self.GWP_N2O_20)/1000
        wind = (ch4_w*self.GWP_CH4_20 + n2o_w*self.GWP_N2O_20)/1000

        return (base.sum() - vermi.sum()), (base.sum() - thermo.sum()), (base.sum() - wind.sum())


# =============================================================================
# FUNÇÕES DE COTAÇÃO (MERCADO DE CARBONO E CÂMBIO)
# =============================================================================
def obter_cotacao_carbono():
    """Obtém a cotação do carbono via Yahoo Finance (ticker CO2.L)."""
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
    """Obtém a cotação EUR/BRL."""
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

# =============================================================================
# FUNÇÕES AUXILIARES DE FORMATAÇÃO BRASILEIRA
# =============================================================================
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

# =============================================================================
# INTERFACE STREAMLIT – BARRA LATERAL E EXIBIÇÃO DE COTAÇÕES
# =============================================================================
def exibir_cotacao_carbono():
    """Exibe na barra lateral os preços do carbono e do câmbio EUR/BRL."""
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
        preco_euro, moeda_real, _, fonte_euro = obter_cotacao_euro_real()
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

        **🌍 Mercado de Referência:**
        - European Union Allowances (EUA)
        - European Emissions Trading System (EU ETS)
        - Contratos futuros de carbono (ICE CO2.L)
        - Preços em tempo real via Yahoo Finance

        **🔄 Atualização:**
        - As cotações são carregadas automaticamente ao abrir o aplicativo
        - Clique em **"Atualizar Cotações"** para obter valores mais recentes
        - Em caso de falha, são utilizados valores de referência.
        """)

def inicializar_session_state():
    """Inicializa as variáveis de estado do Streamlit."""
    if 'preco_carbono' not in st.session_state:
        preco_carbono, moeda, _, _, fonte = obter_cotacao_carbono()
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.fonte_cotacao = fonte
    if 'taxa_cambio' not in st.session_state:
        preco_euro, moeda_real, _, _ = obter_cotacao_euro_real()
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
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

# =============================================================================
# INTERFACE PRINCIPAL E PARÂMETROS DE ENTRADA
# =============================================================================
st.title("Comparação de Tecnologias de Compostagem para Créditos de Carbono")
st.markdown("""
**Tecnologias avaliadas:**
- **Vermicompostagem** – fatores de emissão de **Yang et al. (2017)**
- **Compostagem termofílica** – fatores de emissão de **Yang et al. (2017)**
- **Compostagem convencional em leiras (windrow)** – fatores padrão do **TOOL13** (AMS‑III.F da UNFCCC)

O cenário **baseline** é o aterro sanitário calibrado para o **Aterro CGR Guatapará (Ribeirão Preto)** com captura de metano de 60% e fator φ = 0,85 (UNFCCC 2024 – clima úmido).
""")

exibir_cotacao_carbono()

with st.sidebar:
    st.header("⚙️ Parâmetros de Entrada")
    residuos_kg_dia = st.slider("Quantidade de resíduos (kg/dia)", min_value=10, max_value=1000, value=100, step=10)

    st.subheader("📊 Parâmetros da Análise Sobol")
    st.info("Estes são os parâmetros variados na análise de sensibilidade Sobol")

    st.markdown("**1. Taxa de Decaimento do Aterro**")
    opcao_k = st.selectbox(
        "Selecione a taxa de decaimento (k)",
        options=[
            "k = 0.06 ano⁻¹ (decaimento lento - valor padrão)",
            "k = 0.40 ano⁻¹ (decaimento rápido)"
        ],
        index=0
    )
    k_ano = 0.40 if "0.40" in opcao_k else 0.06
    st.session_state.k_ano = k_ano
    st.write(f"**Valor selecionado:** {formatar_br(k_ano)} ano⁻¹")

    st.markdown("**2. Temperatura Média**")
    T = st.slider("Temperatura média (°C)", min_value=20, max_value=40, value=25, step=1)
    st.write(f"**Valor selecionado:** {formatar_br(T)} °C")

    st.markdown("**3. Carbono Orgânico Degradável**")
    DOC = st.slider("DOC (fração)", min_value=0.10, max_value=0.25, value=0.15, step=0.01)
    st.write(f"**Valor selecionado:** {formatar_br(DOC)}")

    st.markdown("**4. Umidade do Resíduo**")
    umidade_valor = st.slider("Umidade do resíduo (%)", 50, 95, 85, 1,
                              help="Valor fixo (não varia na análise Sobol)")
    umidade = umidade_valor / 100.0
    st.write(f"**Valor fixo:** {formatar_br(umidade_valor)}%")

    with st.expander("ℹ️ Sobre os parâmetros da análise Sobol"):
        st.markdown("""
        **📊 Parâmetros variados na análise de sensibilidade Sobol:**
        1. **Taxa de decaimento (k):** 0.06 a 0.40 ano⁻¹
        2. **Temperatura (T):** 20 a 40°C
        3. **Carbono orgânico degradável (DOC):** 0.10 a 0.25

        **⚙️ Parâmetro fixo (não varia):** Umidade (85%)

        **🔬 Origem dos fatores de emissão:**
        - **Vermicompostagem e Termofílica:** Yang et al. (2017) – *Waste Management*.
        - **Convencional (Leiras):** TOOL13 da UNFCCC (AMS‑III.F) – fatores conservadores para projetos de compostagem.
        """)

    st.subheader("🎯 Configuração de Simulação")
    anos_simulacao = st.slider("Anos de simulação", 5, 50, 20, 5)
    n_simulations = st.slider("Número de simulações Monte Carlo", 50, 1000, 100, 50)
    n_samples = st.slider("Número de amostras Sobol", 32, 256, 64, 16)

    # Botão que aciona a simulação
    if st.button("🚀 Executar Simulação", type="primary"):
        st.session_state.run_simulation = True

# =============================================================================
# FUNÇÕES COM CACHE PARA ANÁLISES PESADAS (otimização)
# =============================================================================
@st.cache_data(show_spinner=False)
def cached_sobol(n_samples, residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao, gwp_ch4, gwp_n2o):
    """Executa análise de sensibilidade Sobol com cache para as três tecnologias."""
    problem = {'num_vars':3, 'names':['k','T','DOC'], 'bounds':[[0.06,0.40],[20,40],[0.10,0.25]]}
    param_values = sample(problem, n_samples, seed=50)
    calc = GHGEmissionCalculator()
    calc.GWP_CH4_20 = gwp_ch4
    calc.GWP_N2O_20 = gwp_n2o

    def results_from_params(p):
        # Retorna (vermi_avoided, thermo_avoided, wind_avoided)
        return calc.calculate_avoided_emissions_fast(residuos_kg_dia, p[0], p[1], p[2], umidade, anos_simulacao)

    all_results = Parallel(n_jobs=-1)(delayed(results_from_params)(p) for p in param_values)
    arr_vermi = np.array([r[0] for r in all_results])
    arr_thermo = np.array([r[1] for r in all_results])
    arr_wind = np.array([r[2] for r in all_results])

    Si_vermi = analyze(problem, arr_vermi, print_to_console=False)
    Si_thermo = analyze(problem, arr_thermo, print_to_console=False)
    Si_wind = analyze(problem, arr_wind, print_to_console=False)
    return Si_vermi, Si_thermo, Si_wind

@st.cache_data(show_spinner=False)
def cached_montecarlo(n_simulations, residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao, gwp_ch4, gwp_n2o):
    """Executa Monte Carlo com cache para as três tecnologias."""
    np.random.seed(50)
    u = np.random.uniform(0.75, 0.90, n_simulations)
    t = np.random.normal(25, 3, n_simulations)
    d = np.random.triangular(0.12, 0.15, 0.18, n_simulations)

    calc = GHGEmissionCalculator()
    calc.GWP_CH4_20 = gwp_ch4
    calc.GWP_N2O_20 = gwp_n2o

    def run_one(i):
        np.random.seed(50 + i)
        return calc.calculate_avoided_emissions_fast(residuos_kg_dia, k_ano, t[i], d[i], u[i], anos_simulacao)

    resultados = Parallel(n_jobs=-1)(delayed(run_one)(i) for i in range(n_simulations))
    arr_vermi = np.array([r[0] for r in resultados])
    arr_thermo = np.array([r[1] for r in resultados])
    arr_wind = np.array([r[2] for r in resultados])
    return arr_vermi, arr_thermo, arr_wind

# =============================================================================
# EXECUÇÃO DA SIMULAÇÃO (QUANDO BOTÃO FOR CLICADO)
# =============================================================================
if st.session_state.get('run_simulation', False):
    with st.spinner('Executando simulação (determinística, Sobol e Monte Carlo)...'):
        # Cenários de GWP
        gwps = {
            "Otimista (GWP-20)": (79.7, 273),
            "Realista (GWP-100)": (27.0, 273),
            "Pessimista (GWP-500)": (7.2, 130)
        }

        # --- RESULTADOS DETERMINÍSTICOS PARA CADA GWP ---
        results_all = {}
        for nome, (gwp_ch4, gwp_n2o) in gwps.items():
            calc_temp = GHGEmissionCalculator()
            calc_temp.GWP_CH4_20 = gwp_ch4
            calc_temp.GWP_N2O_20 = gwp_n2o
            results_all[nome] = calc_temp.calculate_avoided_emissions(
                residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao
            )

        # Usamos o cenário otimista (GWP-20) para gráficos e tabelas principais
        results = results_all["Otimista (GWP-20)"]

        # --- GERAÇÃO DE DADOS DIÁRIOS PARA GRÁFICOS (APENAS GWP-20) ---
        dias = anos_simulacao * 365
        datas = pd.date_range(start=datetime.now(), periods=dias, freq='D')

        calc_g20 = GHGEmissionCalculator()
        calc_g20.GWP_CH4_20, calc_g20.GWP_N2O_20 = gwps["Otimista (GWP-20)"]

        # Emissões baseline (aterro)
        ch4_aterro_dia, n2o_aterro_dia = calc_g20.calculate_landfill_emissions(
            residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao
        )
        # Emissões vermicompostagem
        ch4_vermi_dia, n2o_vermi_dia = calc_g20.calculate_vermicomposting_emissions(residuos_kg_dia, umidade, anos_simulacao)
        # Emissões termofílica
        ch4_thermo_dia, n2o_thermo_dia = calc_g20.calculate_thermophilic_emissions(residuos_kg_dia, umidade, anos_simulacao)
        # Emissões windrow
        ch4_wind_dia, n2o_wind_dia = calc_g20.calculate_windrow_emissions(residuos_kg_dia, umidade, anos_simulacao)

        # Conversão para tCO₂eq por dia
        base_dia = (ch4_aterro_dia * GWP_CH4_20 + n2o_aterro_dia * GWP_N2O_20) / 1000
        vermi_dia = (ch4_vermi_dia * GWP_CH4_20 + n2o_vermi_dia * GWP_N2O_20) / 1000
        thermo_dia = (ch4_thermo_dia * GWP_CH4_20 + n2o_thermo_dia * GWP_N2O_20) / 1000
        wind_dia = (ch4_wind_dia * GWP_CH4_20 + n2o_wind_dia * GWP_N2O_20) / 1000

        df_dia = pd.DataFrame({
            'Data': datas,
            'Base': base_dia,
            'Vermi': vermi_dia,
            'Termo': thermo_dia,
            'Wind': wind_dia
        })
        df_dia['Year'] = df_dia['Data'].dt.year
        df_anual = df_dia.groupby('Year').agg({
            'Base': 'sum',
            'Vermi': 'sum',
            'Termo': 'sum',
            'Wind': 'sum'
        }).reset_index()
        df_anual['Evitado_Vermi'] = df_anual['Base'] - df_anual['Vermi']
        df_anual['Evitado_Termo'] = df_anual['Base'] - df_anual['Termo']
        df_anual['Evitado_Wind'] = df_anual['Base'] - df_anual['Wind']

        # Acumulados
        base_acum = np.cumsum(base_dia)
        vermi_acum = np.cumsum(vermi_dia)
        termo_acum = np.cumsum(thermo_dia)
        wind_acum = np.cumsum(wind_dia)

        # --- EXIBIÇÃO DE RESULTADOS ---
        st.header("📈 Resultados da Simulação (Cenário Otimista GWP-20)")
        st.info(f"""
        **Parâmetros calibrados para Ribeirão Preto (Aterro CGR Guatapará):**
        - Taxa de decaimento (k): {formatar_br(k_ano)} ano⁻¹
        - Temperatura (T): {formatar_br(T)} °C
        - DOC: {formatar_br(DOC)}
        - Umidade: {formatar_br(umidade_valor)}%
        - Resíduos/dia: {formatar_br(residuos_kg_dia)} kg
        - Total de resíduos: {formatar_br(residuos_kg_dia * 365 * anos_simulacao / 1000)} toneladas
        - **Baseline:** Captura de metano = 60%, φ = 0,85 (UNFCCC 2024 - clima úmido)
        """)

        # Tabela comparativa de GWPs (apenas vermi, termo e wind)
        st.subheader("📊 Comparação entre Cenários de GWP (Emissões Evitadas - tCO₂eq)")
        comparacao = []
        for nome, res in results_all.items():
            comparacao.append({
                "Cenário": nome,
                "Vermicompostagem (Yang et al. 2017)": res['vermicomposting']['avoided_co2eq_t'],
                "Compostagem Termofílica (Yang et al. 2017)": res['composting']['avoided_co2eq_t'],
                "Compostagem Convencional (TOOL13 / AMS‑III.F)": res['windrow']['avoided_co2eq_t']
            })
        df_comp_gwp = pd.DataFrame(comparacao)
        st.dataframe(df_comp_gwp.style.format({
            "Vermicompostagem (Yang et al. 2017)": lambda x: formatar_br(x),
            "Compostagem Termofílica (Yang et al. 2017)": lambda x: formatar_br(x),
            "Compostagem Convencional (TOOL13 / AMS‑III.F)": lambda x: formatar_br(x)
        }))

        # Valores financeiros (cenário otimista)
        total_evitado_vermi = results['vermicomposting']['avoided_co2eq_t']
        total_evitado_thermo = results['composting']['avoided_co2eq_t']
        total_evitado_wind = results['windrow']['avoided_co2eq_t']
        preco_carbono = st.session_state.preco_carbono
        moeda = st.session_state.moeda_carbono
        taxa_cambio = st.session_state.taxa_cambio
        fonte_cotacao = st.session_state.fonte_cotacao

        valor_vermi_eur = calcular_valor_creditos(total_evitado_vermi, preco_carbono, moeda)
        valor_thermo_eur = calcular_valor_creditos(total_evitado_thermo, preco_carbono, moeda)
        valor_wind_eur = calcular_valor_creditos(total_evitado_wind, preco_carbono, moeda)
        valor_vermi_brl = calcular_valor_creditos(total_evitado_vermi, preco_carbono, "R$", taxa_cambio)
        valor_thermo_brl = calcular_valor_creditos(total_evitado_thermo, preco_carbono, "R$", taxa_cambio)
        valor_wind_brl = calcular_valor_creditos(total_evitado_wind, preco_carbono, "R$", taxa_cambio)

        st.subheader("💰 Valor Financeiro das Emissões Evitadas (Cenário Otimista)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Preço Carbono (Euro)", f"{moeda} {formatar_br(preco_carbono)}/tCO₂eq",
                      help=f"Fonte: {fonte_cotacao}")
        with col2:
            st.metric("Vermicompostagem (Yang et al.) (Euro)", f"{moeda} {formatar_br(valor_vermi_eur)}",
                      help=f"{formatar_br(total_evitado_vermi)} tCO₂eq evitadas")
        with col3:
            st.metric("Termofílica (Yang et al.) (Euro)", f"{moeda} {formatar_br(valor_thermo_eur)}",
                      help=f"{formatar_br(total_evitado_thermo)} tCO₂eq evitadas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Convencional (TOOL13) (Euro)", f"{moeda} {formatar_br(valor_wind_eur)}",
                      help=f"{formatar_br(total_evitado_wind)} tCO₂eq evitadas")
        with col2:
            st.metric("Preço Carbono (R$)", f"R$ {formatar_br(preco_carbono * taxa_cambio)}/tCO₂eq",
                      help="Preço convertido para Reais")
        with col3:
            st.metric("Vermicompostagem (Yang et al.) (R$)", f"R$ {formatar_br(valor_vermi_brl)}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Termofílica (Yang et al.) (R$)", f"R$ {formatar_br(valor_thermo_brl)}")
        with col2:
            st.metric("Convencional (TOOL13) (R$)", f"R$ {formatar_br(valor_wind_brl)}")

        with st.expander("💡 Como funciona a comercialização no mercado de carbono?"):
            st.markdown(f"""
            **📊 Informações de Mercado:**
            - Preço em Euro: {moeda} {formatar_br(preco_carbono)}/tCO₂eq
            - Preço em Real: R$ {formatar_br(preco_carbono * taxa_cambio)}/tCO₂eq
            - Taxa de câmbio: 1 Euro = R$ {formatar_br(taxa_cambio)}
            - Fonte: {fonte_cotacao}
            **💶 Comprar créditos (compensação):** Exemplo para vermicompostagem: Custo em Euro: {moeda} {formatar_br(valor_vermi_eur)} | Custo em Real: R$ {formatar_br(valor_vermi_brl)}
            **💵 Vender créditos (comercialização):** Mesmos valores como receita.
            """)

        # Resumo emissões evitadas
        st.subheader("📊 Resumo das Emissões Evitadas (Cenário Otimista)")
        media_anual_vermi = total_evitado_vermi / anos_simulacao
        media_anual_thermo = total_evitado_thermo / anos_simulacao
        media_anual_wind = total_evitado_wind / anos_simulacao
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🪱 Vermicompostagem (Yang et al. 2017)")
            st.metric("Total evitado", f"{formatar_br(total_evitado_vermi)} tCO₂eq")
            st.metric("Média anual", f"{formatar_br(media_anual_vermi)} tCO₂eq/ano")
        with col2:
            st.markdown("#### 🔥 Termofílica (Yang et al. 2017)")
            st.metric("Total evitado", f"{formatar_br(total_evitado_thermo)} tCO₂eq")
            st.metric("Média anual", f"{formatar_br(media_anual_thermo)} tCO₂eq/ano")
        with col3:
            st.markdown("#### 🌿 Convencional (TOOL13 / AMS‑III.F)")
            st.metric("Total evitado", f"{formatar_br(total_evitado_wind)} tCO₂eq")
            st.metric("Média anual", f"{formatar_br(media_anual_wind)} tCO₂eq/ano")

        # Gráfico de barras: comparação anual das três tecnologias
        st.subheader("📊 Comparação Anual das Emissões Evitadas (GWP-20)")
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df_anual['Year']))
        width = 0.25
        ax.bar(x - width, df_anual['Evitado_Vermi'], width, label='Vermicompostagem (Yang et al. 2017)', edgecolor='black', color='forestgreen')
        ax.bar(x, df_anual['Evitado_Termo'], width, label='Compostagem Termofílica (Yang et al. 2017)', edgecolor='black', hatch='//', color='orange')
        ax.bar(x + width, df_anual['Evitado_Wind'], width, label='Compostagem Convencional (TOOL13 / AMS‑III.F)', edgecolor='black', hatch='\\\\', color='steelblue')
        # Anotações
        for i, (v1, v2, v3) in enumerate(zip(df_anual['Evitado_Vermi'], df_anual['Evitado_Termo'], df_anual['Evitado_Wind'])):
            ax.text(i - width, v1 + max(v1,v2,v3)*0.01, formatar_br(v1), ha='center', fontsize=8, fontweight='bold')
            ax.text(i, v2 + max(v1,v2,v3)*0.01, formatar_br(v2), ha='center', fontsize=8, fontweight='bold')
            ax.text(i + width, v3 + max(v1,v2,v3)*0.01, formatar_br(v3), ha='center', fontsize=8, fontweight='bold')
        ax.set_xlabel('Ano', fontsize=12)
        ax.set_ylabel('Emissões Evitadas (t CO₂eq)', fontsize=12)
        ax.set_title('Comparação Anual: Vermicompostagem vs Termofílica vs Convencional (TOOL13) – GWP-20', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(df_anual['Year'], fontsize=9)
        ax.legend()
        ax.yaxis.set_major_formatter(FuncFormatter(br_format))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

        # Gráfico de redução acumulada (baseline + três tecnologias)
        st.subheader("📉 Emissões Acumuladas: Baseline vs Tecnologias (GWP-20)")
        fig2, ax2 = plt.subplots(figsize=(11, 6))
        ax2.plot(datas, base_acum, 'r-', label='Baseline (Aterro Sanitário)', linewidth=2)
        ax2.plot(datas, vermi_acum, 'g-', label='Vermicompostagem (Yang et al. 2017)', linewidth=2)
        ax2.plot(datas, termo_acum, 'orange', label='Compostagem Termofílica (Yang et al. 2017)', linewidth=2)
        ax2.plot(datas, wind_acum, 'steelblue', label='Compostagem Convencional (TOOL13 / AMS‑III.F)', linewidth=2)
        ax2.fill_between(datas, vermi_acum, base_acum, color='lightgreen', alpha=0.3, label='Emissões evitadas (Vermi)')
        ax2.set_title(f'Emissões Acumuladas de tCO₂eq em {anos_simulacao} anos (k = {formatar_br(k_ano)} ano⁻¹)', fontsize=13)
        ax2.set_xlabel('Data', fontsize=12)
        ax2.set_ylabel('tCO₂eq Acumulado', fontsize=12)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.yaxis.set_major_formatter(FuncFormatter(br_format))
        st.pyplot(fig2)
        plt.close(fig2)

        # --- ANÁLISE DE SENSIBILIDADE SOBOL (GWP-20) PARA AS TRÊS TECNOLOGIAS ---
        st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - GWP-20")
        with st.spinner("Executando análise Sobol (paralelizada com cache)..."):
            g20_ch4, g20_n2o = gwps["Otimista (GWP-20)"]
            Si_vermi, Si_thermo, Si_wind = cached_sobol(
                n_samples, residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao, g20_ch4, g20_n2o
            )

        nomes_amigaveis = {'k': 'Taxa de Decaimento (k)', 'T': 'Temperatura', 'DOC': 'Carbono Orgânico Degradável'}
        df_sens_vermi = pd.DataFrame({
            'Parâmetro': ['k','T','DOC'],
            'S1': Si_vermi['S1'],
            'ST': Si_vermi['ST']
        })
        df_sens_vermi['Parâmetro'] = df_sens_vermi['Parâmetro'].map(nomes_amigaveis)

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.barplot(x='ST', y='Parâmetro', data=df_sens_vermi, palette='viridis', ax=ax3)
        ax3.set_title('Sensibilidade Global - Vermicompostagem (Yang et al. 2017) - GWP-20')
        ax3.set_xlabel('Índice ST (Sobol Total)')
        ax3.set_ylabel('')
        ax3.grid(axis='x', linestyle='--', alpha=0.7)
        ax3.xaxis.set_major_formatter(FuncFormatter(br_format))
        for i, st_val in enumerate(df_sens_vermi['ST']):
            ax3.text(st_val, i, f' {formatar_br(st_val)}', va='center', fontweight='bold')
        st.pyplot(fig3)
        plt.close(fig3)
        st.dataframe(df_sens_vermi.style.format({'S1': '{:.4f}', 'ST': '{:.4f}'}))

        # Termofílica
        df_sens_thermo = pd.DataFrame({
            'Parâmetro': ['k','T','DOC'],
            'S1': Si_thermo['S1'],
            'ST': Si_thermo['ST']
        })
        df_sens_thermo['Parâmetro'] = df_sens_thermo['Parâmetro'].map(nomes_amigaveis)
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.barplot(x='ST', y='Parâmetro', data=df_sens_thermo, palette='viridis', ax=ax4)
        ax4.set_title('Sensibilidade Global - Compostagem Termofílica (Yang et al. 2017) - GWP-20')
        ax4.set_xlabel('Índice ST (Sobol Total)')
        ax4.grid(axis='x', linestyle='--', alpha=0.7)
        ax4.xaxis.set_major_formatter(FuncFormatter(br_format))
        for i, st_val in enumerate(df_sens_thermo['ST']):
            ax4.text(st_val, i, f' {formatar_br(st_val)}', va='center', fontweight='bold')
        st.pyplot(fig4)
        plt.close(fig4)
        st.dataframe(df_sens_thermo.style.format({'S1': '{:.4f}', 'ST': '{:.4f}'}))

        # Windrow
        df_sens_wind = pd.DataFrame({
            'Parâmetro': ['k','T','DOC'],
            'S1': Si_wind['S1'],
            'ST': Si_wind['ST']
        })
        df_sens_wind['Parâmetro'] = df_sens_wind['Parâmetro'].map(nomes_amigaveis)
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        sns.barplot(x='ST', y='Parâmetro', data=df_sens_wind, palette='viridis', ax=ax5)
        ax5.set_title('Sensibilidade Global - Compostagem Convencional (TOOL13 / AMS‑III.F) - GWP-20')
        ax5.set_xlabel('Índice ST (Sobol Total)')
        ax5.grid(axis='x', linestyle='--', alpha=0.7)
        ax5.xaxis.set_major_formatter(FuncFormatter(br_format))
        for i, st_val in enumerate(df_sens_wind['ST']):
            ax5.text(st_val, i, f' {formatar_br(st_val)}', va='center', fontweight='bold')
        st.pyplot(fig5)
        plt.close(fig5)
        st.dataframe(df_sens_wind.style.format({'S1': '{:.4f}', 'ST': '{:.4f}'}))

        # --- MONTE CARLO E ANÁLISE ESTATÍSTICA (GWP-20) ---
        st.subheader("🎲 Análise de Incerteza (Monte Carlo) e Comparação Estatística - GWP-20")
        with st.spinner("Executando simulações Monte Carlo (paralelizado com cache)..."):
            arr_vermi_mc, arr_thermo_mc, arr_wind_mc = cached_montecarlo(
                n_simulations, residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao, g20_ch4, g20_n2o
            )

        # Distribuições
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        sns.kdeplot(arr_vermi_mc, label='Vermicompostagem (Yang et al. 2017)', linewidth=2, ax=ax6)
        sns.kdeplot(arr_thermo_mc, label='Compostagem Termofílica (Yang et al. 2017)', linewidth=2, ax=ax6)
        sns.kdeplot(arr_wind_mc, label='Compostagem Convencional (TOOL13 / AMS‑III.F)', linewidth=2, ax=ax6)
        ax6.set_title('Distribuição de Probabilidade das Emissões Evitadas – Monte Carlo (GWP-20)', fontsize=13)
        ax6.set_xlabel('Emissões Evitadas (tCO₂eq)', fontsize=12)
        ax6.set_ylabel('Densidade de Probabilidade', fontsize=12)
        ax6.legend()
        ax6.grid(alpha=0.3)
        ax6.xaxis.set_major_formatter(FuncFormatter(br_format))
        st.pyplot(fig6)
        plt.close(fig6)

        # Estatísticas descritivas
        stats_list = []
        for nome, arr in [("Vermicompostagem (Yang et al. 2017)", arr_vermi_mc),
                          ("Compostagem Termofílica (Yang et al. 2017)", arr_thermo_mc),
                          ("Compostagem Convencional (TOOL13 / AMS‑III.F)", arr_wind_mc)]:
            stats_list.append({
                "Tecnologia": nome,
                "Média (tCO₂eq)": np.mean(arr),
                "Mediana (tCO₂eq)": np.median(arr),
                "Desvio Padrão": np.std(arr),
                "IC 95% Inferior": np.percentile(arr, 2.5),
                "IC 95% Superior": np.percentile(arr, 97.5)
            })
        df_mc_stats = pd.DataFrame(stats_list)
        st.dataframe(df_mc_stats.style.format({
            "Média (tCO₂eq)": lambda x: formatar_br(x),
            "Mediana (tCO₂eq)": lambda x: formatar_br(x),
            "Desvio Padrão": lambda x: formatar_br(x),
            "IC 95% Inferior": lambda x: formatar_br(x),
            "IC 95% Superior": lambda x: formatar_br(x)
        }))

        # Testes estatísticos pareados: vermi vs termo, vermi vs wind, termo vs wind
        st.subheader("📊 Testes Estatísticos de Diferença Significativa (GWP-20)")
        diff_vt = arr_vermi_mc - arr_thermo_mc
        diff_vw = arr_vermi_mc - arr_wind_mc
        diff_tw = arr_thermo_mc - arr_wind_mc

        shapiro_vt = stats.shapiro(diff_vt)
        shapiro_vw = stats.shapiro(diff_vw)
        shapiro_tw = stats.shapiro(diff_tw)

        ttest_vt = stats.ttest_rel(arr_vermi_mc, arr_thermo_mc)
        ttest_vw = stats.ttest_rel(arr_vermi_mc, arr_wind_mc)
        ttest_tw = stats.ttest_rel(arr_thermo_mc, arr_wind_mc)

        wilcoxon_vt = stats.wilcoxon(arr_vermi_mc, arr_thermo_mc)
        wilcoxon_vw = stats.wilcoxon(arr_vermi_mc, arr_wind_mc)
        wilcoxon_tw = stats.wilcoxon(arr_thermo_mc, arr_wind_mc)

        comparacao_stats = pd.DataFrame([
            {"Comparação": "Vermi (Yang) vs Termo (Yang)", "p-normalidade": shapiro_vt[1], "p-t pareado": ttest_vt[1], "p-Wilcoxon": wilcoxon_vt[1]},
            {"Comparação": "Vermi (Yang) vs Wind (TOOL13)", "p-normalidade": shapiro_vw[1], "p-t pareado": ttest_vw[1], "p-Wilcoxon": wilcoxon_vw[1]},
            {"Comparação": "Termo (Yang) vs Wind (TOOL13)", "p-normalidade": shapiro_tw[1], "p-t pareado": ttest_tw[1], "p-Wilcoxon": wilcoxon_tw[1]}
        ])
        st.dataframe(comparacao_stats.style.format({c: "{:.5f}" for c in comparacao_stats.columns if c != "Comparação"}))

        # Tabelas anuais formatadas
        st.subheader("📋 Resultados Anuais Detalhados (Cenário Otimista GWP-20)")
        df_anual_fmt = df_anual[['Year', 'Base', 'Vermi', 'Termo', 'Wind', 'Evitado_Vermi', 'Evitado_Termo', 'Evitado_Wind']].copy()
        df_anual_fmt.columns = ['Ano', 'Baseline (tCO₂eq)', 
                                'Vermicompostagem (Yang et al.) (tCO₂eq)', 
                                'Termofílica (Yang et al.) (tCO₂eq)', 
                                'Convencional (TOOL13) (tCO₂eq)',
                                'Evitado Vermi (Yang et al.)', 
                                'Evitado Termo (Yang et al.)', 
                                'Evitado Wind (TOOL13)']
        for col in df_anual_fmt.columns:
            if col != 'Ano':
                df_anual_fmt[col] = df_anual_fmt[col].apply(formatar_br)
        st.dataframe(df_anual_fmt)

    # Reset do estado para permitir nova simulação
    st.session_state.run_simulation = False

else:
    st.info("💡 Ajuste os parâmetros na barra lateral e clique em 'Executar Simulação' para ver os resultados.")

st.markdown("---")
st.markdown("""
**📚 Referências Metodológicas:**

- **Vermicompostagem e Compostagem Termofílica:** Yang et al. (2017) – fatores experimentais (CH₄ = 0,0013 e 0,0060 t/tC; N₂O = 0,0092 e 0,0196 t/tN respectivamente).
- **Compostagem Convencional em Leiras:** TOOL13 (v02.0) – ferramenta da metodologia AMS‑III.F (UNFCCC, 2016). Fatores padrão: CH₄ = 0,002 t/t úmido; N₂O = 0,0005 t/t úmido.
- **Baseline (Aterro Sanitário):** Modelo FOD do IPCC (2006) com calibração para o Aterro CGR Guatapará (Ribeirão Preto) – captura de metano = 60%, φ = 0,85 (UNFCCC 2024).
- **GWP:** Forster et al. (2021) IPCC AR6 (GWP-20, GWP-100, GWP-500).

**⚠️ Nota de Reprodutibilidade:**
- Todas as análises usam seed fixo (50) para garantir resultados reprodutíveis.
- Métodos de cálculo idênticos aos utilizados na validação original.
""")
