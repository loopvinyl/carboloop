import requests
from bs4 import BeautifulSoup
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
from scipy.signal import fftconvolve
from joblib import Parallel, delayed
import warnings
from matplotlib.ticker import FuncFormatter
from SALib.sample.sobol import sample
from SALib.analyze.sobol import analyze
import yfinance as yf   # <-- NOVA IMPORTAÇÃO

np.random.seed(50)

st.set_page_config(page_title="Simulador de Emissões de tCO₂eq e Cálculo de Créditos de Carbono com Análise de Sensibilidade Global", layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# =============================================================================
# Classe GHGEmissionCalculator (modificada para incluir φ e captura)
# =============================================================================
class GHGEmissionCalculator:
    def __init__(self):
        self.TOC = 0.436
        self.TN = 0.0142
        self.f_CH4_vermi = 0.0013
        self.f_N2O_vermi = 0.0092
        self.f_CH4_thermo = 0.0060
        self.f_N2O_thermo = 0.0196
        self.COMPOSTING_DAYS = 50
        self.GWP_CH4_20 = 79.7
        self.GWP_N2O_20 = 273
        self.MCF = 1.0
        self.F = 0.5
        self.OX = 0.1
        self.Ri = 0.0
        self._load_emission_profiles()
        self._setup_pre_disposal_emissions()
    
    def _load_emission_profiles(self):
        self.profile_ch4_vermi = np.array([
            0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06,
            0.07, 0.08, 0.09, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04,
            0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
            0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001
        ])
        self.profile_ch4_vermi /= self.profile_ch4_vermi.sum()
        
        self.profile_n2o_vermi = np.array([
            0.15, 0.10, 0.20, 0.05, 0.03, 0.03, 0.03, 0.04, 0.05, 0.06,
            0.08, 0.09, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02,
            0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
            0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001,
            0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
        ])
        self.profile_n2o_vermi /= self.profile_n2o_vermi.sum()
        
        self.profile_ch4_thermo = self.profile_ch4_vermi.copy()
        
        self.profile_n2o_thermo = np.array([
            0.10, 0.08, 0.15, 0.05, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12,
            0.15, 0.18, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05,
            0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.005, 0.005, 0.005, 0.005, 0.005, 0.002, 0.002, 0.002, 0.002, 0.002,
            0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
        ])
        self.profile_n2o_thermo /= self.profile_n2o_thermo.sum()
        
        self.profile_n2o_landfill = {1: 0.10, 2: 0.30, 3: 0.40, 4: 0.15, 5: 0.05}
    
    def _setup_pre_disposal_emissions(self):
        CH4_pre_ugC_per_kg_h = 2.78
        self.CH4_pre_kg_per_kg_day = CH4_pre_ugC_per_kg_h * (16/12) * 24 / 1_000_000
        
        N2O_pre_mgN_per_kg = 20.26
        N2O_pre_mgN_per_kg_day = N2O_pre_mgN_per_kg / 3
        self.N2O_pre_kg_per_kg_day = N2O_pre_mgN_per_kg_day * (44/28) / 1_000_000
        
        self.profile_n2o_pre = {1: 0.8623, 2: 0.10, 3: 0.0377}
    
    def calculate_landfill_emissions(self, waste_kg_day, k_year, temperature_C,
                                     doc_fraction, moisture_fraction, years=20,
                                     phi=1.0, capture_fraction=0.0):
        """
        Calcula as emissões do aterro usando o método FOD do IPCC.
        
        Parâmetros:
            waste_kg_day (float): resíduos diários (kg/dia)
            k_year (float): taxa de decaimento de primeira ordem (ano⁻¹)
            temperature_C (float): temperatura do aterro (°C)
            doc_fraction (float): fração de carbono orgânico degradável (DOC)
            moisture_fraction (float): fração de umidade do resíduo
            years (int): horizonte de simulação (anos)
            phi (float): fator de correção do modelo (UNFCCC, 2024)
            capture_fraction (float): fração de metano capturada e destruída (default 0,0)
        
        Retorna:
            ch4_emissions (np.array): emissões diárias de CH4 (kg/dia)
            n2o_emissions (np.array): emissões diárias de N2O (kg/dia)
        """
        days = years * 365
        docf = 0.0147 * temperature_C + 0.28
        ch4_potential_per_kg = (doc_fraction * docf * self.MCF * self.F * (16/12) *
                                (1 - self.Ri) * (1 - self.OX))
        ch4_potential_daily = waste_kg_day * ch4_potential_per_kg
        
        t = np.arange(1, days + 1, dtype=float)
        kernel_ch4 = np.exp(-k_year * (t - 1) / 365.0) - np.exp(-k_year * t / 365.0)
        daily_inputs = np.ones(days, dtype=float)
        ch4_emissions = fftconvolve(daily_inputs, kernel_ch4, mode='full')[:days]
        ch4_emissions *= ch4_potential_daily
        
        # Aplicar φ e (1 - captura) APENAS ao CH₄ (UNFCCC, 2024)
        ch4_emissions = ch4_emissions * phi * (1 - capture_fraction)
        
        # Emissões de N2O (Wang et al., 2017)
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
        n2o_emissions = fftconvolve(np.full(days, daily_n2o_kg), kernel_n2o, mode='full')[:days]
        
        ch4_pre, n2o_pre = self._calculate_pre_disposal(waste_kg_day, days)
        
        return ch4_emissions + ch4_pre, n2o_emissions + n2o_pre
    
    def _calculate_pre_disposal(self, waste_kg_day, days):
        ch4_emissions = np.full(days, waste_kg_day * self.CH4_pre_kg_per_kg_day)
        n2o_emissions = np.zeros(days)
        
        for entry_day in range(days):
            for days_after, fraction in self.profile_n2o_pre.items():
                emission_day = entry_day + days_after - 1
                if emission_day < days:
                    n2o_emissions[emission_day] += (waste_kg_day * self.N2O_pre_kg_per_kg_day * fraction)
        
        return ch4_emissions, n2o_emissions
    
    def calculate_vermicomposting_emissions(self, waste_kg_day, moisture_fraction, years=20):
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
    
    def calculate_avoided_emissions(self, waste_kg_day, k_year, temperature_C,
                                    doc_fraction, moisture_fraction, years=20,
                                    phi_baseline=0.85, capture_fraction=0.0):
        """
        Calcula as emissões evitadas para ambas as tecnologias.
        
        phi_baseline: fator de correção do modelo para o baseline (UNFCCC, 2024).
                      Para clima úmido e Aplicação B, o valor recomendado é 0,85.
        capture_fraction: fração de metano capturada e destruída no baseline (default 0,0).
        """
        # Calcular baseline com φ e captura (se houver)
        ch4_landfill, n2o_landfill = self.calculate_landfill_emissions(
            waste_kg_day, k_year, temperature_C, doc_fraction, moisture_fraction, years,
            phi=phi_baseline, capture_fraction=capture_fraction
        )
        
        # Tecnologias (sem φ – apenas para o baseline)
        ch4_vermi, n2o_vermi = self.calculate_vermicomposting_emissions(
            waste_kg_day, moisture_fraction, years
        )
        
        ch4_thermo, n2o_thermo = self.calculate_thermophilic_emissions(
            waste_kg_day, moisture_fraction, years
        )
        
        # Converter para CO2eq (usando GWP-20 como padrão)
        baseline_co2eq = (ch4_landfill * self.GWP_CH4_20 + n2o_landfill * self.GWP_N2O_20) / 1000
        vermi_co2eq = (ch4_vermi * self.GWP_CH4_20 + n2o_vermi * self.GWP_N2O_20) / 1000
        thermo_co2eq = (ch4_thermo * self.GWP_CH4_20 + n2o_thermo * self.GWP_N2O_20) / 1000
        
        avoided_vermi = baseline_co2eq.sum() - vermi_co2eq.sum()
        avoided_thermo = baseline_co2eq.sum() - thermo_co2eq.sum()
        
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
            'thermophilic': {
                'ch4_kg': ch4_thermo.sum(),
                'n2o_kg': n2o_thermo.sum(),
                'co2eq_t': thermo_co2eq.sum(),
                'avoided_co2eq_t': avoided_thermo
            },
            'comparison': {
                'difference_tco2eq': avoided_vermi - avoided_thermo,
                'superiority_percent': ((avoided_vermi / avoided_thermo) - 1) * 100 if avoided_thermo != 0 else 0
            },
            'annual_averages': {
                'baseline_tco2eq_year': baseline_co2eq.sum() / years,
                'vermi_avoided_year': avoided_vermi / years,
                'thermo_avoided_year': avoided_thermo / years
            }
        }
        
        return results

# =============================================================================
# Funções auxiliares (cotações, formatação) - MODIFICADA APENAS A COTAÇÃO DO CARBONO
# =============================================================================
def obter_cotacao_carbono():
    """
    Obtém a cotação do carbono via Yahoo Finance (ticker CO2.L).
    Em caso de falha, retorna valor de referência (85.50 €).
    """
    try:
        ticker = yf.Ticker("CO2.L")
        data = ticker.history(period="1d")
        if not data.empty:
            preco = data['Close'].iloc[-1]
            # Validação básica: preço deve estar entre 10 e 200 €
            if 10 < preco < 200:
                return preco, "€", "Carbon Futures (CO2.L)", True, "Yahoo Finance (CO2.L)"
            else:
                # Preço fora do intervalo esperado, usar fallback
                return 85.50, "€", "Carbon Emissions (Referência)", False, "Referência (valor fora da faixa)"
        else:
            return 85.50, "€", "Carbon Emissions (Referência)", False, "Referência (dados vazios)"
    except Exception as e:
        # Em caso de erro na conexão ou parsing, usa fallback
        return 85.50, "€", "Carbon Emissions (Referência)", False, f"Referência (erro: {str(e)})"

def obter_cotacao_euro_real():
    try:
        url = "https://economia.awesomeapi.com.br/last/EUR-BRL"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = float(data['EURBRL']['bid'])
            return cotacao, "R$", True, "AwesomeAPI"
    except:
        pass
    
    try:
        url = "https://api.exchangerate-api.com/v4/latest/EUR"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = data['rates']['BRL']
            return cotacao, "R$", True, "ExchangeRate-API"
    except:
        pass
    
    return 5.50, "R$", False, "Referência"

def calcular_valor_creditos(emissoes_evitadas_tco2eq, preco_carbono_por_tonelada, moeda, taxa_cambio=1):
    valor_total = emissoes_evitadas_tco2eq * preco_carbono_por_tonelada * taxa_cambio
    return valor_total

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
        
        preco_carbono, moeda, contrato_info, sucesso_carbono, fonte_carbono = obter_cotacao_carbono()
        
        preco_euro, moeda_real, sucesso_euro, fonte_euro = obter_cotacao_euro_real()
        
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        st.session_state.fonte_cotacao = fonte_carbono
        
        st.session_state.mostrar_atualizacao = False
        st.session_state.cotacao_atualizada = False
        
        st.rerun()

    st.sidebar.metric(
        label=f"Preço do Carbono (tCO₂eq)",
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
        label=f"Carbono em Reais (tCO₂eq)",
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
        - Em caso de falha na conexão, são utilizados valores de referência atualizados
        
        **💡 Importante:**
        - Os preços são baseados no mercado regulado da UE
        - Valores em tempo real sujeitos a variações de mercado
        - Conversão para Real utilizando câmbio comercial
        """)

def inicializar_session_state():
    if 'preco_carbono' not in st.session_state:
        preco_carbono, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono()
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.fonte_cotacao = fonte
        
    if 'taxa_cambio' not in st.session_state:
        preco_euro, moeda_real, sucesso_euro, fonte_euro = obter_cotacao_euro_real()
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

st.title("Simulador de Emissões de tCO₂eq e Cálculo de Créditos de Carbono com Análise de Sensibilidade Global")
st.markdown("Esta ferramenta projeta os Créditos de Carbono ao calcular as emissões de gases de efeito estufa para dois contextos de gestão de resíduos")

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
    
    if "0.40" in opcao_k:
        k_ano = 0.40
    else:
        k_ano = 0.06
    
    st.session_state.k_ano = k_ano
    st.write(f"**Valor selecionado:** {formatar_br(k_ano)} ano⁻¹")
    
    st.markdown("**2. Temperatura Média**")
    T = st.slider("Temperatura média (°C)", min_value=20, max_value=40, value=25, step=1)
    st.write(f"**Valor selecionado:** {formatar_br(T)} °C")
    
    st.markdown("**3. Carbono Orgânico Degradável**")
    DOC = st.slider("DOC (fração)", min_value=0.10, max_value=0.25, value=0.15, step=0.01)
    st.write(f"**Valor selecionado:** {formatar_br(DOC)}")
    
    st.markdown("**4. Umidade do Resíduo**")
    umidade_valor = st.slider("Umidade do resíduo (%)", 50, 95, 85, 1)
    umidade = umidade_valor / 100.0
    st.write(f"**Valor fixo:** {formatar_br(umidade_valor)}%")
    
    with st.expander("ℹ️ Sobre os parâmetros da análise Sobol"):
        st.markdown("""
        **📊 Parâmetros variados na análise de sensibilidade Sobol:**
        
        1. **Taxa de decaimento (k):** 0.06 a 0.40 ano⁻¹
           - Controla a velocidade de degradação no aterro
           - Valores mais altos = emissões mais concentradas no início
        
        2. **Temperatura (T):** 20 a 40°C
           - Influencia a taxa de decomposição
           - Temperaturas mais altas aumentam as emissões
        
        3. **Carbono orgânico degradável (DOC):** 0.10 a 0.25
           - Fração do carbono que pode ser degradada
           - Valores mais altos = maior potencial de emissões
        
        **⚙️ Parâmetro fixo (não varia):**
        - **Umidade:** 85% (valor fixo da simulação)
        """)
    
    st.subheader("🎯 Configuração de Simulação")
    anos_simulacao = st.slider("Anos de simulação", 5, 50, 20, 5)
    n_simulations = st.slider("Número de simulações Monte Carlo", 50, 1000, 100, 50)
    n_samples = st.slider("Número de amostras Sobol", 32, 256, 64, 16)
    
    if st.button("🚀 Executar Simulação", type="primary"):
        st.session_state.run_simulation = True

# =============================================================================
# Funções auxiliares para simulação (com suporte a GWP)
# =============================================================================
def compute_results_for_gwp(gwp_ch4, gwp_n2o, waste_kg_day, k_year, temperature_C,
                            doc_fraction, moisture_fraction, years, phi_baseline=0.85):
    """Retorna resultados para um par de GWPs específico."""
    calc = GHGEmissionCalculator()
    calc.GWP_CH4_20 = gwp_ch4
    calc.GWP_N2O_20 = gwp_n2o
    results = calc.calculate_avoided_emissions(
        waste_kg_day, k_year, temperature_C, doc_fraction, moisture_fraction, years,
        phi_baseline=phi_baseline
    )
    return results

def executar_simulacao_vermi_sobol(params_sobol, gwp_ch4, gwp_n2o):
    k_ano_sobol, T_sobol, DOC_sobol = params_sobol
    np.random.seed(50)
    calculator = GHGEmissionCalculator()
    calculator.GWP_CH4_20 = gwp_ch4
    calculator.GWP_N2O_20 = gwp_n2o
    results = calculator.calculate_avoided_emissions(
        waste_kg_day=residuos_kg_dia,
        k_year=k_ano_sobol,
        temperature_C=T_sobol,
        doc_fraction=DOC_sobol,
        moisture_fraction=umidade,
        years=anos_simulacao,
        phi_baseline=0.85
    )
    return results['vermicomposting']['avoided_co2eq_t']

def executar_simulacao_compostagem_sobol(params_sobol, gwp_ch4, gwp_n2o):
    k_ano_sobol, T_sobol, DOC_sobol = params_sobol
    np.random.seed(50)
    calculator = GHGEmissionCalculator()
    calculator.GWP_CH4_20 = gwp_ch4
    calculator.GWP_N2O_20 = gwp_n2o
    results = calculator.calculate_avoided_emissions(
        waste_kg_day=residuos_kg_dia,
        k_year=k_ano_sobol,
        temperature_C=T_sobol,
        doc_fraction=DOC_sobol,
        moisture_fraction=umidade,
        years=anos_simulacao,
        phi_baseline=0.85
    )
    return results['thermophilic']['avoided_co2eq_t']

def gerar_parametros_mc(n):
    np.random.seed(50)
    umidade_vals = np.random.uniform(0.75, 0.90, n)
    temp_vals = np.random.normal(25, 3, n)
    doc_vals = np.random.triangular(0.12, 0.15, 0.18, n)
    return umidade_vals, temp_vals, doc_vals

# =============================================================================
# Execução da simulação principal
# =============================================================================
if st.session_state.get('run_simulation', False):
    with st.spinner('Executando simulação...'):
        # Definir GWPs para os cenários
        gwps = {
            "Otimista (GWP-20)": (79.7, 273),
            "Realista (GWP-100)": (27.0, 273),
            "Pessimista (GWP-500)": (7.2, 130)
        }
        
        # Dicionário para armazenar resultados
        results_all = {}
        
        # Calcular resultados determinísticos para cada GWP
        for nome, (gwp_ch4, gwp_n2o) in gwps.items():
            results_all[nome] = compute_results_for_gwp(
                gwp_ch4, gwp_n2o, residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao
            )
        
        # Usar os resultados do cenário Otimista para os gráficos e tabelas principais
        results = results_all["Otimista (GWP-20)"]
        
        # Gerar dados diários para gráficos (apenas GWP20)
        dias = anos_simulacao * 365
        datas = pd.date_range(start=datetime.now(), periods=dias, freq='D')
        
        # Calcular emissões diárias para plot (usando GWP20)
        calc_g20 = GHGEmissionCalculator()
        calc_g20.GWP_CH4_20, calc_g20.GWP_N2O_20 = gwps["Otimista (GWP-20)"]
        ch4_aterro_dia, n2o_aterro_dia = calc_g20.calculate_landfill_emissions(
            residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao,
            phi=0.85, capture_fraction=0.0
        )
        ch4_vermi_dia, n2o_vermi_dia = calc_g20.calculate_vermicomposting_emissions(
            residuos_kg_dia, umidade, anos_simulacao
        )
        
        df = pd.DataFrame({
            'Data': datas,
            'CH4_Aterro_kg_dia': ch4_aterro_dia,
            'N2O_Aterro_kg_dia': n2o_aterro_dia,
            'CH4_Vermi_kg_dia': ch4_vermi_dia,
            'N2O_Vermi_kg_dia': n2o_vermi_dia,
        })
        
        for gas in ['CH4_Aterro', 'N2O_Aterro', 'CH4_Vermi', 'N2O_Vermi']:
            gwp = calc_g20.GWP_CH4_20 if 'CH4' in gas else calc_g20.GWP_N2O_20
            df[f'{gas}_tCO2eq'] = df[f'{gas}_kg_dia'] * gwp / 1000
        
        df['Total_Aterro_tCO2eq_dia'] = df['CH4_Aterro_tCO2eq'] + df['N2O_Aterro_tCO2eq']
        df['Total_Vermi_tCO2eq_dia'] = df['CH4_Vermi_tCO2eq'] + df['N2O_Vermi_tCO2eq']
        
        df['Total_Aterro_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_dia'].cumsum()
        df['Total_Vermi_tCO2eq_acum'] = df['Total_Vermi_tCO2eq_dia'].cumsum()
        df['Reducao_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_acum'] - df['Total_Vermi_tCO2eq_acum']
        
        df['Year'] = df['Data'].dt.year
        df_anual_revisado = df.groupby('Year').agg({
            'Total_Aterro_tCO2eq_dia': 'sum',
            'Total_Vermi_tCO2eq_dia': 'sum',
        }).reset_index()
        
        df_anual_revisado['Emission reductions (t CO₂eq)'] = df_anual_revisado['Total_Aterro_tCO2eq_dia'] - df_anual_revisado['Total_Vermi_tCO2eq_dia']
        df_anual_revisado['Cumulative reduction (t CO₂eq)'] = df_anual_revisado['Emission reductions (t CO₂eq)'].cumsum()
        
        df_anual_revisado.rename(columns={
            'Total_Aterro_tCO2eq_dia': 'Baseline emissions (t CO₂eq)',
            'Total_Vermi_tCO2eq_dia': 'Vermicomposting emissions (t CO₂eq)',
        }, inplace=True)
        
        # Dados para o cenário de compostagem termofílica com GWP20
        ch4_compost_dia, n2o_compost_dia = calc_g20.calculate_thermophilic_emissions(
            residuos_kg_dia, umidade, anos_simulacao
        )
        ch4_compost_tco2eq = ch4_compost_dia * calc_g20.GWP_CH4_20 / 1000
        n2o_compost_tco2eq = n2o_compost_dia * calc_g20.GWP_N2O_20 / 1000
        total_compost_tco2eq_dia = ch4_compost_tco2eq + n2o_compost_tco2eq
        
        df_compost_dia = pd.DataFrame({
            'Data': datas,
            'Total_Compost_tCO2eq_dia': total_compost_tco2eq_dia
        })
        df_compost_dia['Year'] = df_compost_dia['Data'].dt.year
        
        df_comp_anual_revisado = df_compost_dia.groupby('Year').agg({
            'Total_Compost_tCO2eq_dia': 'sum'
        }).reset_index()
        
        df_comp_anual_revisado = pd.merge(df_comp_anual_revisado,
                                          df_anual_revisado[['Year', 'Baseline emissions (t CO₂eq)']],
                                          on='Year', how='left')
        
        df_comp_anual_revisado['Emission reductions (t CO₂eq)'] = df_comp_anual_revisado['Baseline emissions (t CO₂eq)'] - df_comp_anual_revisado['Total_Compost_tCO2eq_dia']
        df_comp_anual_revisado['Cumulative reduction (t CO₂eq)'] = df_comp_anual_revisado['Emission reductions (t CO₂eq)'].cumsum()
        df_comp_anual_revisado.rename(columns={'Total_Compost_tCO2eq_dia': 'Composting emissions (t CO₂eq)'}, inplace=True)
        
        st.header("📈 Resultados da Simulação")
        
        st.info(f"""
        **Parâmetros utilizados na simulação:**
        - **Taxa de decaimento (k):** {formatar_br(k_ano)} ano⁻¹
        - **Temperatura (T):** {formatar_br(T)} °C
        - **DOC:** {formatar_br(DOC)}
        - **Umidade:** {formatar_br(umidade_valor)}%
        - **Resíduos/dia:** {formatar_br(residuos_kg_dia)} kg
        - **Total de resíduos:** {formatar_br(residuos_kg_dia * 365 * anos_simulacao / 1000)} toneladas
        - **Fator φ (baseline):** 0,85 (UNFCCC 2024 - clima úmido)
        """)
        
        # Tabela comparativa para os três cenários de GWP
        st.subheader("📊 Comparação entre Cenários de GWP")
        comparacao = []
        for nome, res in results_all.items():
            comparacao.append({
                "Cenário": nome,
                "Emissões evitadas (tCO₂eq)": res['vermicomposting']['avoided_co2eq_t'],
                "Média anual (tCO₂eq/ano)": res['vermicomposting']['avoided_co2eq_t'] / anos_simulacao
            })
        df_comp_gwp = pd.DataFrame(comparacao)
        st.dataframe(df_comp_gwp.style.format({
            "Emissões evitadas (tCO₂eq)": lambda x: formatar_br(x),
            "Média anual (tCO₂eq/ano)": lambda x: formatar_br(x)
        }))
        
        # Exibir resultados financeiros para o cenário selecionado (Otimista)
        total_evitado_vermi = results['vermicomposting']['avoided_co2eq_t']
        total_evitado_compost = results['thermophilic']['avoided_co2eq_t']
        
        preco_carbono = st.session_state.preco_carbono
        moeda = st.session_state.moeda_carbono
        taxa_cambio = st.session_state.taxa_cambio
        fonte_cotacao = st.session_state.fonte_cotacao
        
        valor_vermi_eur = calcular_valor_creditos(total_evitado_vermi, preco_carbono, moeda)
        valor_compost_eur = calcular_valor_creditos(total_evitado_compost, preco_carbono, moeda)
        
        valor_vermi_brl = calcular_valor_creditos(total_evitado_vermi, preco_carbono, "R$", taxa_cambio)
        valor_compost_brl = calcular_valor_creditos(total_evitado_compost, preco_carbono, "R$", taxa_cambio)
        
        st.subheader("💰 Valor Financeiro das Emissões Evitadas (Cenário Otimista)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                f"Preço Carbono (Euro)", 
                f"{moeda} {formatar_br(preco_carbono)}/tCO₂eq",
                help=f"Fonte: {fonte_cotacao}"
            )
        with col2:
            st.metric(
                "Vermicompostagem (Euro)", 
                f"{moeda} {formatar_br(valor_vermi_eur)}",
                help=f"Baseado em {formatar_br(total_evitado_vermi)} tCO₂eq evitadas"
            )
        with col3:
            st.metric(
                "Compostagem (Euro)", 
                f"{moeda} {formatar_br(valor_compost_eur)}",
                help=f"Baseado em {formatar_br(total_evitado_compost)} tCO₂eq evitadas"
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                f"Preço Carbono (R$)", 
                f"R$ {formatar_br(preco_carbono * taxa_cambio)}/tCO₂eq",
                help="Preço do carbono convertido para Reais"
            )
        with col2:
            st.metric(
                "Vermicompostagem (R$)", 
                f"R$ {formatar_br(valor_vermi_brl)}",
                help=f"Baseado em {formatar_br(total_evitado_vermi)} tCO₂eq evitadas"
            )
        with col3:
            st.metric(
                "Compostagem (R$)", 
                f"R$ {formatar_br(valor_compost_brl)}",
                help=f"Baseado em {formatar_br(total_evitado_compost)} tCO₂eq evitadas"
            )
        
        with st.expander("💡 Como funciona a comercialização no mercado de carbono?"):
            st.markdown(f"""
            **📊 Informações de Mercado:**
            - **Preço em Euro:** {moeda} {formatar_br(preco_carbono)}/tCO₂eq
            - **Preço em Real:** R$ {formatar_br(preco_carbono * taxa_cambio)}/tCO₂eq
            - **Taxa de câmbio:** 1 Euro = R$ {formatar_br(taxa_cambio)}
            - **Fonte:** {fonte_cotacao}
            
            **💶 Comprar créditos (compensação):**
            - Custo em Euro: **{moeda} {formatar_br(valor_vermi_eur)}**
            - Custo em Real: **R$ {formatar_br(valor_vermi_brl)}**
            
            **💵 Vender créditos (comercialização):**  
            - Receita em Euro: **{moeda} {formatar_br(valor_vermi_eur)}**
            - Receita em Real: **R$ {formatar_br(valor_vermi_brl)}**
            
            **🌍 Mercado de Referência:**
            - European Union Allowances (EUA)
            - European Emissions Trading System (EU ETS)
            - Contratos futuros de carbono
            - Preços em tempo real do mercado regulado
            """)
        
        st.subheader("📊 Resumo das Emissões Evitadas (Cenário Otimista)")
        
        media_anual_vermi = total_evitado_vermi / anos_simulacao
        media_anual_compost = total_evitado_compost / anos_simulacao
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📋 Vermicompostagem")
            st.metric(
                "Total de emissões evitadas", 
                f"{formatar_br(total_evitado_vermi)} tCO₂eq",
                help=f"Total acumulado em {anos_simulacao} anos"
            )
            st.metric(
                "Média anual", 
                f"{formatar_br(media_anual_vermi)} tCO₂eq/ano",
                help=f"Emissões evitadas por ano em média"
            )

        with col2:
            st.markdown("#### 📋 Compostagem")
            st.metric(
                "Total de emissões evitadas", 
                f"{formatar_br(total_evitado_compost)} tCO₂eq",
                help=f"Total acumulado em {anos_simulacao} anos"
            )
            st.metric(
                "Média anual", 
                f"{formatar_br(media_anual_compost)} tCO₂eq/ano",
                help=f"Emissões evitadas por ano em média"
            )

        st.subheader("📊 Comparação Anual das Emissões Evitadas (Cenário Otimista)")
        df_evitadas_anual = pd.DataFrame({
            'Year': df_anual_revisado['Year'],
            'Vermicompostagem': df_anual_revisado['Emission reductions (t CO₂eq)'],
            'Compostagem': df_comp_anual_revisado['Emission reductions (t CO₂eq)']
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        br_formatter = FuncFormatter(br_format)
        x = np.arange(len(df_evitadas_anual['Year']))
        bar_width = 0.35

        ax.bar(x - bar_width/2, df_evitadas_anual['Vermicompostagem'], width=bar_width,
                label='Vermicompostagem', edgecolor='black')
        ax.bar(x + bar_width/2, df_evitadas_anual['Compostagem'], width=bar_width,
                label='Compostagem', edgecolor='black', hatch='//')

        for i, (v1, v2) in enumerate(zip(df_evitadas_anual['Vermicompostagem'], 
                                         df_evitadas_anual['Compostagem'])):
            ax.text(i - bar_width/2, v1 + max(v1, v2)*0.01, 
                    formatar_br(v1), ha='center', fontsize=9, fontweight='bold')
            ax.text(i + bar_width/2, v2 + max(v1, v2)*0.01, 
                    formatar_br(v2), ha='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Ano')
        ax.set_ylabel('Emissões Evitadas (t CO₂eq)')
        ax.set_title('Comparação Anual das Emissões Evitadas: Vermicompostagem vs Compostagem')
        
        ax.set_xticks(x)
        ax.set_xticklabels(df_anual_revisado['Year'], fontsize=8)

        ax.legend(title='Tecnologia')
        ax.yaxis.set_major_formatter(br_formatter)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        st.subheader("📉 Redução de Emissões Acumulada (Cenário Otimista)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Data'], df['Total_Aterro_tCO2eq_acum'], 'r-', label='Cenário Base (Aterro Sanitário)', linewidth=2)
        ax.plot(df['Data'], df['Total_Vermi_tCO2eq_acum'], 'g-', label='Vermicompostagem', linewidth=2)
        ax.fill_between(df['Data'], df['Total_Vermi_tCO2eq_acum'], df['Total_Aterro_tCO2eq_acum'],
                        color='skyblue', alpha=0.5, label='Emissões Evitadas')
        ax.set_title('Redução de Emissões em {} Anos (k = {} ano⁻¹)'.format(anos_simulacao, formatar_br(k_ano)))
        ax.set_xlabel('Ano')
        ax.set_ylabel('tCO₂eq Acumulado')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.set_major_formatter(br_formatter)

        st.pyplot(fig)

        # =============================================================================
        # Análise de Sensibilidade Sobol (apenas GWP20)
        # =============================================================================
        st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - Vermicompostagem (GWP-20)")
        st.info("**Parâmetros variados na análise:** Taxa de Decaimento (k), Temperatura (T), DOC")
        br_formatter_sobol = FuncFormatter(br_format)

        np.random.seed(50)  
        
        problem = {
            'num_vars': 3,
            'names': ['taxa_decaimento', 'T', 'DOC'],
            'bounds': [
                [0.06, 0.40],
                [20.0, 40.0],
                [0.10, 0.25],
            ]
        }

        param_values = sample(problem, n_samples, seed=50)
        # Usar GWP20 para Sobol
        gwp20_ch4, gwp20_n2o = gwps["Otimista (GWP-20)"]
        results_vermi = Parallel(n_jobs=-1)(delayed(executar_simulacao_vermi_sobol)(params, gwp20_ch4, gwp20_n2o) for params in param_values)
        Si_vermi = analyze(problem, np.array(results_vermi), print_to_console=False)
        
        sensibilidade_df_vermi = pd.DataFrame({
            'Parâmetro': problem['names'],
            'S1': Si_vermi['S1'],
            'ST': Si_vermi['ST']
        }).sort_values('ST', ascending=False)

        nomes_amigaveis = {
            'taxa_decaimento': 'Taxa de Decaimento (k)',
            'T': 'Temperatura',
            'DOC': 'Carbono Orgânico Degradável'
        }
        sensibilidade_df_vermi['Parâmetro'] = sensibilidade_df_vermi['Parâmetro'].map(nomes_amigaveis)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ST', y='Parâmetro', data=sensibilidade_df_vermi, palette='viridis', ax=ax)
        ax.set_title('Sensibilidade Global - Vermicompostagem (GWP-20)')
        ax.set_xlabel('Índice ST (Sobol Total)')
        ax.set_ylabel('Parâmetro')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.xaxis.set_major_formatter(br_formatter_sobol)
        
        for i, (st_value) in enumerate(sensibilidade_df_vermi['ST']):
            ax.text(st_value, i, f' {formatar_br(st_value)}', va='center', fontweight='bold')
        
        st.pyplot(fig)
        
        st.subheader("📊 Valores de Sensibilidade - Vermicompostagem (GWP-20)")
        st.dataframe(sensibilidade_df_vermi.style.format({
            'S1': '{:.4f}',
            'ST': '{:.4f}'
        }))

        st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - Compostagem (GWP-20)")
        st.info("**Parâmetros variados na análise:** Taxa de Decaimento (k), Temperatura (T), DOC")

        np.random.seed(50)
        
        results_compost = Parallel(n_jobs=-1)(delayed(executar_simulacao_compostagem_sobol)(params, gwp20_ch4, gwp20_n2o) for params in param_values)
        Si_compost = analyze(problem, np.array(results_compost), print_to_console=False)
        
        sensibilidade_df_compost = pd.DataFrame({
            'Parâmetro': problem['names'],
            'S1': Si_compost['S1'],
            'ST': Si_compost['ST']
        }).sort_values('ST', ascending=False)

        sensibilidade_df_compost['Parâmetro'] = sensibilidade_df_compost['Parâmetro'].map(nomes_amigaveis)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ST', y='Parâmetro', data=sensibilidade_df_compost, palette='viridis', ax=ax)
        ax.set_title('Sensibilidade Global - Compostagem (GWP-20)')
        ax.set_xlabel('Índice ST (Sobol Total)')
        ax.set_ylabel('Parâmetro')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.xaxis.set_major_formatter(br_formatter_sobol)
        
        for i, (st_value) in enumerate(sensibilidade_df_compost['ST']):
            ax.text(st_value, i, f' {formatar_br(st_value)}', va='center', fontweight='bold')
        
        st.pyplot(fig)
        
        st.subheader("📊 Valores de Sensibilidade - Compostagem (GWP-20)")
        st.dataframe(sensibilidade_df_compost.style.format({
            'S1': '{:.4f}',
            'ST': '{:.4f}'
        }))

        # =============================================================================
        # Monte Carlo para todos os cenários de GWP
        # =============================================================================
        st.subheader("🎲 Análise de Incerteza (Monte Carlo) - Comparação entre Cenários de GWP")
        
        umidade_vals, temp_vals, doc_vals = gerar_parametros_mc(n_simulations)
        
        # Dicionário para armazenar resultados de Monte Carlo para ambas as tecnologias
        mc_results = {}
        for nome, (gwp_ch4, gwp_n2o) in gwps.items():
            vermi_arr = []
            thermo_arr = []
            for i in range(n_simulations):
                calculator_mc = GHGEmissionCalculator()
                calculator_mc.GWP_CH4_20 = gwp_ch4
                calculator_mc.GWP_N2O_20 = gwp_n2o
                res = calculator_mc.calculate_avoided_emissions(
                    waste_kg_day=residuos_kg_dia,
                    k_year=k_ano,
                    temperature_C=temp_vals[i],
                    doc_fraction=doc_vals[i],
                    moisture_fraction=umidade_vals[i],
                    years=anos_simulacao,
                    phi_baseline=0.85
                )
                vermi_arr.append(res['vermicomposting']['avoided_co2eq_t'])
                thermo_arr.append(res['thermophilic']['avoided_co2eq_t'])
            mc_results[nome] = {
                'vermicomposting': np.array(vermi_arr),
                'composting': np.array(thermo_arr)  # renomeado internamente para facilitar
            }
        
        # Plotar distribuições para vermicompostagem
        fig, ax = plt.subplots(figsize=(12, 6))
        for nome, arr_dict in mc_results.items():
            sns.kdeplot(arr_dict['vermicomposting'], label=nome, ax=ax, linewidth=2)
        ax.set_title('Distribuição das Emissões Evitadas (Vermicompostagem)')
        ax.set_xlabel('Emissões Evitadas (tCO₂eq)')
        ax.set_ylabel('Densidade')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(br_format)
        st.pyplot(fig)
        
        # Tabela de estatísticas para vermicompostagem
        stats_list = []
        for nome, arr_dict in mc_results.items():
            arr = arr_dict['vermicomposting']
            stats_list.append({
                "Cenário": nome,
                "Média (tCO₂eq)": np.mean(arr),
                "Mediana (tCO₂eq)": np.median(arr),
                "Desvio Padrão": np.std(arr),
                "IC 95% Inferior": np.percentile(arr, 2.5),
                "IC 95% Superior": np.percentile(arr, 97.5)
            })
        df_mc_stats = pd.DataFrame(stats_list)
        st.subheader("📊 Estatísticas do Monte Carlo - Vermicompostagem")
        st.dataframe(df_mc_stats.style.format({
            "Média (tCO₂eq)": lambda x: formatar_br(x),
            "Mediana (tCO₂eq)": lambda x: formatar_br(x),
            "Desvio Padrão": lambda x: formatar_br(x),
            "IC 95% Inferior": lambda x: formatar_br(x),
            "IC 95% Superior": lambda x: formatar_br(x)
        }))
        
        # =============================================================================
        # ANÁLISE ESTATÍSTICA: Comparação Vermicompostagem vs Compostagem
        # =============================================================================
        st.subheader("📊 Análise Estatística de Comparação (Vermicompostagem vs Compostagem)")
        
        # Escolher um cenário de GWP para a comparação (aqui usamos o Otimista - GWP-20)
        gwp_alvo = "Otimista (GWP-20)"
        vermi_arr = mc_results[gwp_alvo]['vermicomposting']
        compost_arr = mc_results[gwp_alvo]['composting']
        
        diferenças = vermi_arr - compost_arr
        
        # Teste de normalidade (Shapiro-Wilk)
        shapiro_stat, shapiro_p = stats.shapiro(diferenças)
        st.write(f"**Teste de normalidade das diferenças (Shapiro‑Wilk):** Estatística = {shapiro_stat:.5f}, p‑valor = {shapiro_p:.5f}")
        
        # Teste t pareado
        t_stat, t_p = stats.ttest_rel(vermi_arr, compost_arr)
        st.write(f"**Teste t pareado:** Estatística t = {t_stat:.5f}, p‑valor = {t_p:.5f}")
        
        # Teste de Wilcoxon (pareado)
        w_stat, w_p = stats.wilcoxon(vermi_arr, compost_arr)
        st.write(f"**Teste de Wilcoxon (pareado):** Estatística = {w_stat:.5f}, p‑valor = {w_p:.5f}")
        
        # Tabela resumo para todos os cenários de GWP
        st.markdown("#### Comparação em todos os cenários de GWP")
        comparacao_stats = []
        for nome in gwps.keys():
            vermi = mc_results[nome]['vermicomposting']
            compost = mc_results[nome]['composting']
            diff = vermi - compost
            shapiro_p = stats.shapiro(diff)[1]
            t_stat, t_p = stats.ttest_rel(vermi, compost)
            w_stat, w_p = stats.wilcoxon(vermi, compost)
            comparacao_stats.append({
                "Cenário GWP": nome,
                "Diferença média (tCO₂eq)": np.mean(diff),
                "p‑normalidade": shapiro_p,
                "p‑t pareado": t_p,
                "p‑Wilcoxon": w_p
            })
        df_comp_stats = pd.DataFrame(comparacao_stats)
        st.dataframe(df_comp_stats.style.format({
            "Diferença média (tCO₂eq)": lambda x: formatar_br(x),
            "p‑normalidade": lambda x: f"{x:.5f}",
            "p‑t pareado": lambda x: f"{x:.5f}",
            "p‑Wilcoxon": lambda x: f"{x:.5f}"
        }))
        
        st.subheader("📋 Resultados Anuais - Vermicompostagem (Cenário Otimista)")

        df_anual_formatado = df_anual_revisado.copy()
        for col in df_anual_formatado.columns:
            if col != 'Year':
                df_anual_formatado[col] = df_anual_formatado[col].apply(formatar_br)

        st.dataframe(df_anual_formatado)

        st.subheader("📋 Resultados Anuais - Compostagem (Cenário Otimista)")

        df_comp_formatado = df_comp_anual_revisado.copy()
        for col in df_comp_formatado.columns:
            if col != 'Year':
                df_comp_formatado[col] = df_comp_formatado[col].apply(formatar_br)

        st.dataframe(df_comp_formatado)

else:
    st.info("💡 Ajuste os parâmetros na barra lateral e clique em 'Executar Simulação' para ver os resultados.")

st.markdown("---")
st.markdown("""
**📚 Referências por Cenário:**

**Cenário de Baseline (Aterro Sanitário):**
- Metano: IPCC (2006), UNFCCC (2016) e Wang et al. (2023) 
- Óxido Nitroso: Wang et al. (2017)
- Metano e Óxido Nitroso no pré-descarte: Feng et al. (2020)
- **Fator φ = 0,85 (UNFCCC, 2024) aplicado ao baseline para clima úmido**

**Vermicompostagem (tecnologia proposta):**
- Metano e Óxido Nitroso: Yang et al. (2017)

**Compostagem termofílica (tecnologia comparativa):**
- Protocolo AMS-III.F: UNFCCC (2016)
- Fatores de emissões: Yang et al. (2017)

**Cenários de Potencial de Aquecimento Global (GWP):**
- **Otimista (GWP-20):** CH₄ = 79,7; N₂O = 273 (Forster et al., 2021)
- **Realista (GWP-100):** CH₄ = 27,0; N₂O = 273 (Forster et al., 2021)
- **Pessimista (GWP-500):** CH₄ = 7,2; N₂O = 130 (Forster et al., 2021)

**⚠️ Nota de Reprodutibilidade:**
- Todas as análises usam seed fixo (50) para garantir resultados idênticos ao script testado no Google Colab
- Ajustados os ranges de parâmetros para DOC (0.10-0.25) e Temperatura (20-40°C)
- Métodos de cálculo idênticos aos do script testado no Google Colab
""")
