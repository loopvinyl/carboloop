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
    page_title="Simulador de Emissões de tCO₂eq e Cálculo de Créditos de Carbono com Análise de Sensibilidade Global",
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
# PARÂMETROS ESPECÍFICOS PARA O PROJETO EM RIBEIRÃO PRETO
# =============================================================================
# Aterro CGR Guatapará (destino dos RSU):
#   - É um aterro sanitário com usina de biogás desde 2014.
#   - Possui captura de metano para geração de energia elétrica.
#   - Conforme a ferramenta A6.4-AMT-003 (v01.0):
#        * MCF = 1,0 (Tabela 8 – anaerobic managed SWDS)
#        * Fator de captura (f_y) = 0,6 (60% do metano gerado é capturado e destruído/utilizado)
#   - Fator φ (model correction) para baseline em clima úmido: 0,85 (Tabela 5, Application B)
CAPTURE_FRACTION_BASELINE = 0.6    # 60% de captura (realidade de Ribeirão Preto)
MCF_BASELINE = 1.0                 # Aterro sanitário anaeróbio gerenciado
OX_BASELINE = 0.1                  # Fator de oxidação para SWDS sem cobertura (não-LDC)
PHI_BASELINE = 0.85                # Clima úmido (Application B)

# Fatores de emissão padrão para compostagem em leiras (TOOL13, v02.0, seção 6.3)
EF_CH4_COMPOST_DEFAULT = 0.002     # t CH4 / t resíduo úmido
EF_N2O_COMPOST_DEFAULT = 0.0005    # t N2O / t resíduo úmido

# CLASSE PARA CÁLCULO DE EMISSÕES DE GEE

class GHGEmissionCalculator:
    """
    Calcula emissões de CH₄ e N₂O para:
    - Aterro sanitário (baseline, método FOD do IPCC) – calibrado para Ribeirão Preto.
    - Compostagem convencional em leiras (windrow composting) – conforme TOOL13.

    Referências normativas:
    - Baseline: A6.4-AMT-003 (v01.0) "Emissions from solid waste disposal sites"
    - Compostagem: TOOL13 (v02.0) "Project and leakage emissions from composting"
    - Metodologia geral: AMS-III.F (v12.0) "Avoidance of methane emissions through composting"
    """

    def __init__(self):
        # Parâmetros do baseline (aterro) – valores fixos para Ribeirão Preto
        self.MCF = MCF_BASELINE               # A6.4-AMT-003, Tabela 8
        self.F = 0.5                          # A6.4-AMT-003, Tabela 3 (fração de metano no biogás)
        self.OX = OX_BASELINE                 # A6.4-AMT-003, Tabela 6 (sem cobertura, não-LDC)
        self.Ri = 0.0                         # Fração recuperada (default 0)
        
        # Fatores de emissão padrão para compostagem em leiras (TOOL13, v02.0)
        self.EF_CH4_default = EF_CH4_COMPOST_DEFAULT
        self.EF_N2O_default = EF_N2O_COMPOST_DEFAULT
        
        # Duração típica do processo de compostagem em leiras (dias)
        self.COMPOSTING_DAYS = 50
        
        # Potenciais de aquecimento global (GWP-20) – Forster et al. 2021
        self.GWP_CH4_20 = 79.7
        self.GWP_N2O_20 = 273
        
        # Carrega perfis temporais de emissões (apenas para distribuição diária)
        self._load_emission_profiles()
        self._setup_pre_disposal_emissions()

    def _load_emission_profiles(self):
        """
        Perfis temporais diários de emissões (fração por dia).
        Estes perfis são baseados na literatura e usados apenas para distribuir
        as emissões totais ao longo do tempo. Não afetam o total de emissões.
        """
        # Perfil de CH4 para compostagem em leiras
        self.profile_ch4_compost = np.array([
            0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06,
            0.07, 0.08, 0.09, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04,
            0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
            0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001
        ])
        self.profile_ch4_compost /= self.profile_ch4_compost.sum()

        # Perfil de N2O para compostagem em leiras
        self.profile_n2o_compost = np.array([
            0.10, 0.08, 0.15, 0.05, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12,
            0.15, 0.18, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05,
            0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.005, 0.005, 0.005, 0.005, 0.005, 0.002, 0.002, 0.002, 0.002, 0.002,
            0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
        ])
        self.profile_n2o_compost /= self.profile_n2o_compost.sum()

        # Perfil de N2O para aterro (Wang et al. 2017)
        self.profile_n2o_landfill = {1: 0.10, 2: 0.30, 3: 0.40, 4: 0.15, 5: 0.05}

    def _setup_pre_disposal_emissions(self):
        """Emissões na fase de pré-descarte (antes do tratamento)."""
        # CH₄: taxa constante diária (2,78 μg C kg⁻¹ h⁻¹ → kg CH₄ kg⁻¹ dia⁻¹)
        CH4_pre_ugC_per_kg_h = 2.78
        self.CH4_pre_kg_per_kg_day = CH4_pre_ugC_per_kg_h * (16/12) * 24 / 1_000_000_000

        # N₂O: emissão total nos 3 dias (20,26 mg N kg⁻¹, segundo Feng et al. 2020)
        N2O_pre_mgN_per_kg_total = 20.26
        self.N2O_pre_kg_per_kg_total = N2O_pre_mgN_per_kg_total * (44/28) / 1_000_000
        self.profile_n2o_pre = {1: 0.8623, 2: 0.10, 3: 0.0377}

    def calculate_landfill_emissions(self, waste_kg_day, k_year, temperature_C,
                                     doc_fraction, moisture_fraction, years=20,
                                     phi=PHI_BASELINE, capture_fraction=CAPTURE_FRACTION_BASELINE):
        """
        Emissões do aterro sanitário (método FOD do IPCC).
        Conforme a ferramenta A6.4-AMT-003 (v01.0) "Emissions from solid waste disposal sites".
        - Equação (1) ou (2) para o modelo FOD.
        - Parâmetros: MCF, F, OX, phi, k, DOC_f seguem as tabelas da ferramenta.
        - Fator phi = 0,85 para baseline em Application B (clima úmido) – Tabela 5.
        - Fator OX = 0,1 para SWDS sem cobertura em não-LDCs – Tabela 6.
        - capture_fraction = 0,6 representa a fração de metano capturada e destruída/utilizada no aterro CGR Guatapará.
        """
        days = years * 365
        # Fração DOC que realmente se decompõe (IPCC)
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

        ch4_pre, n2o_pre = self._calculate_pre_disposal(waste_kg_day, days)
        return ch4_emissions + ch4_pre, n2o_emissions + n2o_pre

    def _calculate_pre_disposal(self, waste_kg_day, days):
        """Emissões diárias durante o pré-descarte (antes do tratamento)."""
        ch4_emissions = np.full(days, waste_kg_day * self.CH4_pre_kg_per_kg_day)
        n2o_emissions = np.zeros(days)
        for entry_day in range(days):
            for days_after, fraction in self.profile_n2o_pre.items():
                emission_day = entry_day + days_after - 1
                if emission_day < days:
                    n2o_emissions[emission_day] += (waste_kg_day * self.N2O_pre_kg_per_kg_total * fraction)
        return ch4_emissions, n2o_emissions

    def calculate_composting_emissions(self, waste_kg_day, moisture_fraction, years=20):
        """
        Emissões da compostagem convencional em leiras (windrow).
        Cálculo baseado no TOOL13, versão 02.0 (UNFCCC, 2017):
        - Seção 6.1.4: PE_CH4,y = Q_y × EF_CH4,y × GWP_CH4
        - Seção 6.1.5: PE_N2O,y = Q_y × EF_N2O,y × GWP_N2O
        - Valores padrão (default) fornecidos na seção 6.3:
            EF_CH4_default = 0,002 t CH4/t resíduo úmido
            EF_N2O_default = 0,0005 t N2O/t resíduo úmido
        A distribuição temporal diária utiliza perfis típicos da literatura,
        mas não altera o total de emissões.
        """
        days = years * 365
        total_waste_kg = waste_kg_day * days
        total_waste_t = total_waste_kg / 1000.0

        # Emissões totais (toneladas) conforme TOOL13
        total_ch4_t = total_waste_t * self.EF_CH4_default
        total_n2o_t = total_waste_t * self.EF_N2O_default

        # Emissões por kg de resíduo (kg gás / kg resíduo)
        ch4_per_kg = self.EF_CH4_default / 1000.0   # kg CH4 / kg resíduo
        n2o_per_kg = self.EF_N2O_default / 1000.0   # kg N2O / kg resíduo

        ch4_per_batch_kg = waste_kg_day * ch4_per_kg
        n2o_per_batch_kg = waste_kg_day * n2o_per_kg

        ch4_emissions = np.zeros(days)
        n2o_emissions = np.zeros(days)

        # Distribuição diária usando perfis normalizados
        for entry_day in range(days):
            for compost_day in range(self.COMPOSTING_DAYS):
                emission_day = entry_day + compost_day
                if emission_day < days:
                    ch4_emissions[emission_day] += ch4_per_batch_kg * self.profile_ch4_compost[compost_day]
                    n2o_emissions[emission_day] += n2o_per_batch_kg * self.profile_n2o_compost[compost_day]

        return ch4_emissions, n2o_emissions

    def calculate_avoided_emissions(self, waste_kg_day, k_year, temperature_C,
                                    doc_fraction, moisture_fraction, years=20,
                                    phi_baseline=PHI_BASELINE, capture_fraction=CAPTURE_FRACTION_BASELINE):
        """
        Calcula emissões evitadas (tCO₂eq) pela compostagem em leiras em relação ao aterro.
        """
        ch4_landfill, n2o_landfill = self.calculate_landfill_emissions(
            waste_kg_day, k_year, temperature_C, doc_fraction, moisture_fraction, years,
            phi=phi_baseline, capture_fraction=capture_fraction
        )
        ch4_compost, n2o_compost = self.calculate_composting_emissions(waste_kg_day, moisture_fraction, years)

        baseline_co2eq = (ch4_landfill * self.GWP_CH4_20 + n2o_landfill * self.GWP_N2O_20) / 1000
        compost_co2eq = (ch4_compost * self.GWP_CH4_20 + n2o_compost * self.GWP_N2O_20) / 1000

        avoided_compost = baseline_co2eq.sum() - compost_co2eq.sum()

        results = {
            'baseline': {
                'ch4_kg': ch4_landfill.sum(),
                'n2o_kg': n2o_landfill.sum(),
                'co2eq_t': baseline_co2eq.sum()
            },
            'composting': {
                'ch4_kg': ch4_compost.sum(),
                'n2o_kg': n2o_compost.sum(),
                'co2eq_t': compost_co2eq.sum(),
                'avoided_co2eq_t': avoided_compost
            },
            'annual_averages': {
                'baseline_tco2eq_year': baseline_co2eq.sum() / years,
                'compost_avoided_year': avoided_compost / years
            }
        }
        return results


# FUNÇÕES DE COTAÇÃO (MERCADO DE CARBONO E CÂMBIO)

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


# FUNÇÕES AUXILIARES DE FORMATAÇÃO BRASILEIRA

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


# INTERFACE STREAMLIT – BARRA LATERAL E EXIBIÇÃO DE COTAÇÕES

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


# INTERFACE PRINCIPAL E PARÂMETROS DE ENTRADA

st.title("Simulador de Emissões de tCO₂eq e Cálculo de Créditos de Carbono com Análise de Sensibilidade Global")
st.markdown("Esta ferramenta projeta os Créditos de Carbono ao calcular as emissões de gases de efeito estufa para o contexto de gestão de resíduos: **compostagem convencional em leiras** (windrow) em comparação com aterro sanitário (calibrado para a realidade de Ribeirão Preto – aterro CGR Guatapará com captura de biogás).")

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
        """)

    st.subheader("🎯 Configuração de Simulação")
    anos_simulacao = st.slider("Anos de simulação", 5, 50, 20, 5)
    n_simulations = st.slider("Número de simulações Monte Carlo", 50, 1000, 100, 50)
    n_samples = st.slider("Número de amostras Sobol", 32, 256, 64, 16)

    # Botão que aciona a simulação
    if st.button("🚀 Executar Simulação", type="primary"):
        st.session_state.run_simulation = True


# FUNÇÕES AUXILIARES PARA SIMULAÇÃO (GWP, SOBOL, MONTE CARLO)

def compute_results_for_gwp(gwp_ch4, gwp_n2o, waste_kg_day, k_year, temperature_C,
                            doc_fraction, moisture_fraction, years, 
                            phi_baseline=PHI_BASELINE, capture_fraction=CAPTURE_FRACTION_BASELINE):
    """Executa cálculo de emissões evitadas com valores específicos de GWP."""
    calc = GHGEmissionCalculator()
    calc.GWP_CH4_20 = gwp_ch4
    calc.GWP_N2O_20 = gwp_n2o
    return calc.calculate_avoided_emissions(
        waste_kg_day, k_year, temperature_C, doc_fraction, moisture_fraction, years,
        phi_baseline=phi_baseline, capture_fraction=capture_fraction
    )

def executar_simulacao_compostagem_sobol(params_sobol, gwp_ch4, gwp_n2o):
    """Função auxiliar para paralelização da análise Sobol – compostagem em leiras."""
    k_ano_sobol, T_sobol, DOC_sobol = params_sobol
    np.random.seed(50)
    calc = GHGEmissionCalculator()
    calc.GWP_CH4_20 = gwp_ch4
    calc.GWP_N2O_20 = gwp_n2o
    res = calc.calculate_avoided_emissions(
        waste_kg_day=residuos_kg_dia,
        k_year=k_ano_sobol,
        temperature_C=T_sobol,
        doc_fraction=DOC_sobol,
        moisture_fraction=umidade,
        years=anos_simulacao,
        phi_baseline=PHI_BASELINE,
        capture_fraction=CAPTURE_FRACTION_BASELINE
    )
    return res['composting']['avoided_co2eq_t']

def gerar_parametros_mc(n):
    """Gera parâmetros aleatórios para simulação Monte Carlo."""
    np.random.seed(50)
    umidade_vals = np.random.uniform(0.75, 0.90, n)
    temp_vals = np.random.normal(25, 3, n)
    doc_vals = np.random.triangular(0.12, 0.15, 0.18, n)
    return umidade_vals, temp_vals, doc_vals


# EXECUÇÃO DA SIMULAÇÃO (QUANDO BOTÃO FOR CLICADO)

if st.session_state.get('run_simulation', False):
    with st.spinner('Executando simulação...'):
        # Cenários de GWP
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

        # Resultados do cenário Otimista (GWP-20) para gráficos e tabelas principais
        results = results_all["Otimista (GWP-20)"]

        # --- Geração de dados diários para gráficos (apenas GWP-20) ---
        dias = anos_simulacao * 365
        datas = pd.date_range(start=datetime.now(), periods=dias, freq='D')

        calc_g20 = GHGEmissionCalculator()
        calc_g20.GWP_CH4_20, calc_g20.GWP_N2O_20 = gwps["Otimista (GWP-20)"]
        ch4_aterro_dia, n2o_aterro_dia = calc_g20.calculate_landfill_emissions(
            residuos_kg_dia, k_ano, T, DOC, umidade, anos_simulacao,
            phi=PHI_BASELINE, capture_fraction=CAPTURE_FRACTION_BASELINE
        )
        ch4_compost_dia, n2o_compost_dia = calc_g20.calculate_composting_emissions(
            residuos_kg_dia, umidade, anos_simulacao
        )

        df = pd.DataFrame({
            'Data': datas,
            'CH4_Aterro_kg_dia': ch4_aterro_dia,
            'N2O_Aterro_kg_dia': n2o_aterro_dia,
            'CH4_Compost_kg_dia': ch4_compost_dia,
            'N2O_Compost_kg_dia': n2o_compost_dia,
        })

        for gas in ['CH4_Aterro', 'N2O_Aterro', 'CH4_Compost', 'N2O_Compost']:
            gwp = calc_g20.GWP_CH4_20 if 'CH4' in gas else calc_g20.GWP_N2O_20
            df[f'{gas}_tCO2eq'] = df[f'{gas}_kg_dia'] * gwp / 1000

        df['Total_Aterro_tCO2eq_dia'] = df['CH4_Aterro_tCO2eq'] + df['N2O_Aterro_tCO2eq']
        df['Total_Compost_tCO2eq_dia'] = df['CH4_Compost_tCO2eq'] + df['N2O_Compost_tCO2eq']
        df['Total_Aterro_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_dia'].cumsum()
        df['Total_Compost_tCO2eq_acum'] = df['Total_Compost_tCO2eq_dia'].cumsum()
        df['Reducao_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_acum'] - df['Total_Compost_tCO2eq_acum']
        df['Year'] = df['Data'].dt.year

        # Agregação anual – compostagem
        df_anual_revisado = df.groupby('Year').agg({
            'Total_Aterro_tCO2eq_dia': 'sum',
            'Total_Compost_tCO2eq_dia': 'sum',
        }).reset_index()
        df_anual_revisado['Emission reductions (t CO₂eq)'] = df_anual_revisado['Total_Aterro_tCO2eq_dia'] - df_anual_revisado['Total_Compost_tCO2eq_dia']
        df_anual_revisado['Cumulative reduction (t CO₂eq)'] = df_anual_revisado['Emission reductions (t CO₂eq)'].cumsum()
        df_anual_revisado.rename(columns={
            'Total_Aterro_tCO2eq_dia': 'Baseline emissions (t CO₂eq)',
            'Total_Compost_tCO2eq_dia': 'Composting emissions (t CO₂eq)',
        }, inplace=True)

        # --- EXIBIÇÃO DE RESULTADOS ---
        st.header("📈 Resultados da Simulação")
        st.info(f"""
        **Parâmetros utilizados na simulação (calibrados para Ribeirão Preto):**
        - Taxa de decaimento (k): {formatar_br(k_ano)} ano⁻¹
        - Temperatura (T): {formatar_br(T)} °C
        - DOC: {formatar_br(DOC)}
        - Umidade: {formatar_br(umidade_valor)}%
        - Resíduos/dia: {formatar_br(residuos_kg_dia)} kg
        - Total de resíduos: {formatar_br(residuos_kg_dia * 365 * anos_simulacao / 1000)} toneladas
        - **Aterro de Ribeirão Preto (CGR Guatapará):**
            - MCF = 1,0 (aterro sanitário anaeróbio)
            - Captura de metano = {CAPTURE_FRACTION_BASELINE*100:.0f}% (usina de biogás)
            - Fator φ (baseline) = {PHI_BASELINE} (clima úmido, UNFCCC 2024)
        - **Compostagem em leiras (TOOL13, v02.0):**
            - Fatores de emissão padrão: CH₄ = {EF_CH4_COMPOST_DEFAULT} t CH₄/t resíduo úmido; N₂O = {EF_N2O_COMPOST_DEFAULT} t N₂O/t resíduo úmido
        """)

        # Tabela comparativa de GWP
        st.subheader("📊 Comparação entre Cenários de GWP")
        comparacao = []
        for nome, res in results_all.items():
            comparacao.append({
                "Cenário": nome,
                "Emissões evitadas (tCO₂eq)": res['composting']['avoided_co2eq_t'],
                "Média anual (tCO₂eq/ano)": res['composting']['avoided_co2eq_t'] / anos_simulacao
            })
        df_comp_gwp = pd.DataFrame(comparacao)
        st.dataframe(df_comp_gwp.style.format({
            "Emissões evitadas (tCO₂eq)": lambda x: formatar_br(x),
            "Média anual (tCO₂eq/ano)": lambda x: formatar_br(x)
        }))

        # Valores financeiros (cenário otimista)
        total_evitado_compost = results['composting']['avoided_co2eq_t']
        preco_carbono = st.session_state.preco_carbono
        moeda = st.session_state.moeda_carbono
        taxa_cambio = st.session_state.taxa_cambio
        fonte_cotacao = st.session_state.fonte_cotacao

        valor_compost_eur = calcular_valor_creditos(total_evitado_compost, preco_carbono, moeda)
        valor_compost_brl = calcular_valor_creditos(total_evitado_compost, preco_carbono, "R$", taxa_cambio)

        st.subheader("💰 Valor Financeiro das Emissões Evitadas (Cenário Otimista)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Preço Carbono (Euro)", f"{moeda} {formatar_br(preco_carbono)}/tCO₂eq",
                      help=f"Fonte: {fonte_cotacao}")
        with col2:
            st.metric("Compostagem (Euro)", f"{moeda} {formatar_br(valor_compost_eur)}",
                      help=f"{formatar_br(total_evitado_compost)} tCO₂eq evitadas")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Preço Carbono (R$)", f"R$ {formatar_br(preco_carbono * taxa_cambio)}/tCO₂eq",
                      help="Preço convertido para Reais")
        with col2:
            st.metric("Compostagem (R$)", f"R$ {formatar_br(valor_compost_brl)}",
                      help=f"{formatar_br(total_evitado_compost)} tCO₂eq evitadas")

        with st.expander("💡 Como funciona a comercialização no mercado de carbono?"):
            st.markdown(f"""
            **📊 Informações de Mercado:**
            - Preço em Euro: {moeda} {formatar_br(preco_carbono)}/tCO₂eq
            - Preço em Real: R$ {formatar_br(preco_carbono * taxa_cambio)}/tCO₂eq
            - Taxa de câmbio: 1 Euro = R$ {formatar_br(taxa_cambio)}
            - Fonte: {fonte_cotacao}
            **💶 Comprar créditos (compensação):** Custo em Euro: {moeda} {formatar_br(valor_compost_eur)} | Custo em Real: R$ {formatar_br(valor_compost_brl)}
            **💵 Vender créditos (comercialização):** Receita em Euro: {moeda} {formatar_br(valor_compost_eur)} | Receita em Real: R$ {formatar_br(valor_compost_brl)}
            """)

        # Resumo emissões evitadas
        st.subheader("📊 Resumo das Emissões Evitadas (Cenário Otimista)")
        media_anual_compost = total_evitado_compost / anos_simulacao
        st.markdown("#### 📋 Compostagem Convencional em Leiras")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de emissões evitadas", f"{formatar_br(total_evitado_compost)} tCO₂eq",
                      help=f"Total em {anos_simulacao} anos")
        with col2:
            st.metric("Média anual", f"{formatar_br(media_anual_compost)} tCO₂eq/ano")

        # Gráfico de barras: emissões evitadas anuais
        st.subheader("📊 Emissões Evitadas Anuais (Cenário Otimista)")
        fig, ax = plt.subplots(figsize=(10, 6))
        br_formatter = FuncFormatter(br_format)
        anos_labels = df_anual_revisado['Year']
        valores = df_anual_revisado['Emission reductions (t CO₂eq)']
        ax.bar(anos_labels, valores, edgecolor='black', color='#2ca02c')
        for i, v in enumerate(valores):
            ax.text(anos_labels.iloc[i], v + max(valores)*0.01, formatar_br(v),
                    ha='center', fontsize=9, fontweight='bold')
        ax.set_xlabel('Ano')
        ax.set_ylabel('Emissões Evitadas (t CO₂eq)')
        ax.set_title('Emissões Evitadas Anuais pela Compostagem em Leiras')
        ax.yaxis.set_major_formatter(br_formatter)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

        # Gráfico de redução acumulada
        st.subheader("📉 Redução de Emissões Acumulada (Cenário Otimista)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Data'], df['Total_Aterro_tCO2eq_acum'], 'r-',
                label='Cenário Base (Aterro Sanitário)', linewidth=2)
        ax.plot(df['Data'], df['Total_Compost_tCO2eq_acum'], 'g-',
                label='Compostagem em Leiras', linewidth=2)
        ax.fill_between(df['Data'], df['Total_Compost_tCO2eq_acum'],
                        df['Total_Aterro_tCO2eq_acum'], color='skyblue',
                        alpha=0.5, label='Emissões Evitadas')
        ax.set_title(f'Redução de Emissões em {anos_simulacao} Anos (k = {formatar_br(k_ano)} ano⁻¹)')
        ax.set_xlabel('Ano')
        ax.set_ylabel('tCO₂eq Acumulado')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.set_major_formatter(br_formatter)
        st.pyplot(fig)
        plt.close(fig)

        # --- ANÁLISE DE SENSIBILIDADE SOBOL (GWP-20) ---
        st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - Compostagem em Leiras (GWP-20)")
        st.info("**Parâmetros variados:** Taxa de Decaimento (k), Temperatura (T), DOC")
        br_formatter_sobol = FuncFormatter(br_format)

        problem = {
            'num_vars': 3,
            'names': ['taxa_decaimento', 'T', 'DOC'],
            'bounds': [[0.06, 0.40], [20.0, 40.0], [0.10, 0.25]]
        }
        param_values = sample(problem, n_samples, seed=50)
        gwp20_ch4, gwp20_n2o = gwps["Otimista (GWP-20)"]

        # Paralelismo com n_jobs=1 para compatibilidade total com Streamlit Cloud
        results_compost = Parallel(n_jobs=1)(
            delayed(executar_simulacao_compostagem_sobol)(params, gwp20_ch4, gwp20_n2o)
            for params in param_values
        )
        Si_compost = analyze(problem, np.array(results_compost), print_to_console=False)

        sensibilidade_df_compost = pd.DataFrame({
            'Parâmetro': problem['names'],
            'S1': Si_compost['S1'],
            'ST': Si_compost['ST']
        }).sort_values('ST', ascending=False)
        nomes_amigaveis = {
            'taxa_decaimento': 'Taxa de Decaimento (k)',
            'T': 'Temperatura',
            'DOC': 'Carbono Orgânico Degradável'
        }
        sensibilidade_df_compost['Parâmetro'] = sensibilidade_df_compost['Parâmetro'].map(nomes_amigaveis)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ST', y='Parâmetro', data=sensibilidade_df_compost, palette='viridis', ax=ax)
        ax.set_title('Sensibilidade Global - Compostagem em Leiras (GWP-20)')
        ax.set_xlabel('Índice ST (Sobol Total)')
        ax.set_ylabel('Parâmetro')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.xaxis.set_major_formatter(br_formatter_sobol)
        for i, st_val in enumerate(sensibilidade_df_compost['ST']):
            ax.text(st_val, i, f' {formatar_br(st_val)}', va='center', fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(sensibilidade_df_compost.style.format({'S1': '{:.4f}', 'ST': '{:.4f}'}))

        # --- MONTE CARLO (todos os GWP) ---
        st.subheader("🎲 Análise de Incerteza (Monte Carlo) - Comparação entre Cenários de GWP")
        umidade_vals, temp_vals, doc_vals = gerar_parametros_mc(n_simulations)
        mc_results = {}
        for nome, (gwp_ch4, gwp_n2o) in gwps.items():
            thermo_arr = []
            for i in range(n_simulations):
                calc_mc = GHGEmissionCalculator()
                calc_mc.GWP_CH4_20 = gwp_ch4
                calc_mc.GWP_N2O_20 = gwp_n2o
                res = calc_mc.calculate_avoided_emissions(
                    waste_kg_day=residuos_kg_dia,
                    k_year=k_ano,
                    temperature_C=temp_vals[i],
                    doc_fraction=doc_vals[i],
                    moisture_fraction=umidade_vals[i],
                    years=anos_simulacao,
                    phi_baseline=PHI_BASELINE,
                    capture_fraction=CAPTURE_FRACTION_BASELINE
                )
                thermo_arr.append(res['composting']['avoided_co2eq_t'])
            mc_results[nome] = {
                'composting': np.array(thermo_arr)
            }

        # Distribuições (apenas compostagem)
        fig, ax = plt.subplots(figsize=(12, 6))
        for nome, arr_dict in mc_results.items():
            sns.kdeplot(arr_dict['composting'], label=nome, ax=ax, linewidth=2)
        ax.set_title('Distribuição das Emissões Evitadas (Compostagem em Leiras)')
        ax.set_xlabel('Emissões Evitadas (tCO₂eq)')
        ax.set_ylabel('Densidade')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(br_format))
        st.pyplot(fig)
        plt.close(fig)

        # Estatísticas descritivas
        stats_list = []
        for nome, arr_dict in mc_results.items():
            arr = arr_dict['composting']
            stats_list.append({
                "Cenário": nome,
                "Média (tCO₂eq)": np.mean(arr),
                "Mediana (tCO₂eq)": np.median(arr),
                "Desvio Padrão": np.std(arr),
                "IC 95% Inferior": np.percentile(arr, 2.5),
                "IC 95% Superior": np.percentile(arr, 97.5)
            })
        df_mc_stats = pd.DataFrame(stats_list)
        st.subheader("📊 Estatísticas do Monte Carlo - Compostagem em Leiras")
        st.dataframe(df_mc_stats.style.format({
            "Média (tCO₂eq)": lambda x: formatar_br(x),
            "Mediana (tCO₂eq)": lambda x: formatar_br(x),
            "Desvio Padrão": lambda x: formatar_br(x),
            "IC 95% Inferior": lambda x: formatar_br(x),
            "IC 95% Superior": lambda x: formatar_br(x)
        }))

        # Tabelas anuais formatadas
        st.subheader("📋 Resultados Anuais - Compostagem em Leiras (Cenário Otimista)")
        df_anual_formatado = df_anual_revisado.copy()
        for col in df_anual_formatado.columns:
            if col != 'Year':
                df_anual_formatado[col] = df_anual_formatado[col].apply(formatar_br)
        st.dataframe(df_anual_formatado)

    # Reset do estado para permitir nova simulação
    st.session_state.run_simulation = False

else:
    st.info("💡 Ajuste os parâmetros na barra lateral e clique em 'Executar Simulação' para ver os resultados.")

st.markdown("---")
st.markdown("""
**📚 Referências por Cenário:**

**Cenário de Baseline (Aterro Sanitário – calibrado para Ribeirão Preto):**
- Metano: IPCC (2006), ferramenta A6.4-AMT-003 (v01.0) – "Emissions from solid waste disposal sites".
- Óxido Nitroso: Wang et al. (2017).
- Metano e Óxido Nitroso no pré-descarte: Feng et al. (2020).
- **Aterro CGR Guatapará:** MCF = 1,0 (sanitário anaeróbio), captura de metano = 60% (usina de biogás em operação), fator φ = 0,85 (clima úmido, Application B).

**Compostagem convencional em leiras (windrow):**
- Protocolo AMS-III.F: UNFCCC (2016) – "Avoidance of methane emissions through composting".
- Ferramenta TOOL13, versão 02.0 (UNFCCC, 2017) – "Project and leakage emissions from composting".
- Fatores de emissão padrão (seção 6.3 do TOOL13):
    - CH₄ = 0,002 t CH₄ / t resíduo úmido
    - N₂O = 0,0005 t N₂O / t resíduo úmido

**Cenários de Potencial de Aquecimento Global (GWP):**
- **Otimista (GWP-20):** CH₄ = 79,7; N₂O = 273 (Forster et al., 2021)
- **Realista (GWP-100):** CH₄ = 27,0; N₂O = 273 (Forster et al., 2021)
- **Pessimista (GWP-500):** CH₄ = 7,2; N₂O = 130 (Forster et al., 2021)

**⚠️ Nota de Reprodutibilidade:**
- Todas as análises usam seed fixo (50) para garantir resultados reprodutíveis.
- Métodos de cálculo idênticos aos utilizados na validação original, com a compostagem ajustada aos valores padrão da UNFCCC e o baseline calibrado para a realidade operacional do aterro de Ribeirão Preto.
""")
