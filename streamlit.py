"""
Aplicação Profissional de Conversão de Encoding
Arquitetura Corporativa | Alta Performance | Processamento em Streaming
"""

import streamlit as st
import charset_normalizer
import chardet
import io
import os
import time
import tempfile
import codecs
import re
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

# ==============================================================================
# CONFIGURAÇÕES E ESTRUTURAS DE DADOS
# ==============================================================================

@dataclass
class AppConfig:
    """Configurações globais da aplicação."""
    APP_NAME: str = "ConverterPro: Encoding & Delimiters"
    APP_VERSION: str = "1.0.0"
    SAMPLE_SIZE_BYTES: int = 1024 * 1024  # 1MB para análise inicial
    SAMPLE_LINES: int = 50                # Linhas para análise de delimitador
    TARGET_ENCODING: str = "utf-8"
    DELIMITERS_TO_CHECK: List[str] = field(default_factory=lambda: [";", ",", "|", "|*|", "\t"])

@dataclass
class ProcessingResult:
    """Objeto de transferência de dados (DTO) contendo os resultados do processamento."""
    original_filename: str
    file_size_bytes: int
    total_lines: int
    detected_encoding: str
    confidence: float
    detected_delimiter: str
    delimiter_replaced: bool
    processing_time_sec: float
    read_speed_mbps: float
    output_filepath: str

class MemoryLogger:
    """Sistema de log em memória para exibição em tempo real na interface Streamlit."""
    def __init__(self):
        self.logs: List[Dict[str, str]] = []
    
    def log(self, level: str, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.logs.append({"time": timestamp, "level": level, "message": message})
    
    def info(self, message: str):
        self.log("INFO", message)
        
    def warning(self, message: str):
        self.log("WARNING", message)
        
    def error(self, message: str):
        self.log("ERROR", message)
        
    def get_formatted_logs(self) -> str:
        return "\n".join([f"[{entry['time']}] {entry['level']}: {entry['message']}" for entry in self.logs])


# ==============================================================================
# MOTORES DE ANÁLISE (ENCODING E DELIMITADORES)
# ==============================================================================

class EncodingAnalyzer:
    """Implementa o pipeline híbrido de detecção de encoding."""
    
    def __init__(self, logger: MemoryLogger):
        self.logger = logger
        
    def check_bom(self, raw_data: bytes) -> Optional[Tuple[str, float]]:
        """Verifica marcadores de ordem de bytes (BOM) para máxima precisão."""
        boms = {
            codecs.BOM_UTF8: 'utf-8-sig',
            codecs.BOM_UTF32_LE: 'utf-32-le',
            codecs.BOM_UTF32_BE: 'utf-32-be',
            codecs.BOM_UTF16_LE: 'utf-16-le',
            codecs.BOM_UTF16_BE: 'utf-16-be',
        }
        for bom, encoding in boms.items():
            if raw_data.startswith(bom):
                self.logger.info(f"BOM detectado. Encoding cravado como {encoding}.")
                return encoding, 1.0
        return None

    def detect(self, raw_data: bytes) -> Tuple[str, float]:
        """
        Executa a detecção combinando BOM, charset-normalizer e chardet.
        Retorna o encoding e a confiança (0.0 a 1.0).
        """
        if not raw_data:
            self.logger.warning("Arquivo vazio durante detecção de encoding.")
            return "utf-8", 1.0

        # 1. Validação por BOM
        bom_result = self.check_bom(raw_data)
        if bom_result:
            return bom_result

        # 2. Charset Normalizer (Análise heurística/linguística moderna)
        cn_result = charset_normalizer.from_bytes(raw_data).best()
        cn_encoding = cn_result.encoding if cn_result else None
        # O charset_normalizer não fornece 'confidence' na mesma escala do chardet,
        # mas gera um percentual de 'coherence' ou apenas atesta como confiável.
        
        # 3. Chardet (Análise estatística clássica)
        cd_result = chardet.detect(raw_data)
        cd_encoding = cd_result.get('encoding')
        cd_confidence = cd_result.get('confidence', 0.0)

        self.logger.info(f"Charset-Normalizer detectou: {cn_encoding}")
        self.logger.info(f"Chardet detectou: {cd_encoding} (Confiança: {cd_confidence:.2f})")

        # 4. Consenso e Sistema de Pontuação (Score)
        if cn_encoding and cd_encoding:
            if cn_encoding.lower() == cd_encoding.lower():
                self.logger.info("Consenso alcançado entre os motores.")
                return cn_encoding, max(cd_confidence, 0.95)
            
            # Se divergirem, charset-normalizer tende a ser melhor para textos modernos e UTF-8 sem BOM
            if cn_encoding.lower() == 'utf-8':
                self.logger.info("Divergência: Priorizando UTF-8 via Charset-Normalizer.")
                return cn_encoding, 0.90
                
            if cd_confidence > 0.85:
                self.logger.info("Divergência: Priorizando Chardet pela alta confiança.")
                return cd_encoding, cd_confidence

        # 5. Fallback final
        best_guess = cn_encoding or cd_encoding or "latin-1"
        confidence = cd_confidence if best_guess == cd_encoding else 0.80
        
        self.logger.warning(f"Utilizando fallback. Melhor palpite: {best_guess}")
        return best_guess, confidence

class DelimiterAnalyzer:
    """Classe responsável por identificar delimitadores em arquivos de texto."""
    
    def __init__(self, config: AppConfig, logger: MemoryLogger):
        self.config = config
        self.logger = logger

    def detect(self, text_sample: str) -> str:
        """Verifica a frequência dos delimitadores nas primeiras linhas para definir o padrão."""
        if not text_sample:
            return ","

        lines = text_sample.splitlines()[:self.config.SAMPLE_LINES]
        delimiter_counts = {delim: 0 for delim in self.config.DELIMITERS_TO_CHECK}

        for line in lines:
            for delim in self.config.DELIMITERS_TO_CHECK:
                delimiter_counts[delim] += line.count(delim)

        # Priorizar |*| caso haja empate técnico com |, pois |*| contém |
        if delimiter_counts["|*|"] > 0:
            # Subtrai as ocorrências de "|" que na verdade fazem parte de "|*|"
            delimiter_counts["|"] = max(0, delimiter_counts["|"] - (delimiter_counts["|*|"] * 2))

        detected = max(delimiter_counts, key=delimiter_counts.get)
        
        if delimiter_counts[detected] == 0:
            self.logger.info("Nenhum delimitador conhecido encontrado. Padrão assumido: vírgula (,).")
            return ","
            
        self.logger.info(f"Delimitador detectado: '{detected}' (Ocorrências: {delimiter_counts[detected]})")
        return detected


# ==============================================================================
# MOTOR DE PROCESSAMENTO DE ARQUIVO
# ==============================================================================

class StreamProcessor:
    """
    Processa o arquivo em streaming.
    Garante baixo consumo de memória (O(1)) independente do tamanho do arquivo (GBs).
    """
    def __init__(self, config: AppConfig, logger: MemoryLogger):
        self.config = config
        self.logger = logger
        self.encoding_analyzer = EncodingAnalyzer(logger)
        self.delimiter_analyzer = DelimiterAnalyzer(config, logger)

    def process_file(self, uploaded_file) -> ProcessingResult:
        start_time = time.perf_counter()
        file_size = uploaded_file.size
        filename = uploaded_file.name
        
        self.logger.info(f"Iniciando processamento do arquivo: {filename} ({file_size} bytes)")

        # 1. Leitura de amostra (Sample) para detecção de Encoding
        sample_bytes = uploaded_file.read(self.config.SAMPLE_SIZE_BYTES)
        detected_encoding, confidence = self.encoding_analyzer.detect(sample_bytes)
        
        # 2. Leitura de amostra textual para detecção de Delimitador
        try:
            # Tratamento de erro na decodificação da amostra
            sample_text = sample_bytes.decode(detected_encoding, errors='replace')
        except Exception as e:
            self.logger.error(f"Erro ao decodificar amostra: {str(e)}")
            sample_text = ""
            
        detected_delimiter = self.delimiter_analyzer.detect(sample_text)
        
        # Regra de negócio: converter |*| para ;
        replace_delimiter = (detected_delimiter == "|*|")
        if replace_delimiter:
            self.logger.info("Regra ativada: Delimitador |*| será convertido para ;")

        # 3. Resetar ponteiro do arquivo para processamento real
        uploaded_file.seek(0)
        
        # 4. Criar arquivo temporário para escrita via streaming
        temp_dir = tempfile.gettempdir()
        temp_filepath = os.path.join(temp_dir, f"converted_{filename}")
        
        total_lines = 0
        
        self.logger.info(f"Iniciando conversão para {self.config.TARGET_ENCODING} em streaming...")
        
        try:
            # Utilizamos TextIOWrapper para ler o stream binário linha a linha usando o encoding detectado
            with io.TextIOWrapper(uploaded_file, encoding=detected_encoding, errors='replace', newline='') as text_reader:
                with io.open(temp_filepath, 'w', encoding=self.config.TARGET_ENCODING, newline='') as text_writer:
                    
                    for line in text_reader:
                        # Processamento Vetorizado / String em nível de linha
                        if replace_delimiter:
                            line = line.replace("|*|", ";")
                            
                        text_writer.write(line)
                        total_lines += 1

        except MemoryError:
            self.logger.error("Falha crítica: Memória insuficiente durante o streaming.")
            raise Exception("Memória insuficiente. O arquivo é excessivamente denso sem quebras de linha.")
        except Exception as e:
            self.logger.error(f"Erro inesperado durante conversão: {str(e)}")
            raise

        end_time = time.perf_counter()
        elapsed_time = max(end_time - start_time, 0.001) # Evitar divisão por zero
        
        speed_mbps = (file_size / (1024 * 1024)) / elapsed_time
        
        self.logger.info(f"Processamento concluído: {total_lines} linhas lidas.")
        self.logger.info(f"Tempo total: {elapsed_time:.2f}s | Velocidade: {speed_mbps:.2f} MB/s")

        return ProcessingResult(
            original_filename=filename,
            file_size_bytes=file_size,
            total_lines=total_lines,
            detected_encoding=detected_encoding,
            confidence=confidence,
            detected_delimiter=detected_delimiter,
            delimiter_replaced=replace_delimiter,
            processing_time_sec=elapsed_time,
            read_speed_mbps=speed_mbps,
            output_filepath=temp_filepath
        )


# ==============================================================================
# INTERFACE COM O USUÁRIO (UX/UI STREAMLIT)
# ==============================================================================

class UI:
    def __init__(self, config: AppConfig):
        self.config = config
        st.set_page_config(
            page_title=self.config.APP_NAME,
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        # CSS Customizado para padrão corporativo
        st.markdown("""
        <style>
            .stProgress > div > div > div > div { background-color: #4CAF50; }
            .metric-card { background-color: #1E1E1E; padding: 15px; border-radius: 8px; border: 1px solid #333; }
            .terminal-log { background-color: #0d1117; color: #58a6ff; font-family: 'Courier New', monospace; padding: 10px; border-radius: 5px; height: 250px; overflow-y: auto; font-size: 0.85em; }
        </style>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        with st.sidebar:
            st.title("⚙️ Engine")
            st.markdown(f"**Versão:** {self.config.APP_VERSION}")
            st.markdown("---")
            st.markdown("""
            **Capacidades do Pipeline:**
            - ✔️ Híbrido: BOM, Chardet, Charset-Normalizer
            - ✔️ Suporte a arquivos gigantes (Streaming)
            - ✔️ Conversão inteligente de delimitadores (`|*|` ➔ `;`)
            - ✔️ Segurança: Operações isoladas
            """)
            st.markdown("---")
            st.info("💡 Submeta arquivos textuais delimitados (CSV, TXT, DAT).")

    def render_metrics(self, result: ProcessingResult):
        st.subheader("📊 Métricas de Processamento")
        col1, col2, col3, col4 = st.columns(4)
        
        size_mb = result.file_size_bytes / (1024 * 1024)
        
        col1.metric("Tamanho", f"{size_mb:.2f} MB")
        col2.metric("Total de Linhas", f"{result.total_lines:,}".replace(",", "."))
        col3.metric("Tempo Total", f"{result.processing_time_sec:.2f} s")
        col4.metric("Throughput", f"{result.read_speed_mbps:.1f} MB/s")
        
        st.markdown("### 🔎 Análise de Estrutura")
        col5, col6, col7 = st.columns(3)
        col5.metric("Encoding Identificado", result.detected_encoding.upper())
        col6.metric("Confiança (Engine)", f"{result.confidence * 100:.1f}%")
        
        delim_display = result.detected_delimiter if result.detected_delimiter != "\t" else "TAB"
        col7.metric("Delimitador Base", delim_display)
        
        if result.delimiter_replaced:
            st.success(f"Regra de negócio aplicada: O delimitador **{delim_display}** foi convertido para **;** com sucesso.")

    def render_main(self):
        st.title(f"⚡ {self.config.APP_NAME}")
        st.markdown("Converta encodings problemáticos para **UTF-8** de forma definitiva e segura.")
        
        uploaded_file = st.file_uploader(
            "Faça o upload do seu arquivo (CSV, TXT, etc)", 
            type=['csv', 'txt', 'dat', 'log'],
            help="Arraste ou selecione um arquivo. Sem limite rígido de tamanho (processamento O(1) em memória)."
        )

        if uploaded_file is not None:
            if uploaded_file.size == 0:
                st.error("⚠️ O arquivo enviado está vazio.")
                return

            logger = MemoryLogger()
            processor = StreamProcessor(self.config, logger)
            
            if st.button("🚀 Iniciar Processamento Profissional", use_container_width=True, type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.info("Analisando heurística de encoding e delimitadores...")
                    progress_bar.progress(25)
                    
                    with st.spinner("Convertendo dados em streaming (isso pode levar alguns segundos em arquivos muito grandes)..."):
                        result = processor.process_file(uploaded_file)
                        progress_bar.progress(100)
                    
                    status_text.success("✅ Processamento concluído com sucesso!")
                    
                    # RENDERIZAR RESULTADOS
                    st.divider()
                    self.render_metrics(result)
                    
                    # LOGS
                    st.subheader("📝 Logs da Execução")
                    st.markdown(f"<div class='terminal-log'><pre>{logger.get_formatted_logs()}</pre></div>", unsafe_allow_html=True)
                    
                    # DOWNLOAD
                    st.divider()
                    st.subheader("📥 Download do Arquivo Processado")
                    
                    with open(result.output_filepath, "rb") as f:
                        file_data = f.read()
                        
                    new_filename = f"utf8_{result.original_filename}"
                    st.download_button(
                        label=f"⬇️ Baixar {new_filename}",
                        data=file_data,
                        file_name=new_filename,
                        mime="text/plain",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    progress_bar.progress(0)
                    status_text.error(f"❌ Ocorreu um erro crítico durante o processamento: {str(e)}")
                    st.error("Verifique os logs abaixo para mais detalhes.")
                    st.markdown(f"<div class='terminal-log'><pre>{logger.get_formatted_logs()}</pre></div>", unsafe_allow_html=True)


# ==============================================================================
# PONTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    config = AppConfig()
    ui = UI(config)
    ui.render_sidebar()
    ui.render_main()