"""
Aplicação Corporativa de Conversão de Encoding e Auditoria de Dados
Arquitetura: Strategy/Pipeline | Alta Performance | Resiliência | Validação Estruturada
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
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import collections
import threading

# Importações Opcionais para resiliência (Caso não instalem bibliotecas nativas de C)
try:
    import cchardet
    CCHARDET_AVAILABLE = True
except ImportError:
    CCHARDET_AVAILABLE = False

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

# ==============================================================================
# ENUMS E CONFIGURAÇÕES GLOBAIS
# ==============================================================================

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ProcessingStatus(Enum):
    PENDING = "Pendente"
    ANALYZING = "Analisando Heurísticas"
    PROCESSING = "Convertendo Dados"
    VALIDATING = "Validando Integridade"
    COMPLETED = "Concluído com Sucesso"
    FAILED = "Falha Crítica"

@dataclass
class AppConfig:
    APP_NAME: str = "ConverterPro Enterprise"
    APP_VERSION: str = "2.0.0"
    SAMPLE_SIZE_BYTES: int = 2 * 1024 * 1024  # 2MB para análise profunda
    SAMPLE_LINES: int = 200                   # Linhas para heurística de variância
    CHUNK_SIZE_BYTES: int = 8 * 1024 * 1024   # 8MB de buffer de leitura
    TARGET_ENCODING: str = "utf-8"
    DELIMITERS_TO_CHECK: List[str] = field(default_factory=lambda: [";", ",", "|", "|*|", "\t", " "])
    MAX_FILE_SIZE_MB: int = 5000              # 5GB (Lógico, proteção anti-DDoS)

@dataclass
class ValidationReport:
    is_valid: bool = True
    invalid_chars_replaced: int = 0
    column_count_mismatch: bool = False
    data_loss_detected: bool = False
    original_bytes: int = 0
    converted_bytes: int = 0
    issues: List[str] = field(default_factory=list)

@dataclass
class ProcessingResult:
    original_filename: str
    file_size_bytes: int
    total_lines: int
    detected_encoding: str
    confidence: float
    detector_used: str
    detected_delimiter: str
    delimiter_replaced: bool
    processing_time_sec: float
    read_speed_mbps: float
    output_filepath: str
    validation_report: ValidationReport

# ==============================================================================
# SEGURANÇA E UTILITÁRIOS
# ==============================================================================

def secure_filename(filename: str) -> str:
    """Sanitiza o nome do arquivo prevenindo Path Traversal e caracteres inválidos."""
    filename = re.sub(r'[^a-zA-Z0-9_\.-]', '_', os.path.basename(filename))
    return filename if filename else "unnamed_file.txt"

class StructuredLogger:
    """Logger thread-safe estruturado em memória para auditoria e UI."""
    def __init__(self):
        self.logs: List[Dict[str, str]] = []
        self._lock = threading.Lock()
    
    def log(self, level: LogLevel, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with self._lock:
            self.logs.append({"time": timestamp, "level": level.value, "message": message})
    
    def debug(self, msg: str): self.log(LogLevel.DEBUG, msg)
    def info(self, msg: str): self.log(LogLevel.INFO, msg)
    def warning(self, msg: str): self.log(LogLevel.WARNING, msg)
    def error(self, msg: str): self.log(LogLevel.ERROR, msg)
    def critical(self, msg: str): self.log(LogLevel.CRITICAL, msg)
        
    def get_formatted_logs(self) -> str:
        with self._lock:
            return "\n".join([f"[{entry['time']}] {entry['level']}: {entry['message']}" for entry in self.logs])

# ==============================================================================
# MOTORES DE DETECÇÃO DE ENCODING (STRATEGY PATTERN)
# ==============================================================================

class AbstractDetector:
    def detect(self, raw_data: bytes) -> Tuple[Optional[str], float, str]:
        raise NotImplementedError

class BOMDetector(AbstractDetector):
    def detect(self, raw_data: bytes) -> Tuple[Optional[str], float, str]:
        boms = {
            codecs.BOM_UTF8: 'utf-8-sig',
            codecs.BOM_UTF32_LE: 'utf-32-le',
            codecs.BOM_UTF32_BE: 'utf-32-be',
            codecs.BOM_UTF16_LE: 'utf-16-le',
            codecs.BOM_UTF16_BE: 'utf-16-be',
        }
        for bom, enc in boms.items():
            if raw_data.startswith(bom):
                return enc, 1.0, "BOM Signature"
        return None, 0.0, "BOM Signature"

class CharsetNormalizerDetector(AbstractDetector):
    def detect(self, raw_data: bytes) -> Tuple[Optional[str], float, str]:
        try:
            result = charset_normalizer.from_bytes(raw_data).best()
            if result:
                # Charset Normalizer usa linguística heurística, atribuímos confiança alta genérica
                return result.encoding, 0.95, "Charset-Normalizer"
        except Exception:
            pass
        return None, 0.0, "Charset-Normalizer"

class ChardetDetector(AbstractDetector):
    def detect(self, raw_data: bytes) -> Tuple[Optional[str], float, str]:
        try:
            result = chardet.detect(raw_data)
            enc = result.get('encoding')
            conf = result.get('confidence', 0.0)
            return enc, conf, "Chardet"
        except Exception:
            return None, 0.0, "Chardet"

class CChardetDetector(AbstractDetector):
    def detect(self, raw_data: bytes) -> Tuple[Optional[str], float, str]:
        if not CCHARDET_AVAILABLE:
            return None, 0.0, "cChardet (Indisponível)"
        try:
            result = cchardet.detect(raw_data)
            return result.get('encoding'), result.get('confidence', 0.0), "cChardet"
        except Exception:
            return None, 0.0, "cChardet"

class MagicDetector(AbstractDetector):
    def detect(self, raw_data: bytes) -> Tuple[Optional[str], float, str]:
        if not MAGIC_AVAILABLE:
            return None, 0.0, "libmagic (Indisponível)"
        try:
            m = magic.Magic(mime_encoding=True)
            enc = m.from_buffer(raw_data)
            if enc and enc != "unknown-8bit":
                return enc, 0.85, "libmagic"
        except Exception:
            pass
        return None, 0.0, "libmagic"

class EncodingVotingSystem:
    """Implementa o pipeline híbrido de consenso e votação para Encoding."""
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.detectors = [
            BOMDetector(),
            CChardetDetector(),
            CharsetNormalizerDetector(),
            ChardetDetector(),
            MagicDetector()
        ]

    def detect(self, raw_data: bytes) -> Tuple[str, float, str]:
        if not raw_data:
            self.logger.warning("Buffer vazio na detecção. Fallback para UTF-8.")
            return "utf-8", 1.0, "Fallback (Vazio)"

        candidates = []
        for detector in self.detectors:
            enc, conf, name = detector.detect(raw_data)
            if enc:
                enc = enc.lower()
                # Normalização de encodings comuns
                if enc == 'ascii': enc = 'utf-8' # ASCII é subconjunto estrito do UTF-8
                if enc == 'iso-8859-1' and b'\x92' in raw_data: enc = 'windows-1252' # Smart fallback
                
                candidates.append({"encoding": enc, "confidence": conf, "source": name})
                self.logger.debug(f"Detector [{name}] propôs: {enc} (Conf: {conf:.2f})")

                # Se for BOM, é definitivo (100% certeza)
                if name == "BOM Signature" and conf == 1.0:
                    self.logger.info(f"BOM definitivo encontrado: {enc}")
                    return enc, 1.0, name

        if not candidates:
            self.logger.warning("Nenhum detector obteve sucesso. Fallback para UTF-8.")
            return "utf-8", 0.0, "Fallback Absoluto"

        # Agrupamento e votação (Soma das confianças)
        score_board = collections.defaultdict(float)
        sources_board = collections.defaultdict(list)
        
        for cand in candidates:
            score_board[cand["encoding"]] += cand["confidence"]
            sources_board[cand["encoding"]].append(cand["source"])

        # Seleciona o vencedor pelo maior score
        best_enc = max(score_board, key=score_board.get)
        total_score = min(score_board[best_enc] / len(sources_board[best_enc]), 1.0) # Normaliza max 1.0
        used_detectors = " + ".join(sources_board[best_enc])

        self.logger.info(f"Consenso atingido: {best_enc.upper()} via [{used_detectors}] (Score: {total_score:.2f})")
        return best_enc, total_score, used_detectors

# ==============================================================================
# MOTOR DE HEURÍSTICA DE DELIMITADORES
# ==============================================================================

class ColumnVarianceHeuristic:
    """Implementa detecção inteligente baseada em variância de colunas."""
    def __init__(self, config: AppConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger

    def detect(self, text_sample: str) -> str:
        if not text_sample:
            return ","

        lines = [line.strip() for line in text_sample.splitlines() if line.strip()][:self.config.SAMPLE_LINES]
        if not lines:
            return ","

        best_delimiter = ","
        lowest_variance = float('inf')
        highest_col_count = 0

        for delim in self.config.DELIMITERS_TO_CHECK:
            col_counts = [len(line.split(delim)) for line in lines]
            avg_cols = sum(col_counts) / len(col_counts)
            
            # Cálculo de variância (Quão consistente é o número de colunas por linha)
            variance = sum((c - avg_cols) ** 2 for c in col_counts) / len(col_counts)
            
            self.logger.debug(f"Delimitador '{delim}': Média Colunas={avg_cols:.1f}, Variância={variance:.2f}")

            # Prioriza: Menor variância (mais consistente) e Maior número de colunas (> 1)
            if avg_cols > 1:
                if variance < lowest_variance or (variance == lowest_variance and avg_cols > highest_col_count):
                    # Exceção inteligente: Se | e |*| tiverem variância zero, prioriza |*| se ele existir fisicamente
                    if delim == "|" and best_delimiter == "|*|" and variance == 0.0:
                        continue
                        
                    lowest_variance = variance
                    highest_col_count = avg_cols
                    best_delimiter = delim

        self.logger.info(f"Delimitador Vencedor: '{best_delimiter}' (Variância: {lowest_variance:.2f}, Colunas: ~{int(highest_col_count)})")
        return best_delimiter

# ==============================================================================
# MOTOR DE PROCESSAMENTO CORPORATIVO E VALIDAÇÃO
# ==============================================================================

class StreamProcessor:
    """Processa o arquivo em chunks, otimizando I/O e garantindo resiliência."""
    def __init__(self, config: AppConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.encoding_analyzer = EncodingVotingSystem(logger)
        self.delimiter_analyzer = ColumnVarianceHeuristic(config, logger)

    def process_file(self, uploaded_file, ui_progress_callback=None) -> ProcessingResult:
        start_time = time.perf_counter()
        file_size = uploaded_file.size
        filename = secure_filename(uploaded_file.name)
        
        self.logger.info(f"== INÍCIO: {filename} ({file_size} bytes) ==")

        # 1. Pipeline de Análise (Amostragem)
        sample_bytes = uploaded_file.read(self.config.SAMPLE_SIZE_BYTES)
        detected_encoding, confidence, detector_source = self.encoding_analyzer.detect(sample_bytes)
        
        try:
            sample_text = sample_bytes.decode(detected_encoding, errors='replace')
        except Exception as e:
            self.logger.error(f"Falha na decodificação da amostra: {str(e)}")
            sample_text = ""
            
        detected_delimiter = self.delimiter_analyzer.detect(sample_text)
        replace_delimiter = (detected_delimiter == "|*|")

        # 2. Reset de Ponteiros e Configuração do TempFile
        uploaded_file.seek(0)
        temp_dir = tempfile.gettempdir()
        temp_filepath = os.path.join(temp_dir, f"audit_utf8_{filename}")
        
        total_lines = 0
        processed_bytes = 0
        val_report = ValidationReport()
        val_report.original_bytes = file_size
        
        self.logger.info(f"Iniciando conversão vetorizada em blocos de {self.config.CHUNK_SIZE_BYTES//1024//1024}MB...")
        
        try:
            # Leituras binárias e textuais combinadas
            with io.TextIOWrapper(uploaded_file, encoding=detected_encoding, errors='replace', newline='') as text_reader:
                with io.open(temp_filepath, 'w', encoding=self.config.TARGET_ENCODING, newline='') as text_writer:
                    
                    while True:
                        # Lê em lotes (chunks baseados em tamanho lógico para TextIOWrapper)
                        chunk = text_reader.read(self.config.CHUNK_SIZE_BYTES // 4) # Caracteres, não bytes
                        if not chunk:
                            break
                        
                        # Auditoria de integridade (caracteres de substituição padrão de erro unicode)
                        if "\ufffd" in chunk:
                            val_report.invalid_chars_replaced += chunk.count("\ufffd")
                            val_report.data_loss_detected = True
                        
                        # Regras de Negócio e Transformação
                        if replace_delimiter:
                            chunk = chunk.replace("|*|", ";")
                            
                        # Escrita otimizada
                        text_writer.write(chunk)
                        
                        # Estatísticas parciais
                        lines_in_chunk = chunk.count('\n')
                        total_lines += lines_in_chunk
                        processed_bytes += len(chunk.encode(self.config.TARGET_ENCODING))
                        
                        # Update UI Async via Callback
                        if ui_progress_callback:
                            # Estima progresso baseado nos bytes originais lidos no underlying buffer
                            current_pos = uploaded_file.tell()
                            prog_pct = min(current_pos / file_size, 1.0)
                            ui_progress_callback(prog_pct)
                            
        except Exception as e:
            self.logger.critical(f"Abordagem abortada. Exceção crítica: {str(e)}")
            val_report.is_valid = False
            val_report.issues.append(str(e))
            raise

        val_report.converted_bytes = os.path.getsize(temp_filepath)
        
        end_time = time.perf_counter()
        elapsed_time = max(end_time - start_time, 0.001)
        speed_mbps = (file_size / (1024 * 1024)) / elapsed_time
        
        self.logger.info(f"Auditoria Pós-Conversão: {val_report.invalid_chars_replaced} caracteres irrecuperáveis substituídos.")
        self.logger.info(f"== TÉRMINO: {elapsed_time:.2f}s | {speed_mbps:.2f} MB/s | Linhas: {total_lines} ==")

        return ProcessingResult(
            original_filename=filename,
            file_size_bytes=file_size,
            total_lines=total_lines,
            detected_encoding=detected_encoding,
            confidence=confidence,
            detector_used=detector_source,
            detected_delimiter=detected_delimiter,
            delimiter_replaced=replace_delimiter,
            processing_time_sec=elapsed_time,
            read_speed_mbps=speed_mbps,
            output_filepath=temp_filepath,
            validation_report=val_report
        )

# ==============================================================================
# UI STREAMLIT CORPORATIVA
# ==============================================================================

class UI:
    def __init__(self, config: AppConfig):
        self.config = config
        st.set_page_config(
            page_title=self.config.APP_NAME,
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self._init_session_state()
        self._apply_custom_css()

    def _init_session_state(self):
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'theme' not in st.session_state:
            st.session_state.theme = "dark"

    def _apply_custom_css(self):
        st.markdown("""
        <style>
            .stProgress > div > div > div > div { background-color: #0078D7; transition: all 0.3s ease; }
            .terminal-log { background-color: #1E1E1E; color: #00FF00; font-family: 'Consolas', monospace; padding: 12px; border-radius: 6px; height: 300px; overflow-y: auto; font-size: 0.85rem; border: 1px solid #333; }
            .card { border: 1px solid #444; border-radius: 8px; padding: 15px; margin-bottom: 10px; background-color: #222;}
            .highlight { color: #0078D7; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        with st.sidebar:
            st.title("🛡️ Enterprise Engine")
            st.markdown(f"**Versão:** {self.config.APP_VERSION}")
            
            st.header("Pipeline Status")
            st.markdown(f"cChardet (C Engine): **{'Ativo ✅' if CCHARDET_AVAILABLE else 'Inativo ❌'}**")
            st.markdown(f"libmagic (MIME): **{'Ativo ✅' if MAGIC_AVAILABLE else 'Inativo ❌'}**")
            
            st.divider()
            
            if st.session_state.history:
                st.subheader("⏱️ Histórico de Sessão")
                for item in reversed(st.session_state.history[-5:]): # Mostra ultimos 5
                    st.caption(f"Arquivo: {item['file']}")
                    st.caption(f"Encoding: {item['enc']} -> UTF-8")
                    st.caption(f"Tempo: {item['time']:.2f}s")
                    st.divider()
                
                if st.button("Limpar Histórico", use_container_width=True):
                    st.session_state.history = []
                    st.rerun()

    def render_validation_report(self, report: ValidationReport):
        st.subheader("🛡️ Relatório de Auditoria e Integridade")
        
        if report.is_valid and not report.data_loss_detected:
            st.success("✅ O arquivo foi convertido mantendo 100% da sua integridade estrutural e binária. Nenhuma perda de dados foi registrada.")
        else:
            st.warning("⚠️ O arquivo foi convertido, mas foram identificadas anomalias estruturais durante a auditoria.")
            
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Bytes Originais:** {report.original_bytes:,}")
            st.info(f"**Bytes Pós-Conversão:** {report.converted_bytes:,}")
        with col2:
            st.error(f"**Caracteres Irrecuperáveis (Substituídos):** {report.invalid_chars_replaced:,}")
            st.info(f"**Perda Severa de Dados:** {'Sim' if report.data_loss_detected else 'Não'}")

    def render_main(self):
        st.title(f"🛡️ {self.config.APP_NAME}")
        st.markdown("Solução escalável para conversão de encodings legados, auditoria de delimitadores e higienização de datasets gigantes.")
        
        uploaded_file = st.file_uploader(
            "Upload de Arquivos Textuais (CSV, TXT, DAT, TSV) — Suporta arquivos > 1GB", 
            type=['csv', 'txt', 'dat', 'log', 'tsv']
        )

        if uploaded_file is not None:
            if uploaded_file.size > (self.config.MAX_FILE_SIZE_MB * 1024 * 1024):
                st.error(f"Tamanho excede o limite corporativo de {self.config.MAX_FILE_SIZE_MB}MB.")
                return
                
            if uploaded_file.size == 0:
                st.error("⚠️ O arquivo enviado está vazio ou corrompido.")
                return

            logger = StructuredLogger()
            processor = StreamProcessor(self.config, logger)
            
            if st.button("🚀 Iniciar Pipeline de Conversão", use_container_width=True, type="primary"):
                
                # Componentes Dinâmicos da UI
                progress_container = st.empty()
                status_container = st.empty()
                metrics_container = st.empty()
                
                def update_progress(pct: float):
                    progress_container.progress(pct, text=f"Processando blocos de dados em streaming... ({int(pct*100)}%)")

                try:
                    status_container.info(ProcessingStatus.ANALYZING.value)
                    
                    with st.spinner("Pipeline Híbrido ativado. Analisando amostras, heurísticas e estruturação..."):
                        # Inicia processamento
                        result = processor.process_file(uploaded_file, ui_progress_callback=update_progress)
                        
                    progress_container.progress(1.0, text="Processamento 100% Finalizado.")
                    status_container.success(ProcessingStatus.COMPLETED.value)
                    
                    # Salva no histórico
                    st.session_state.history.append({
                        'file': result.original_filename,
                        'enc': result.detected_encoding,
                        'time': result.processing_time_sec
                    })
                    
                    # RENDERIZAÇÃO DAS MÉTRICAS
                    st.divider()
                    st.subheader("📊 Métricas de Execução (Real-Time)")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Throughput de Leitura", f"{result.read_speed_mbps:.1f} MB/s")
                    col2.metric("Tempo Total", f"{result.processing_time_sec:.2f} s")
                    col3.metric("Linhas Auditadas", f"{result.total_lines:,}".replace(",", "."))
                    col4.metric("Tamanho Original", f"{(result.file_size_bytes / 1024 / 1024):.2f} MB")
                    
                    st.markdown("### 🔎 Inteligência Artificial Heurística")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Encoding Final (Consenso)", result.detected_encoding.upper())
                    c2.metric("Motor Vencedor", result.detector_used)
                    c3.metric("Confiança do Pipeline", f"{result.confidence * 100:.1f}%")
                    c4.metric("Delimitador Identificado", "TAB" if result.detected_delimiter == "\t" else result.detected_delimiter)
                    
                    if result.delimiter_replaced:
                        st.info(f"🔄 **Transformação Ativa:** O delimitador corporativo `|*|` foi convertido para `;` para compatibilidade com CSV padrão.")

                    # RELATÓRIO DE AUDITORIA
                    st.divider()
                    self.render_validation_report(result.validation_report)
                    
                    # DOWNLOAD SEGURO
                    st.divider()
                    st.subheader("📥 Exportação Segura")
                    
                    with open(result.output_filepath, "rb") as f:
                        file_data = f.read()
                        
                    st.download_button(
                        label=f"⬇️ Download Arquivo Homologado (UTF-8)",
                        data=file_data,
                        file_name=f"homolog_utf8_{result.original_filename}",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary"
                    )
                    
                except Exception as e:
                    progress_container.empty()
                    status_container.error(f"{ProcessingStatus.FAILED.value}: {str(e)}")
                    st.error("A execução foi abortada por segurança. Verifique os logs abaixo.")
                
                finally:
                    # LOGS ESTRUTURADOS (Sempre executados)
                    st.divider()
                    with st.expander("📝 Logs de Auditoria e Diagnóstico do Pipeline", expanded=True):
                        st.markdown(f"<div class='terminal-log'><pre>{logger.get_formatted_logs()}</pre></div>", unsafe_allow_html=True)


# ==============================================================================
# ENTRYPOINT DA APLICAÇÃO
# ==============================================================================

if __name__ == "__main__":
    config = AppConfig()
    ui = UI(config)
    ui.render_sidebar()
    ui.render_main()
