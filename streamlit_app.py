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
from typing import Tuple, Dict, List, Optional
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
    APP_VERSION: str = "2.0.1 (Patch: Heurística e Chunking Seguros)"
    SAMPLE_SIZE_BYTES: int = 2 * 1024 * 1024  # 2MB para análise profunda
    SAMPLE_LINES: int = 200                   # Linhas para heurística de variância
    CHUNK_SIZE_BYTES: int = 8 * 1024 * 1024   # 8MB de buffer lógico
    TARGET_ENCODING: str = "utf-8"
    DELIMITERS_TO_CHECK: List[str] = field(default_factory=lambda: [";", ",", "|", "|*|", "\t", " "])
    MAX_FILE_SIZE_MB: int = 5000              # 5GB

@dataclass
class ValidationReport:
    is_valid: bool = True
    invalid_chars_replaced: int = 0
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
# LOGS E UTILITÁRIOS
# ==============================================================================

def secure_filename(filename: str) -> str:
    filename = re.sub(r'[^a-zA-Z0-9_\.-]', '_', os.path.basename(filename))
    return filename if filename else "unnamed_file.txt"

class StructuredLogger:
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
# MOTORES DE DETECÇÃO DE ENCODING
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
                return result.encoding, 0.95, "Charset-Normalizer"
        except Exception:
            pass
        return None, 0.0, "Charset-Normalizer"

class ChardetDetector(AbstractDetector):
    def detect(self, raw_data: bytes) -> Tuple[Optional[str], float, str]:
        try:
            result = chardet.detect(raw_data)
            return result.get('encoding'), result.get('confidence', 0.0), "Chardet"
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
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.detectors = [
            BOMDetector(), CChardetDetector(), CharsetNormalizerDetector(), 
            ChardetDetector(), MagicDetector()
        ]

    def detect(self, raw_data: bytes) -> Tuple[str, float, str]:
        if not raw_data:
            return "utf-8", 1.0, "Fallback (Vazio)"

        candidates = []
        for detector in self.detectors:
            enc, conf, name = detector.detect(raw_data)
            if enc:
                enc = enc.lower()
                if enc == 'ascii': enc = 'utf-8'
                if enc == 'iso-8859-1' and b'\x92' in raw_data: enc = 'windows-1252'
                
                candidates.append({"encoding": enc, "confidence": conf, "source": name})
                self.logger.debug(f"Detector [{name}] propôs: {enc} (Conf: {conf:.2f})")
                if name == "BOM Signature" and conf == 1.0:
                    return enc, 1.0, name

        if not candidates:
            return "utf-8", 0.0, "Fallback Absoluto"

        score_board = collections.defaultdict(float)
        sources_board = collections.defaultdict(list)
        
        for cand in candidates:
            score_board[cand["encoding"]] += cand["confidence"]
            sources_board[cand["encoding"]].append(cand["source"])

        best_enc = max(score_board, key=score_board.get)
        total_score = min(score_board[best_enc] / len(sources_board[best_enc]), 1.0)
        used_detectors = " + ".join(sources_board[best_enc])

        self.logger.info(f"Consenso atingido: {best_enc.upper()} via [{used_detectors}]")
        return best_enc, total_score, used_detectors

# ==============================================================================
# MOTOR DE HEURÍSTICA DE DELIMITADORES (CORRIGIDO)
# ==============================================================================

class ColumnVarianceHeuristic:
    """Implementa detecção inteligente baseada em variância, protegendo delimitadores compostos."""
    def __init__(self, config: AppConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger

    def detect(self, text_sample: str) -> str:
        if not text_sample:
            return ","

        lines = [line.strip() for line in text_sample.splitlines() if line.strip()][:self.config.SAMPLE_LINES]
        if not lines:
            return ","

        candidates = []
        for delim in self.config.DELIMITERS_TO_CHECK:
            col_counts = [len(line.split(delim)) for line in lines]
            avg_cols = sum(col_counts) / len(col_counts)
            
            # Cálculo de variância estatística
            variance = sum((c - avg_cols) ** 2 for c in col_counts) / len(col_counts)
            
            if avg_cols > 1:
                # Armazenamos: variância, comprimento do delimitador, média de colunas, e o próprio delimitador
                candidates.append((variance, len(delim), avg_cols, delim))
                self.logger.debug(f"Analítico - Delimitador '{delim}': Variância={variance:.2f}, Colunas Médias={avg_cols:.1f}")

        if not candidates:
            self.logger.info("Nenhum delimitador consistente encontrado. Padrão assumido: vírgula (,).")
            return ","

        # A MÁGICA DA CORREÇÃO:
        # 1. Menor variância vence (x[0]) -> Quanto mais perto de 0, mais perfeito.
        # 2. Maior comprimento vence (-x[1]) -> Isso garante que "|*|" espanque "|" em caso de empate na variância.
        # 3. Maior número de colunas desempata finais (-x[2]).
        candidates.sort(key=lambda x: (x[0], -x[1], -x[2]))
        
        best_delimiter = candidates[0][3]
        lowest_variance = candidates[0][0]
        col_count = candidates[0][2]
        
        self.logger.info(f"Delimitador Vencedor: '{best_delimiter}' (Variância: {lowest_variance:.2f}, Colunas: ~{int(col_count)})")
        return best_delimiter

# ==============================================================================
# MOTOR DE PROCESSAMENTO E VALIDAÇÃO (CORRIGIDO PARA STREAMING SEGURO)
# ==============================================================================

class StreamProcessor:
    def __init__(self, config: AppConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.encoding_analyzer = EncodingVotingSystem(logger)
        self.delimiter_analyzer = ColumnVarianceHeuristic(config, logger)

    def process_file(self, uploaded_file, ui_progress_callback=None) -> ProcessingResult:
        start_time = time.perf_counter()
        file_size = uploaded_file.size
        filename = secure_filename(uploaded_file.name)
        
        self.logger.info(f"== INÍCIO DO PIPELINE: {filename} ==")

        sample_bytes = uploaded_file.read(self.config.SAMPLE_SIZE_BYTES)
        detected_encoding, confidence, detector_source = self.encoding_analyzer.detect(sample_bytes)
        
        try:
            sample_text = sample_bytes.decode(detected_encoding, errors='replace')
        except Exception as e:
            self.logger.error(f"Falha na decodificação da amostra: {str(e)}")
            sample_text = ""
            
        detected_delimiter = self.delimiter_analyzer.detect(sample_text)
        replace_delimiter = (detected_delimiter == "|*|")

        uploaded_file.seek(0)
        temp_dir = tempfile.gettempdir()
        temp_filepath = os.path.join(temp_dir, f"audit_utf8_{filename}")
        
        total_lines = 0
        processed_bytes = 0
        val_report = ValidationReport()
        val_report.original_bytes = file_size
        
        try:
            with io.TextIOWrapper(uploaded_file, encoding=detected_encoding, errors='replace', newline='') as text_reader:
                with io.open(temp_filepath, 'w', encoding=self.config.TARGET_ENCODING, newline='') as text_writer:
                    
                    while True:
                        # CORREÇÃO CRÍTICA: readlines() respeita quebras de linha.
                        # Isso garante que um "|*|" nunca será cortado ao meio entre duas leituras.
                        lines = text_reader.readlines(self.config.CHUNK_SIZE_BYTES // 4)
                        if not lines:
                            break
                        
                        chunk_text = "".join(lines)
                        
                        if "\ufffd" in chunk_text:
                            val_report.invalid_chars_replaced += chunk_text.count("\ufffd")
                            val_report.data_loss_detected = True
                        
                        if replace_delimiter:
                            chunk_text = chunk_text.replace("|*|", ";")
                            
                        text_writer.write(chunk_text)
                        
                        total_lines += len(lines)
                        
                        if ui_progress_callback:
                            current_pos = uploaded_file.tell()
                            prog_pct = min(current_pos / file_size, 1.0)
                            ui_progress_callback(prog_pct)
                            
        except Exception as e:
            self.logger.critical(f"Exceção crítica durante o processamento de blocos: {str(e)}")
            val_report.is_valid = False
            val_report.issues.append(str(e))
            raise

        val_report.converted_bytes = os.path.getsize(temp_filepath)
        
        end_time = time.perf_counter()
        elapsed_time = max(end_time - start_time, 0.001)
        speed_mbps = (file_size / (1024 * 1024)) / elapsed_time
        
        self.logger.info(f"== TÉRMINO: {elapsed_time:.2f}s | {speed_mbps:.2f} MB/s ==")

        return ProcessingResult(
            original_filename=filename, file_size_bytes=file_size,
            total_lines=total_lines, detected_encoding=detected_encoding,
            confidence=confidence, detector_used=detector_source,
            detected_delimiter=detected_delimiter, delimiter_replaced=replace_delimiter,
            processing_time_sec=elapsed_time, read_speed_mbps=speed_mbps,
            output_filepath=temp_filepath, validation_report=val_report
        )

# ==============================================================================
# UI STREAMLIT
# ==============================================================================

class UI:
    def __init__(self, config: AppConfig):
        self.config = config
        st.set_page_config(page_title=self.config.APP_NAME, page_icon="🛡️", layout="wide")
        self._init_session_state()
        self._apply_custom_css()

    def _init_session_state(self):
        if 'history' not in st.session_state:
            st.session_state.history = []

    def _apply_custom_css(self):
        st.markdown("""
        <style>
            .stProgress > div > div > div > div { background-color: #0078D7; transition: all 0.2s ease; }
            .terminal-log { background-color: #1E1E1E; color: #00FF00; font-family: 'Consolas', monospace; padding: 12px; border-radius: 6px; height: 300px; overflow-y: auto; font-size: 0.85rem; border: 1px solid #333; }
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
                for item in reversed(st.session_state.history[-5:]):
                    st.caption(f"Arquivo: {item['file']} | Enc: {item['enc']} -> UTF-8")
                if st.button("Limpar Histórico", use_container_width=True):
                    st.session_state.history = []
                    st.rerun()

    def render_validation_report(self, report: ValidationReport):
        st.subheader("🛡️ Relatório de Auditoria e Integridade")
        if report.is_valid and not report.data_loss_detected:
            st.success("✅ O arquivo foi convertido mantendo 100% da sua integridade estrutural. Nenhuma perda de dados foi registrada.")
        else:
            st.warning("⚠️ O arquivo foi convertido, mas caracteres corrompidos foram substituídos.")
        col1, col2 = st.columns(2)
        col1.info(f"**Bytes Pós-Conversão:** {report.converted_bytes:,}")
        col2.error(f"**Caracteres Irrecuperáveis:** {report.invalid_chars_replaced:,}")

    def render_main(self):
        st.title(f"🛡️ {self.config.APP_NAME}")
        st.markdown("Solução escalável para higienização e padronização de bases de dados textuais e integrações corporativas.")
        
        uploaded_file = st.file_uploader(
            "Upload de Arquivos (CSV, TXT, DAT, TSV) — Motor Otimizado", 
            type=['csv', 'txt', 'dat', 'log', 'tsv']
        )

        if uploaded_file is not None:
            if uploaded_file.size > (self.config.MAX_FILE_SIZE_MB * 1024 * 1024):
                st.error(f"Tamanho excede o limite de {self.config.MAX_FILE_SIZE_MB}MB.")
                return
            if uploaded_file.size == 0:
                st.error("⚠️ O arquivo enviado está vazio.")
                return

            logger = StructuredLogger()
            processor = StreamProcessor(self.config, logger)
            
            if st.button("🚀 Iniciar Pipeline de Conversão", use_container_width=True, type="primary"):
                progress_container = st.empty()
                status_container = st.empty()
                
                try:
                    status_container.info(ProcessingStatus.ANALYZING.value)
                    with st.spinner("Analisando heurísticas e alinhamento estrutural..."):
                        result = processor.process_file(uploaded_file, ui_progress_callback=lambda p: progress_container.progress(p, text=f"Processando blocos de dados seguros... ({int(p*100)}%)"))
                        
                    progress_container.progress(1.0, text="100% Finalizado.")
                    status_container.success(ProcessingStatus.COMPLETED.value)
                    st.session_state.history.append({'file': result.original_filename, 'enc': result.detected_encoding})
                    
                    st.divider()
                    st.subheader("📊 Métricas de Execução (Real-Time)")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Tempo", f"{result.processing_time_sec:.2f} s")
                    c2.metric("Linhas", f"{result.total_lines:,}".replace(",", "."))
                    c3.metric("Motor", result.detector_used)
                    c4.metric("Delimitador", result.detected_delimiter)
                    
                    if result.delimiter_replaced:
                        st.info("🔄 **Regra Aplicada:** Delimitador corporativo `|*|` substituído por `;`.")

                    st.divider()
                    self.render_validation_report(result.validation_report)
                    
                    st.divider()
                    with open(result.output_filepath, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Arquivo Homologado (UTF-8)",
                            data=f.read(),
                            file_name=f"homolog_utf8_{result.original_filename}",
                            mime="text/csv",
                            use_container_width=True,
                            type="primary"
                        )
                except Exception as e:
                    progress_container.empty()
                    status_container.error(f"Falha: {str(e)}")
                finally:
                    st.divider()
                    with st.expander("📝 Logs de Auditoria do Pipeline", expanded=True):
                        st.markdown(f"<div class='terminal-log'><pre>{logger.get_formatted_logs()}</pre></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    ui = UI(AppConfig())
    ui.render_sidebar()
    ui.render_main()
