"""Legacy binary Word (.doc) -> .docx conversion helpers.

Docling cannot open the legacy binary ``.doc`` format by itself: its MS Word
backend shells out to LibreOffice (``soffice``) to convert ``.doc`` -> ``.docx``
first. On Windows machines that have MS Office installed, converting through
Word COM (pywin32) is faster and needs no extra software; LibreOffice is used
as a fallback when Word is unavailable.
"""

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

from core.exceptions import ParseError
from core.logging import get_logger

logger = get_logger(__name__)

# 同一时刻只允许一个 Word COM 实例做转换,避免 Office 自动化进程互相干扰
_WORD_LOCK = threading.Lock()

# wdFormatXMLDocument(.docx)
_SAVE_AS_DOCX = 12


def _find_libreoffice() -> Optional[str]:
    """Return a usable soffice executable path, or None."""
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found

    # Windows 默认安装位置(通常不在 PATH 上)
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        / "LibreOffice"
        / "program"
        / "soffice.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
        / "LibreOffice"
        / "program"
        / "soffice.exe",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs"
        / "LibreOffice"
        / "program"
        / "soffice.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _convert_doc_to_docx_with_word(source: Path, target: Path) -> bool:
    """Convert via MS Word COM (pywin32). Returns True on success."""
    try:
        import pythoncom
        from win32com.client import DispatchEx
    except Exception:  # pywin32 未安装
        logger.warning("pywin32 不可用,跳过 Word 转换: %s", source)
        return False

    word = None
    with _WORD_LOCK:
        pythoncom.CoInitialize()
        try:
            word = DispatchEx("Word.Application")
            word.Visible = False
            try:
                word.DisplayAlerts = 0  # wdAlertsNone
            except Exception:
                pass
            doc = word.Documents.Open(
                FileName=str(source),
                ReadOnly=True,
                AddToRecentFiles=False,
                ConfirmConversions=False,
            )
            try:
                doc.SaveAs2(FileName=str(target), FileFormat=_SAVE_AS_DOCX)
            except AttributeError:  # 旧版 Word
                doc.SaveAs(FileName=str(target), FileFormat=_SAVE_AS_DOCX)
            doc.Close(SaveChanges=False)
            return target.is_file() and target.stat().st_size > 0
        except Exception:
            logger.exception("Word 转换 .doc 失败: %s", source)
            return False
        finally:
            try:
                if word is not None:
                    word.Quit()
            except Exception:
                pass
            pythoncom.CoUninitialize()


def _convert_doc_to_docx_with_libreoffice(source: Path, target: Path) -> bool:
    """Convert via LibreOffice headless. Returns True on success."""
    soffice = _find_libreoffice()
    if soffice is None:
        return False
    try:
        proc = subprocess.run(
            [
                soffice,
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                str(target.parent),
                str(source),
            ],
            capture_output=True,
            timeout=180,
        )
    except Exception:
        logger.exception("LibreOffice 转换 .doc 失败: %s", source)
        return False

    if proc.returncode != 0:
        logger.warning(
            "LibreOffice 转换失败(%s): %s",
            source,
            proc.stderr.decode("utf-8", errors="ignore")[-300:],
        )
        return False
    return target.is_file() and target.stat().st_size > 0


def convert_doc_to_docx(source: Path) -> Path:
    """Convert a legacy ``.doc`` file into a ``.docx`` next to it.

    Tries MS Word COM first, then LibreOffice. Raises ParseError with a clear
    message when neither is available or the conversion fails.

    Returns the path of the produced .docx file.
    """
    if source.suffix.lower() != ".doc":
        raise ParseError(message=f"仅支持 .doc 文件,实际为:{source.suffix}")

    target = source.with_suffix(".docx")
    converted = _convert_doc_to_docx_with_word(source, target)
    if not converted:
        converted = _convert_doc_to_docx_with_libreoffice(source, target)

    if not converted:
        raise ParseError(
            message=(
                "无法转换 .doc 文件:服务器上未找到可用的 Microsoft Word"
                "(需安装 pywin32)或 LibreOffice,请安装其一,"
                "或将文件另存为 .docx 后重新上传"
            )
        )

    logger.info(".doc 已转为 .docx: %s -> %s", source.name, target.name)
    return target
