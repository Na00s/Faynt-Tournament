#!/usr/bin/env python3
"""Read-only, standard-library checks for a code-only research export.

This is a conservative release screening tool, not a license determination.
Matched text and credential values are never included in findings.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import stat
from typing import Iterable

GAME_SUFFIXES = {'.iso', '.gcm', '.rvz', '.wbfs', '.wad', '.ciso', '.nkit', '.wud', '.wux', '.dol'}
CHECKPOINT_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.h5', '.hdf5', '.pkl', '.pickle', '.npz', '.npy', '.tflite', '.pb'}
NATIVE_SUFFIXES = {'.exe', '.dll', '.dylib', '.so', '.appimage', '.dmg', '.pkg', '.msi', '.deb', '.rpm', '.o', '.a', '.pyc', '.pyo', '.class', '.wasm'}
ARCHIVE_SUFFIXES = {'.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar', '.tgz', '.zst'}
DOC_SUFFIXES = {'.md', '.rst', '.txt', '.adoc', '.html', '.htm', '.tex'}
CODE_SUFFIXES = {'.py', '.sh', '.bash', '.zsh', '.ps1', '.bat', '.cmd', '.js', '.mjs', '.cjs', '.ts', '.tsx', '.jsx', '.yml', '.yaml', '.toml', '.json', '.ini', '.cfg', '.c', '.h', '.cpp', '.m', '.mm'}
VENDOR_PARTS = {'vendor', 'vendors', 'vendored', 'third_party', 'third-party', 'external', 'externals', 'node_modules', 'site-packages', '.venv', 'venv'}
LFS_HEADER = b'version https://git-lfs.github.com/spec/v1'
MAX_TEXT_BYTES = 8 * 1024 * 1024
SECRET_PATTERNS = [
    ('private_key', re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----')),
    ('hf_token', re.compile(r'\bhf_[A-Za-z0-9]{20,}\b')),
    ('github_token', re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b')),
    ('aws_access_key', re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
    ('api_key', re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b')),
    ('slack_token', re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{20,}\b')),
    ('url_credentials', re.compile(r'https?://[^\s/:@]+:[^\s/@]{4,}@')),
]
HOME_PATH = re.compile(r'(?:/Users/[A-Za-z0-9_.-]+/|/home/[A-Za-z0-9_.-]+/|[A-Za-z]:[\\/]+Users[\\/]+[^\s\\/"\']+[\\/])')
ASSIGNMENT_SECRET = re.compile(r'''(?ix)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|hf[_-]?token|github[_-]?token|password|secret[_-]?key|client[_-]?secret)\b\s*[:=]\s*["']([^"'\r\n]{8,})["']''')
PLACEHOLDER = re.compile(r'(?i)(?:example|placeholder|changeme|your[_ -]|replace[_ -]|redacted|dummy|test[_ -]|fake|<|\$\{|\{\{|\*{3}|x{6})')
HAL_NAME = re.compile(r'(?i)(?:^|[._/\\-])hal(?:$|[._/\\-])|^Hal(?:Runtime|LivePolicy|Policy|Checkpoint)$')
HAL_REFERENCE = re.compile(r'(?i)\bhal\b')
DOWNLOAD = re.compile(r'(?i)(?:\b(?:curl|wget|Invoke-WebRequest)\b|\b(?:urlretrieve|urlopen|download(?:_file|_url)?|fetch)\s*\(|\b(?:requests|httpx)\.(?:get|post)\s*\(|\bgit\s+clone\b)')
EMULATOR_OR_GAME = re.compile(r'(?i)(?:dolphin|slippi-launcher|slippi-dolphin|project-slippi|game[_ -]?image|rom[_ -]?url|\.iso(?:\b|[?])|\.gcm(?:\b|[?])|\.rvz(?:\b|[?])|\.wbfs(?:\b|[?]))')
NETWORK_URL = re.compile(r'https?://[^\s"\'<>]+', re.I)
ATTRIBUTION = re.compile(r'(?i)\b(?:copied|adapted|ported|vendored)\s+from|\bderived\s+(?:code|implementation)\s+from|SPDX-License-Identifier:|Copyright\s+(?:\(c\)\s*)?\d{4}')
RUNTIME_TARGET_KEYS = {'module', 'module_name', 'import_module', 'loader', 'loader_class', 'entry_point', 'class_path', '_target_'}
RUNTIME_CODE_KEYS = {'code', 'python_code', 'source_code'}
QUALIFIED_TARGET = re.compile(r'^[A-Za-z_]\w*(?:[./:][A-Za-z_]\w*)*$')


def runtime_symbol(name: str) -> bool:
    return bool(HAL_NAME.search(name))


def runtime_finding(node: ast.AST) -> tuple[str, str] | None:
    """Classify executable syntax without interpreting strings as source code."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        modules = [node.module or ''] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
        symbols = [a.name for a in node.names] if isinstance(node, ast.ImportFrom) else []
        if any(HAL_NAME.search(name) for name in modules) or any(runtime_symbol(name) for name in symbols):
            return 'hal_import_or_loader', 'HAL module or runtime symbol import detected.'
    elif isinstance(node, ast.Call):
        callee = node.func
        # Check the called identifier, not an unrelated metadata object's name.
        symbol = callee.id if isinstance(callee, ast.Name) else callee.attr if isinstance(callee, ast.Attribute) else ''
        root = callee
        while isinstance(root, ast.Attribute):
            root = root.value
        if runtime_symbol(symbol) or (isinstance(root, ast.Name) and root.id.lower() == 'hal'):
            return 'hal_import_or_loader', 'HAL runtime or loader call detected.'
        if symbol in {'import_module', '__import__', 'spec_from_file_location'}:
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            if any(isinstance(arg, ast.Constant) and isinstance(arg.value, str) and HAL_NAME.search(arg.value) for arg in arguments):
                return 'hal_import_or_loader', 'Dynamic HAL import or source loader detected.'
    elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and runtime_symbol(node.name):
        return 'hal_runtime_definition', 'HAL-specific runtime or loader definition detected.'
    return None

@dataclass(frozen=True)
class Finding:
    path: str
    line: int | None
    category: str
    severity: str
    message: str

class Audit:
    def __init__(self, root: Path):
        self.root = root
        self.findings: list[Finding] = []
        self.scanned_files = 0
        self.scanned_bytes = 0
        self._seen: set[tuple] = set()

    def add(self, path: str, category: str, severity: str, message: str, line: int | None = None):
        key = (path, line, category, severity)
        if key not in self._seen:
            self.findings.append(Finding(path, line, category, severity, message))
            self._seen.add(key)

    def scan(self) -> dict:
        if self.root.is_symlink():
            self.add('.', 'symlink', 'hard_failure', 'Export root is a symbolic link; target was not followed.')
        elif not self.root.is_dir():
            self.add('.', 'invalid_root', 'hard_failure', 'Export directory does not exist or is not a directory.')
        else:
            for current, dirs, files in os.walk(self.root, followlinks=False):
                base = Path(current)
                for name in list(dirs):
                    p = base / name; rel = p.relative_to(self.root).as_posix()
                    if p.is_symlink():
                        self.add(rel, 'symlink', 'hard_failure', 'Directory symbolic link was not followed.')
                        dirs.remove(name)
                    elif name == '.git':
                        self.add(rel, 'vcs_metadata', 'review', 'Git metadata is present; excluded from payload inspection.')
                        dirs.remove(name)
                    elif name.lower().endswith('.app'):
                        self.add(rel, 'application_bundle', 'hard_failure', 'Application bundle is prohibited in a code-only export.')
                for name in files:
                    self.scan_file(base / name)
        self.findings.sort(key=lambda f:(f.path, f.line or 0, f.category))
        counts = Counter(f.severity for f in self.findings)
        status = 'fail' if counts['hard_failure'] else 'review' if counts['review'] else 'pass'
        return {
            'schema_version': 'code_export_audit.v1', 'target': self.root.name,
            'status': status,
            'summary': {'files_scanned': self.scanned_files, 'bytes_scanned': self.scanned_bytes,
                        'hard_failures': counts['hard_failure'], 'review_findings': counts['review']},
            'findings': [asdict(f) for f in self.findings],
            'limitations': [
                'Heuristic screening does not determine copyright ownership, license compatibility, or legal clearance.',
                'Archives and unrecognized binary files require review; their embedded content is not unpacked.',
                'Dynamic or obfuscated imports, generated download URLs, and unknown secret formats can evade detection.',
                'Symbolic-link targets and Git metadata are not followed or scanned.',
                f'Text files larger than {MAX_TEXT_BYTES} bytes receive a hard incomplete-scan finding.',
            ],
        }

    def scan_file(self, path: Path):
        rel = path.relative_to(self.root).as_posix()
        try:
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                self.add(rel, 'symlink', 'hard_failure', 'File symbolic link was not followed.'); return
            if not stat.S_ISREG(mode):
                self.add(rel, 'special_file', 'hard_failure', 'Non-regular filesystem entry is prohibited.'); return
            size = path.stat().st_size
            with path.open('rb') as stream:
                data = stream.read(MAX_TEXT_BYTES + 1)
        except OSError:
            self.add(rel, 'unreadable_file', 'hard_failure', 'File could not be read; export is incompletely scanned.'); return
        self.scanned_files += 1; self.scanned_bytes += min(size, len(data))
        suffix = path.suffix.lower()
        if suffix in GAME_SUFFIXES:
            self.add(rel, 'game_image', 'hard_failure', 'File extension identifies a game image or game executable.')
        if suffix in CHECKPOINT_SUFFIXES or re.search(r'\.data-\d{5}-of-\d{5}$', path.name):
            self.add(rel, 'checkpoint_payload', 'hard_failure', 'Serialized model, checkpoint, or numeric-array payload is prohibited.')
        if suffix in NATIVE_SUFFIXES:
            self.add(rel, 'binary_or_installer', 'hard_failure', 'Native, bytecode, library, or installer payload is prohibited.')
        if suffix in ARCHIVE_SUFFIXES:
            self.add(rel, 'archive_payload', 'review', 'Archive requires separate content review before release.')
        magic = data[:4]
        if magic == b'\x7fELF' or data[:2] == b'MZ' or magic in {b'\xfe\xed\xfa\xce', b'\xfe\xed\xfa\xcf', b'\xce\xfa\xed\xfe', b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca'}:
            self.add(rel, 'native_binary_magic', 'hard_failure', 'Executable or compiled binary signature detected.')
        # GameCube and Wii disc header magic, including images without a suffix.
        if data[0x1c:0x20] == b'\xc2\x33\x9f\x3d' or data[0x18:0x1c] == b'\x5d\x1c\x9e\xa3':
            self.add(rel, 'game_image_magic', 'hard_failure', 'Console game-image signature detected.')
        if data.lstrip().startswith(LFS_HEADER):
            self.add(rel, 'git_lfs_pointer', 'hard_failure', 'Git LFS pointer detected; referenced payload is outside this code-only export.')
        if any(part.lower() in VENDOR_PARTS for part in path.relative_to(self.root).parts[:-1]) and suffix not in DOC_SUFFIXES and not path.name.lower().startswith(('license', 'copying', 'notice')):
            self.add(rel, 'vendored_payload_path', 'hard_failure', 'Code or asset appears within a dependency or vendor directory.')
        if any(part.lower().startswith(('dolphin', 'slippi-dolphin')) and part.lower().endswith('.app') for part in path.parts):
            self.add(rel, 'emulator_bundle_payload', 'hard_failure', 'File is inside an emulator application bundle.')
        if b'\0' in data[:8192]:
            if suffix not in GAME_SUFFIXES | CHECKPOINT_SUFFIXES | NATIVE_SUFFIXES | ARCHIVE_SUFFIXES and not any(f.path==rel and f.severity=='hard_failure' for f in self.findings):
                self.add(rel, 'unclassified_binary', 'review', 'Binary data requires provenance and payload review.')
            return
        try: text = data.decode('utf-8')
        except UnicodeDecodeError:
            self.add(rel, 'unclassified_binary', 'review', 'Non-UTF-8 content requires provenance and payload review.'); return
        if size > MAX_TEXT_BYTES:
            self.add(rel, 'text_scan_truncated', 'hard_failure', 'Text exceeds scanner size limit; full text has not been inspected.')
        doc = suffix in DOC_SUFFIXES or path.name.lower().startswith(('license', 'copying', 'notice'))
        lines = text.splitlines()
        factual_lines = set()
        if suffix == '.py':
            try:
                for node in ast.walk(ast.parse(text)):
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        factual_lines.update(range(node.lineno, node.end_lineno + 1))
            except SyntaxError:
                pass  # scan_python emits the incomplete-inspection finding.
        for number, line in enumerate(lines, 1):
            for label, pattern in SECRET_PATTERNS:
                if pattern.search(line): self.add(rel, 'possible_secret', 'hard_failure', 'Potential credential detected ('+label+'); matched value withheld.', number)
            for match in ASSIGNMENT_SECRET.finditer(line):
                if not PLACEHOLDER.search(match.group(1)):
                    self.add(rel, 'possible_secret', 'hard_failure', 'Potential credential assignment detected; matched value withheld.', number)
            if HOME_PATH.search(line):
                self.add(rel, 'embedded_home_path', 'hard_failure', 'User-specific absolute home-directory path detected; path value withheld.', number)
            if ATTRIBUTION.search(line) and not doc:
                self.add(rel, 'source_provenance', 'review', 'Source attribution or license marker needs comparison with the export provenance inventory.', number)
            if doc or number in factual_lines or line.lstrip().startswith(('#', '//', '*', '<!--')): continue
            if re.match(r'\s*\[(?:hal)(?:[.\]])', line, re.I):
                self.add(rel, 'hal_runtime_config', 'hard_failure', 'HAL runtime configuration section detected.', number)
            if suffix not in {'.py', '.json'} and re.search(r'(?i)\b(?:import|from|require\s*\(|import_module\s*\()[^\n]*\bhal\b', line):
                self.add(rel, 'hal_import_or_loader', 'hard_failure', 'HAL import or loader reference detected.', number)
            if not line.lstrip().startswith(('#', '//', '*', '<!--')):
                context = '\n'.join(lines[max(0,number-4):number+3])
                if suffix != '.py' and DOWNLOAD.search(NETWORK_URL.sub('', line)) and EMULATOR_OR_GAME.search(context):
                    self.add(rel, 'emulator_or_game_download', 'hard_failure', 'Executable code contains an emulator or game acquisition command.', number)
                elif '--fetch-source' in line and EMULATOR_OR_GAME.search(text):
                    self.add(rel, 'emulator_source_acquisition_option', 'review', 'Code exposes a source-acquisition option near emulator references; inspect the invoked helper.', number)
                elif NETWORK_URL.search(line) and EMULATOR_OR_GAME.search(line):
                    self.add(rel, 'emulator_or_game_url', 'review', 'Code contains an emulator or game URL; determine whether it is metadata or an acquisition path.', number)
        if suffix == '.py': self.scan_python(rel, text)
        elif suffix == '.json': self.scan_json(rel, text)

    def scan_json(self, rel: str, text: str):
        try:
            metadata = json.loads(text)
        except (ValueError, RecursionError):
            self.add(rel, 'json_parse_failure', 'review', 'JSON metadata could not be parsed; inspect runtime references manually.')
            return
        lines = text.splitlines()

        def visit(value, key=''):
            if isinstance(value, dict):
                for field, child in value.items(): visit(child, field)
            elif isinstance(value, list):
                for child in value: visit(child, key)
            elif isinstance(value, str) and (HAL_NAME.search(value) or HAL_REFERENCE.search(value)):
                encoded = {json.dumps(value), json.dumps(value, ensure_ascii=False)}
                number = next((i for i, line in enumerate(lines, 1) if any(item in line for item in encoded)), None)
                explicit_loader = key.lower() in RUNTIME_TARGET_KEYS and QUALIFIED_TARGET.fullmatch(value) is not None
                if key.lower() in RUNTIME_CODE_KEYS:
                    try:
                        explicit_loader = any(runtime_finding(node) for node in ast.walk(ast.parse(value)))
                    except SyntaxError:
                        pass
                if explicit_loader:
                    self.add(rel, 'hal_import_or_loader', 'hard_failure', 'Explicit serialized HAL code or runtime loader target detected.', number)
                else:
                    self.add(rel, 'hal_metadata_reference', 'review', 'Metadata refers to HAL; review the historical description or runtime context.', number)

        visit(metadata)

    def scan_python(self, rel: str, text: str):
        try: tree = ast.parse(text)
        except SyntaxError as exc:
            self.add(rel, 'python_parse_failure', 'hard_failure', 'Python source could not be parsed; code inspection is incomplete.', exc.lineno); return
        for node in ast.walk(tree):
            finding = runtime_finding(node)
            if finding:
                self.add(rel, finding[0], 'hard_failure', finding[1], node.lineno)
            if isinstance(node, ast.Call):
                func = ast.unparse(node.func)
                if func.rsplit('.', 1)[-1] in {'run', 'Popen', 'check_call', 'check_output', 'system', 'run_command', 'run_commands', 'urlretrieve', 'urlopen', 'download_file', 'download', 'fetch'} or func in {'requests.get', 'requests.post', 'httpx.get'}:
                    segment = ast.get_source_segment(text, node) or ''
                    if DOWNLOAD.search(NETWORK_URL.sub('', segment)) and EMULATOR_OR_GAME.search(segment):
                        self.add(rel, 'emulator_or_game_download', 'hard_failure', 'Executable call contains an emulator or game acquisition command.', node.lineno)


def audit_export(root: str | Path) -> dict:
    return Audit(Path(root)).scan()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path, help='Export directories to inspect without modifying them.')
    parser.add_argument('--output', type=Path, help='Write JSON here; default is stdout.')
    parser.add_argument('--fail-on-review', action='store_true', help='Exit nonzero when review findings remain.')
    args = parser.parse_args(argv)
    reports = [audit_export(root) for root in args.roots]
    report = reports[0] if len(reports) == 1 else {'schema_version':'code_export_audit_bundle.v1', 'exports':reports}
    rendered = json.dumps(report, indent=2, ensure_ascii=True)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(rendered)
    else: print(rendered, end='')
    return int(any(r['summary']['hard_failures'] or (args.fail_on_review and r['summary']['review_findings']) for r in reports))

if __name__ == '__main__':
    raise SystemExit(main())
