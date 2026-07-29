'''
MEMORY - Fragment-based long-term memory (memory v3)

Memory is stored as per-device rows (MemoryFragment) grouped into named banks,
updated by DIFFS (operation lists emitted by an LLM extractor) rather than
whole-state rewrites, and assembled into prompt context per volley within a
budget. Ground truth remains ChatTranscript; fragments are derived data.

Both entry points are safe to call from worker threads (DB access is wrapped
in run_db_atomic) and are intended to be used from conversation `code` hooks:

    from hive.memory import apply_memory_ops, assemble_memory
'''
import logging
import re

from .models import MemoryFragment, MoxieDevice
from .mqtt.util import run_db_atomic

logger = logging.getLogger(__name__)

_VALID_OPS = ('add', 'update', 'retire', 'restore')
_WORD_RE = re.compile(r"[a-z0-9']+")
# words too common to signal relevance
_STOP = set('the a an and or but so to of in on at for with is are was were be been do does did '
            'you your i my me we our he she it they them this that these those what when where '
            'who how why not no yes have has had can could will would like about'.split())


def _apply_ops_atomic(device_id, ops):
    device = MoxieDevice.objects.filter(device_id=device_id).first()
    if not device:
        return {'error': f'unknown device {device_id}'}
    applied, skipped = 0, 0
    for op in ops:
        if not isinstance(op, dict) or op.get('op') not in _VALID_OPS \
                or not op.get('bank') or not op.get('key'):
            skipped += 1
            continue
        kind = op['op']
        bank, key = str(op['bank'])[:40], str(op['key'])[:120].strip().casefold()
        frag = MemoryFragment.objects.filter(device=device, bank=bank, key=key).first()
        if kind in ('add', 'update'):
            if not op.get('text'):
                skipped += 1
                continue
            if frag:
                frag.text = str(op['text'])
                frag.times_seen += 1
                if 'confidence' in op:
                    try:
                        frag.confidence = min(1.0, max(0.0, float(op['confidence'])))
                    except (TypeError, ValueError):
                        pass
                frag.active = True
                frag.save()
            else:
                try:
                    conf = min(1.0, max(0.0, float(op.get('confidence', 0.5))))
                except (TypeError, ValueError):
                    conf = 0.5
                MemoryFragment.objects.create(device=device, bank=bank, key=key,
                                              text=str(op['text']), confidence=conf)
            applied += 1
        elif kind in ('retire', 'restore'):
            if frag:
                frag.active = (kind == 'restore')
                frag.save()
                applied += 1
            else:
                skipped += 1
    return {'applied': applied, 'skipped': skipped}


# Apply an extractor-emitted operation list to a device's memory fragments.
# ops: [{op: add|update|retire|restore, bank, key, text?, confidence?}, ...]
# Invalid operations are skipped, never fatal. Returns {'applied': n, 'skipped': m}.
def apply_memory_ops(device_id, ops):
    try:
        return run_db_atomic(_apply_ops_atomic, device_id, list(ops or []))
    except Exception as e:
        logger.warning(f'apply_memory_ops failed: {e}')
        return {'error': str(e)}


def _words(text):
    return set(_WORD_RE.findall(text.casefold())) - _STOP


def _assemble_atomic(device_id, utterance, budget_chars, core_banks):
    device = MoxieDevice.objects.filter(device_id=device_id).first()
    if not device:
        return ''
    frags = list(MemoryFragment.objects.filter(device=device, active=True))
    if not frags:
        return ''
    newest = max(f.updated for f in frags)
    spoken = _words(utterance or '')

    def score(f):
        overlap = len(spoken & _words(f.text + ' ' + f.key)) if spoken else 0
        recency = 1.0 - min(1.0, (newest - f.updated).total_seconds() / (30 * 86400))
        return overlap * 2.0 + f.confidence + recency * 0.5

    core = [f for f in frags if f.bank in core_banks]
    rest = sorted((f for f in frags if f.bank not in core_banks), key=score, reverse=True)
    picked, used = [], 0
    for f in core + rest:
        cost = len(f.text) + len(f.bank) + 4
        if f not in core and used + cost > budget_chars:
            continue
        picked.append(f)
        used += cost
    by_bank = {}
    for f in picked:
        by_bank.setdefault(f.bank, []).append(f.text)
    lines = []
    for bank in sorted(by_bank, key=lambda b: (b not in core_banks, b)):
        lines.append(bank.replace('_', ' ').upper() + ':')
        lines.extend('- ' + t for t in by_bank[bank])
        lines.append('')
    return '\n'.join(lines).strip()


# Build the memory context block for a prompt: core banks always included,
# remaining fragments ranked by lexical relevance to the utterance, confidence,
# and recency, within a character budget (~4 chars/token).
def assemble_memory(device_id, utterance='', budget_chars=6000,
                    core_banks=('profile', 'magic_tricks', 'goals')):
    try:
        return run_db_atomic(_assemble_atomic, device_id, utterance, budget_chars, tuple(core_banks))
    except Exception as e:
        logger.warning(f'assemble_memory failed: {e}')
        return ''
