'''
HOOKS - standard conversation hook implementations for content packs

Content pack `code` fields used to carry copies of this logic (memory assembly,
director guidance, ops-diff consolidation, checkpointing, dynamic openers).
They now collapse to a shim:

    from hive.hooks import make_standard_hooks
    _hooks = make_standard_hooks()
    pre_process = _hooks['pre_process']
    post_process = _hooks['post_process']
    complete_handler = _hooks['complete_handler']

Packs can still layer custom behavior on top (wrap the returned functions or
define their own hooks) - this module is the tested default, not a cage.
'''
import json
import logging
import threading
import time

from .director import build_directives
from .memory import apply_memory_ops, assemble_memory, list_fragments_compact

logger = logging.getLogger(__name__)

CONSOLIDATION_MODEL = 'gpt-5.6-terra'
BANKS = 'profile, magic_tricks, people, places, goals, likes, running_jokes, other'


# Per-volley context: memory assembly + director guidance into volley.local_data,
# for the prompt template to render. Never raises.
def standard_pre_process(volley, session):
    try:
        speech = volley.request.get('speech', '') or ''
        volley.local_data['memory_context'] = assemble_memory(volley.device_id, speech)
    except Exception:
        volley.local_data.setdefault('memory_context', '')
    try:
        volley.local_data['director_notes'] = build_directives(volley, session)
    except Exception:
        volley.local_data.setdefault('director_notes', '')


# One structured extraction pass: current fragments + transcript -> op diffs.
# include_episode=True at session end (adds an episode record); False at
# mid-session checkpoints (refreshes the rolling convo_summary instead).
def consolidate(volley, session, include_episode, guidance=''):
    device_id = volley.device_id
    frags = list_fragments_compact(device_id)
    prompt = ('You maintain the long-term memory of Moxie, a robot friend of a child. '
              'CURRENT MEMORY FRAGMENTS are listed below, one per line as: bank | key | text. '
              'Given the conversation transcript that follows, reply with ONLY a JSON array of '
              'operations to update memory: [{"op": "add"|"update"|"retire", "bank": str, '
              '"key": str, "text": str, "confidence": float}]. Banks: ' + BANKS + '. '
              'Keys are short stable slugs (e.g. trick:french-drop, person:sam). Use update to '
              'revise an existing key, add for new memories, retire for facts that are now wrong '
              'or obsolete. Record what HAPPENED, neutrally - NEVER store advice, warnings, or safety reminders that Moxie gave as memories. Only store things the child said, did, or clearly confirmed with real enthusiasm - never topics Moxie itself introduced that got only polite agreement. ' + (guidance + ' ' if guidance else '') +
              'Only emit operations for real changes; an empty array is fine.'
              '\n\nCURRENT MEMORY FRAGMENTS:\n' + (frags or '(none yet)'))
    raw = session.summarize(model=CONSOLIDATION_MODEL, prompt_base=prompt, max_tokens=2000)
    try:
        cleaned = raw
        if '```' in cleaned:
            cleaned = cleaned.split('```')[1]
            if cleaned.startswith('json'):
                cleaned = cleaned[4:]
        ops = json.loads(cleaned)
        if isinstance(ops, list) and ops:
            result = apply_memory_ops(device_id, ops)
            logger.info(f'consolidation {device_id}: {result} from {len(ops)} ops')
    except Exception:
        logger.warning(f'consolidation for {device_id}: unparseable extractor output')
    if include_episode:
        summary = session.summarize(max_tokens=300)
        apply_memory_ops(device_id, [{'op': 'add', 'bank': 'episodes',
                                      'key': 'ep:' + str(int(time.time())),
                                      'text': summary, 'confidence': 0.6}])
    else:
        # compaction: keep trimmed-away conversation available to the prompt
        volley.local_data['convo_summary'] = session.summarize(max_tokens=200)


# Opener styles rotate so greetings do not converge on one memory or one shape.
# Only some styles feature a specific memory; those draw a novelty-weighted anchor
# (least-recently-surfaced) which is then logged in the habituation ledger.
OPENER_STYLES = [
    'Give a warm open greeting and ask how they are doing or what is new. Do not mention any specific memory.',
    'Ask ONE zoomed-out question about one of their interest areas from the gist below (e.g. "have you learned any new tricks?" or "played with any animals lately?") - no fine details.',
    'Playfully reference the SPECIFIC MEMORY below and ask a light question about it.',
    'Share a tiny fun thought, wondering, or gentle joke inspired by the SPECIFIC MEMORY below.',
    'Suggest exploring or imagining something brand new together today, connected loosely to the gist below.',
]


# Replace the <opener> marker with a freshly generated, memory-aware greeting.
def generate_opener(volley, session):
    import random

    from django.utils import timezone

    from .memory import assemble_memory, pick_opener_anchor
    hour = timezone.localtime().hour
    part = 'morning' if hour < 12 else ('afternoon' if hour < 18 else 'evening')
    nick = (volley.config.get('child_pii') or {}).get('nickname') or 'your friend'
    style = random.choice(OPENER_STYLES)
    anchor = None
    if 'SPECIFIC MEMORY' in style:
        anchor = pick_opener_anchor(volley.device_id)
        if not anchor:
            style = OPENER_STYLES[0]
    gist = assemble_memory(volley.device_id, include_details=False)
    prompt = ('You are Moxie, a friendly robot from the Daddy Robotics Laboratory. '
              'Write ONE short, warm opening line (under 25 words) to start a chat with '
              'your friend ' + nick + '. It is ' + part + '. ' + style +
              ' Never use more than one exclamation point.'
              + ('\n\nSPECIFIC MEMORY:\n' + anchor if anchor else '')
              + '\n\nGIST OF WHAT YOU KNOW:\n' + gist)
    opener = session.summarize(append_transcript=False, max_tokens=80, prompt_base=prompt)
    return opener.strip().strip('"')


def make_standard_hooks(min_volleys=4, checkpoint_volleys=40, guidance=''):
    def pre_process(volley, session):
        standard_pre_process(volley, session)

    def complete_handler(volley, session):
        if getattr(session, 'total_volleys', 0) < min_volleys:
            return
        consolidate(volley, session, True, guidance)

    def post_process(volley, session):
        text = volley.response.get('output', {}).get('text', '')
        if '<opener>' in text:
            try:
                volley.set_output(generate_opener(volley, session), None)
            except Exception as e:
                logger.warning(f'dynamic opener failed: {e}')
                volley.set_output(text.replace('<opener>', 'Hello, my friend! What should we talk about?'), None)
        tv = getattr(session, 'total_volleys', 0)
        last = volley.local_data.get('mem_ckpt', 0)
        if tv - last >= checkpoint_volleys:
            volley.local_data['mem_ckpt'] = tv
            threading.Thread(target=consolidate, args=(volley, session, False, guidance),
                             daemon=True).start()

    return {'pre_process': pre_process, 'post_process': post_process,
            'complete_handler': complete_handler}
