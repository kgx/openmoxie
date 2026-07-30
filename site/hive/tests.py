import json
import random
import tempfile

import numpy
from django.core.management import CommandError, call_command
from django.test import TestCase

from . import memory as memory_mod
from .memory import apply_memory_ops, assemble_memory
from .models import (ChatTranscript, MemoryFragment, MoxieDevice, MoxieSchedule,
                     PersistentData, SinglePromptChat)
from .mqtt.conversations import ChatSession, inference_token_params
from .mqtt.robot_data import RobotData
from .mqtt.scheduler import distribute_elements, expand_schedule, ransac_select
from .mqtt.volley import Volley


class ChatSessionHistoryTests(TestCase):
    def test_history_trims_in_place(self):
        s = ChatSession(max_history=4)
        for i in range(6):
            s.add_history('user' if i % 2 == 0 else 'assistant', f'line {i}')
        self.assertEqual(len(s._history), 4)
        self.assertEqual(s._history[-1]['content'], 'line 5')
        self.assertEqual(s._history[0]['content'], 'line 2')

    def test_volleys_counted_only_for_session_history(self):
        s = ChatSession()
        s.add_history('user', 'hello')
        self.assertEqual(s.total_volleys, 1)
        clone = []
        s.add_history('user', 'aside', clone)
        self.assertEqual(s.total_volleys, 1)

    def test_empty_clone_history_not_redirected_to_session(self):
        # an empty cloned list must receive the write; the session history stays empty
        s = ChatSession()
        clone = []
        s.add_history('user', 'hello', clone)
        self.assertEqual(clone, [{'role': 'user', 'content': 'hello'}])
        self.assertTrue(s.is_empty())
        self.assertEqual(s.total_volleys, 0)

    def test_same_role_messages_concatenate(self):
        s = ChatSession()
        s.add_history('user', 'hello')
        s.add_history('user', 'there')
        self.assertEqual(len(s._history), 1)
        self.assertEqual(s._history[0]['content'], 'hello there')


class InferenceTokenParamsTests(TestCase):
    def test_legacy_models_use_max_tokens(self):
        self.assertEqual(inference_token_params('gpt-4.1-mini', 70), {'max_tokens': 70})
        self.assertEqual(inference_token_params('gpt-3.5-turbo', 70), {'max_tokens': 70})

    def test_gpt5_and_o_series_use_completion_tokens_without_reasoning(self):
        for model in ('gpt-5.6-luna', 'gpt-5-mini', 'o3-mini'):
            self.assertEqual(inference_token_params(model, 70),
                             {'max_completion_tokens': 70, 'reasoning_effort': 'none'})


class VolleyTests(TestCase):
    def make_volley(self, speech='hi'):
        request = {'command': 'continue', 'speech': speech, 'backend': 'router', 'event_id': 'e1',
                   'recommend': {'exits': [{'module_id': 'NEXTMOD', 'content_id': 'NEXTCID'}]}}
        return Volley(request, device_id='d_test')

    def test_exit_tag_becomes_launch_to_recommended_exit(self):
        v = self.make_volley()
        v.set_output('Goodbye friend <exit>', None)
        v.ingest_action_tags()
        self.assertNotIn('<', v.response['output']['text'])
        action = v.response['response_actions'][0]
        self.assertEqual(action['action'], 'launch')
        self.assertEqual(action['module_id'], 'NEXTMOD')
        self.assertEqual(action['content_id'], 'NEXTCID')

    def test_launch_tag_with_module_and_content(self):
        v = self.make_volley()
        v.set_output('Lets play <launch:STORY:tale1>', None)
        v.ingest_action_tags()
        action = v.response['response_actions'][0]
        self.assertEqual(action['action'], 'launch')
        self.assertEqual(action['module_id'], 'STORY')
        self.assertEqual(action['content_id'], 'tale1')

    def test_set_output_updates_both_action_records(self):
        v = self.make_volley()
        v.set_output('hello', None, output_type='GLOBAL_COMMAND')
        self.assertEqual(v.response['response_action']['output_type'], 'GLOBAL_COMMAND')
        self.assertEqual(v.response['response_actions'][0]['output_type'], 'GLOBAL_COMMAND')


class SchedulerTests(TestCase):
    def setUp(self):
        random.seed(1234)
        numpy.random.seed(1234)

    def test_expand_schedule_generates_and_strips_generate_key(self):
        schedule = {'provided_schedule': [{'module_id': 'OPENMOXIE_CHAT', 'content_id': 'short'}],
                    'generate': {'chat_count': 2, 'module_count': 3,
                                 'chat_modules': [{'module_id': 'OPENMOXIE_CHAT', 'content_id': 'short'}],
                                 'extra_modules': [], 'excluded_module_ids': []},
                    'chat_request': {'module_id': 'OPENMOXIE_CHAT', 'content_id': 'default'}}
        expanded = expand_schedule(schedule, 'd_nonexistent')
        self.assertNotIn('generate', expanded)
        self.assertEqual(len(expanded['provided_schedule']), 1 + 3 + 2)
        # original schedule object must keep its generate block for the next expansion
        self.assertIn('generate', schedule)

    def test_ransac_select_bounds(self):
        pool = [{'module_id': f'M{i}', 'category': f'C{i % 3}'} for i in range(10)]
        self.assertEqual(len(ransac_select(pool, 4)), 4)
        self.assertEqual(len(ransac_select(pool, 99)), 10)

    def test_distribute_elements_preserves_all_items(self):
        merged = distribute_elements([1, 2, 3, 4, 5], ['a', 'b'])
        self.assertEqual(len(merged), 7)
        self.assertEqual(set(merged), {1, 2, 3, 4, 5, 'a', 'b'})


class PersistentDataFlushTests(TestCase):
    def test_save_persist_flushes_connected_device(self):
        device = MoxieDevice.objects.create(device_id='d_test')
        pdata, _ = PersistentData.objects.get_or_create(device=device, defaults={'data': {}})
        rd = RobotData()
        rd._robot_map['d_test'] = {'persistent_data': pdata}
        pdata.data['memory_chat'] = {'facts': 'likes dinosaurs'}
        rd.save_persist('d_test')
        self.assertEqual(PersistentData.objects.get(device=device).data['memory_chat']['facts'],
                         'likes dinosaurs')

    def test_save_persist_ignores_unknown_device(self):
        RobotData().save_persist('d_never_connected')


class LoadContentTests(TestCase):
    def write_pack(self, pack):
        f = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
        json.dump(pack, f)
        f.close()
        return f.name

    def test_loads_conversations_and_is_idempotent(self):
        pack = {'conversations': [{'name': 'Test Chat', 'module_id': 'TESTMOD', 'content_id': 'default',
                                   'prompt': 'You are a test robot.', 'model': 'gpt-5.6-luna',
                                   'source_version': 100}]}
        path = self.write_pack(pack)
        call_command('load_content', path)
        call_command('load_content', path)
        self.assertEqual(SinglePromptChat.objects.filter(module_id='TESTMOD').count(), 1)
        chat = SinglePromptChat.objects.get(module_id='TESTMOD')
        self.assertEqual(chat.model, 'gpt-5.6-luna')
        self.assertEqual(chat.source_version, 100)

    def test_high_source_version_survives_init_data(self):
        pack = {'conversations': [{'name': 'Custom Long', 'module_id': 'OPENMOXIE_CHAT',
                                   'content_id': 'default', 'prompt': 'Daddy Robotics prompt',
                                   'source_version': 100}]}
        call_command('init_data')
        call_command('load_content', self.write_pack(pack))
        call_command('init_data')  # factory version 2 < 100 -> must not clobber
        chat = SinglePromptChat.objects.get(module_id='OPENMOXIE_CHAT', content_id='default')
        self.assertEqual(chat.prompt, 'Daddy Robotics prompt')

    def test_persist_seed_merges_and_replaces(self):
        device = MoxieDevice.objects.create(device_id='d_seed')
        PersistentData.objects.create(device=device, data={'memory_v2': {'facts': ['old fact']},
                                                           'other': 'keep me'})
        merge_pack = {'persist': [{'device_id': 'd_seed',
                                   'data': {'memory_v2': {'profile': 'aspiring magician'}}}]}
        call_command('load_content', self.write_pack(merge_pack))
        data = PersistentData.objects.get(device=device).data
        self.assertEqual(data['other'], 'keep me')
        self.assertEqual(data['memory_v2']['facts'], ['old fact'])
        self.assertEqual(data['memory_v2']['profile'], 'aspiring magician')

        replace_pack = {'persist': [{'device_id': 'd_seed', 'replace': True,
                                     'data': {'memory_v2': {'fresh': True}}}]}
        call_command('load_content', self.write_pack(replace_pack))
        self.assertEqual(PersistentData.objects.get(device=device).data, {'memory_v2': {'fresh': True}})

    def test_persist_seed_unknown_device_fails(self):
        pack = {'persist': [{'device_id': 'd_ghost', 'data': {}}]}
        with self.assertRaises(CommandError):
            call_command('load_content', self.write_pack(pack))


class TranscriptTests(TestCase):
    def test_notify_lines_persisted_with_roles(self):
        from types import SimpleNamespace
        from .mqtt.moxie_remote_chat import RemoteChat
        MoxieDevice.objects.create(device_id='d_test')
        rc = RemoteChat(SimpleNamespace())
        rcr = {'module_id': 'OPENMOXIE_CHAT', 'content_id': 'memory2',
               'speech': 'That is a great trick, George!',
               'extra_lines': [{'context_type': 'input', 'text': 'I learned a new trick'},
                               {'context_type': 'output', 'text': 'not an input line'}]}
        rc.save_transcript('d_test', rcr)
        rows = list(ChatTranscript.objects.order_by('id'))
        self.assertEqual([(r.role, r.text) for r in rows],
                         [('user', 'I learned a new trick'),
                          ('moxie', 'That is a great trick, George!')])
        self.assertEqual(rows[0].module_id, 'OPENMOXIE_CHAT')
        self.assertEqual(rows[0].content_id, 'memory2')

    def test_animation_speech_and_unknown_device_skipped(self):
        from types import SimpleNamespace
        from .mqtt.moxie_remote_chat import RemoteChat
        MoxieDevice.objects.create(device_id='d_test')
        rc = RemoteChat(SimpleNamespace())
        rc.save_transcript('d_test', {'speech': 'animation:happy', 'extra_lines': []})
        rc.save_transcript('d_ghost', {'speech': 'hello', 'extra_lines': []})
        self.assertEqual(ChatTranscript.objects.count(), 0)


class MemoryFragmentTests(TestCase):
    def setUp(self):
        self.device = MoxieDevice.objects.create(device_id='d_mem')

    def test_ops_add_update_retire_restore(self):
        r = apply_memory_ops('d_mem', [
            {'op': 'add', 'bank': 'magic_tricks', 'key': 'trick:French-Drop',
             'text': 'Learning the French Drop', 'confidence': 0.9},
            {'op': 'add', 'bank': 'people', 'key': 'friend:sam', 'text': 'Best friend Sam'},
        ])
        self.assertEqual(r, {'applied': 2, 'skipped': 0})
        # keys are casefolded so extractor casing wobble cannot duplicate fragments
        apply_memory_ops('d_mem', [{'op': 'update', 'bank': 'magic_tricks',
                                    'key': 'trick:french-drop', 'text': 'Mastered the French Drop'}])
        frag = MemoryFragment.objects.get(device=self.device, bank='magic_tricks')
        self.assertEqual(frag.text, 'Mastered the French Drop')
        self.assertEqual(frag.times_seen, 2)
        apply_memory_ops('d_mem', [{'op': 'retire', 'bank': 'people', 'key': 'friend:sam'}])
        self.assertFalse(MemoryFragment.objects.get(bank='people').active)
        apply_memory_ops('d_mem', [{'op': 'restore', 'bank': 'people', 'key': 'friend:sam'}])
        self.assertTrue(MemoryFragment.objects.get(bank='people').active)

    def test_invalid_ops_are_skipped_not_fatal(self):
        r = apply_memory_ops('d_mem', [
            {'op': 'explode', 'bank': 'x', 'key': 'y', 'text': 'z'},
            {'op': 'add', 'bank': 'x', 'key': 'y'},           # missing text
            'not even a dict',
            {'op': 'retire', 'bank': 'x', 'key': 'never-existed'},
            {'op': 'add', 'bank': 'ok', 'key': 'k', 'text': 'kept', 'confidence': 'bogus'},
        ])
        self.assertEqual(r, {'applied': 1, 'skipped': 4})
        self.assertEqual(MemoryFragment.objects.get(bank='ok').confidence, 0.5)
        self.assertEqual(apply_memory_ops('d_ghost', [{'op': 'add', 'bank': 'a', 'key': 'b',
                                                       'text': 'c'}]),
                         {'error': 'unknown device d_ghost'})

    def test_assemble_core_relevance_and_budget(self):
        apply_memory_ops('d_mem', [
            {'op': 'add', 'bank': 'profile', 'key': 'summary', 'text': 'George, age 7, aspiring magician'},
            {'op': 'add', 'bank': 'places', 'key': 'ruby-falls', 'text': 'Visited Ruby Falls waterfall'},
            {'op': 'add', 'bank': 'people', 'key': 'friend:sam', 'text': 'Best friend Sam plays soccer'},
            {'op': 'add', 'bank': 'running_jokes', 'key': 'joke', 'text': 'Retired joke about socks'},
        ])
        apply_memory_ops('d_mem', [{'op': 'retire', 'bank': 'running_jokes', 'key': 'joke'}])
        ctx = assemble_memory('d_mem', utterance='remember the waterfall at ruby falls?')
        self.assertIn('George, age 7', ctx)          # core bank always present
        self.assertIn('Ruby Falls', ctx)             # relevant fragment selected
        self.assertNotIn('socks', ctx)               # retired fragments excluded
        self.assertIn('PROFILE:', ctx)
        # tight budget: core survives, non-core is squeezed out
        tight = assemble_memory('d_mem', utterance='', budget_chars=10)
        self.assertIn('George, age 7', tight)
        self.assertNotIn('Sam', tight)
        self.assertEqual(assemble_memory('d_ghost'), '')


class MemoryEmbeddingTests(TestCase):
    MAGIC_VEC = [1.0, 0.0]
    DINO_VEC = [0.0, 1.0]

    def setUp(self):
        MoxieDevice.objects.create(device_id='d_mem')
        self._orig_embed = memory_mod._embed_texts
        memory_mod._topic_vectors.clear()
        memory_mod._embed_texts = lambda texts: [
            self.MAGIC_VEC if 'magic' in t.lower() else self.DINO_VEC for t in texts]

    def tearDown(self):
        memory_mod._embed_texts = self._orig_embed
        memory_mod._topic_vectors.clear()

    def test_fragments_embedded_on_write(self):
        apply_memory_ops('d_mem', [{'op': 'add', 'bank': 'shows', 'key': 'talent',
                                    'text': 'Planning a magic show'}])
        frag = MemoryFragment.objects.get(key='talent')
        self.assertEqual(frag.embedding, self.MAGIC_VEC)

    def test_topic_vector_rolls_with_decay(self):
        memory_mod.update_topic_vector('d_mem', 'we talked about magic tricks')
        self.assertEqual(memory_mod.get_topic_vector('d_mem'), self.MAGIC_VEC)
        memory_mod.update_topic_vector('d_mem', 'now we talk dinosaurs')
        v = memory_mod.get_topic_vector('d_mem')
        self.assertAlmostEqual(v[0], 0.7)
        self.assertAlmostEqual(v[1], 0.3)

    def test_assemble_prefers_topic_matching_fragment(self):
        apply_memory_ops('d_mem', [
            {'op': 'add', 'bank': 'shows', 'key': 'talent', 'text': 'Planning a magic show'},
            {'op': 'add', 'bank': 'animals', 'key': 'trex', 'text': 'Loves the T-Rex dinosaur'},
        ])
        memory_mod.update_topic_vector('d_mem', 'magic magic magic')
        # utterance gives no lexical signal; budget fits only one non-core fragment
        ctx = assemble_memory('d_mem', utterance='', budget_chars=40, core_banks=())
        self.assertIn('magic show', ctx)
        self.assertNotIn('T-Rex', ctx)

    def test_embedding_outage_degrades_to_lexical(self):
        memory_mod._embed_texts = lambda texts: None
        r = apply_memory_ops('d_mem', [{'op': 'add', 'bank': 'shows', 'key': 'talent',
                                        'text': 'Planning a magic show'}])
        self.assertEqual(r, {'applied': 1, 'skipped': 0})
        self.assertIsNone(MemoryFragment.objects.get(key='talent').embedding)
        memory_mod.update_topic_vector('d_mem', 'anything')
        self.assertIsNone(memory_mod.get_topic_vector('d_mem'))
        ctx = assemble_memory('d_mem', utterance='tell me about the magic show', core_banks=())
        self.assertIn('magic show', ctx)


class LoadContentFragmentTests(TestCase):
    def test_fragments_section_seeds_rows(self):
        MoxieDevice.objects.create(device_id='d_seed')
        pack = {'fragments': [{'device_id': 'd_seed',
                               'ops': [{'op': 'add', 'bank': 'profile', 'key': 'summary',
                                        'text': 'George, aspiring magician', 'confidence': 1.0}]}]}
        import tempfile
        f = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
        json.dump(pack, f)
        f.close()
        call_command('load_content', f.name)
        self.assertEqual(MemoryFragment.objects.filter(bank='profile').count(), 1)
        bad = {'fragments': [{'device_id': 'd_ghost', 'ops': []}]}
        f2 = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
        json.dump(bad, f2)
        f2.close()
        with self.assertRaises(CommandError):
            call_command('load_content', f2.name)


class DreamTests(TestCase):
    def setUp(self):
        import datetime as dt

        from django.utils import timezone as djtz

        from . import dream as dream_mod
        self.dream = dream_mod
        MoxieDevice.objects.create(device_id='d_mem')
        self._orig_embed = memory_mod._embed_texts
        memory_mod._embed_texts = lambda texts: [[1.0, 0.0] for _ in texts]
        apply_memory_ops('d_mem', [
            {'op': 'add', 'bank': 'other', 'key': 'stale', 'text': 'An old trivial fact', 'confidence': 0.6},
            {'op': 'add', 'bank': 'other', 'key': 'dupe-a', 'text': 'Loves magic shows', 'confidence': 0.7},
            {'op': 'add', 'bank': 'other', 'key': 'dupe-b', 'text': 'Really likes magic shows', 'confidence': 0.7},
            {'op': 'add', 'bank': 'profile', 'key': 'summary', 'text': 'George the magician', 'confidence': 1.0},
        ])
        old = djtz.now() - dt.timedelta(days=90)
        MemoryFragment.objects.filter(key__in=['stale', 'profile']).update(updated=old)
        MemoryFragment.objects.filter(key='stale').update(updated=old)

    def tearDown(self):
        memory_mod._embed_texts = self._orig_embed

    def test_decay_hits_stale_noncore_only(self):
        import datetime as dt

        from django.utils import timezone as djtz
        MemoryFragment.objects.filter(key='summary').update(
            updated=djtz.now() - dt.timedelta(days=90))
        decayed = self.dream._decay_atomic('d_mem')
        self.assertEqual(decayed, 1)  # only 'stale'; dupes are fresh, profile is core
        self.assertAlmostEqual(MemoryFragment.objects.get(key='stale').confidence, 0.45)
        self.assertEqual(MemoryFragment.objects.get(key='summary').confidence, 1.0)

    def test_duplicate_pairs_by_embedding(self):
        frags = list(MemoryFragment.objects.filter(device__device_id='d_mem', active=True))
        pairs = self.dream.duplicate_pairs(frags)
        self.assertTrue(any('dupe-a' in a + b and 'dupe-b' in a + b for a, b in pairs))

    def test_run_dream_applies_llm_ops(self):
        self._orig_llm = self.dream._dream_llm
        self.dream._dream_llm = lambda prompt: json.dumps([
            {'op': 'update', 'bank': 'other', 'key': 'dupe-a',
             'text': 'Loves magic shows', 'confidence': 0.8},
            {'op': 'retire', 'bank': 'other', 'key': 'dupe-b'}])
        try:
            r = self.dream.run_dream('d_mem')
        finally:
            self.dream._dream_llm = self._orig_llm
        self.assertEqual(r['applied'], 2)
        self.assertGreaterEqual(r['dup_candidates'], 1)
        self.assertFalse(MemoryFragment.objects.get(key='dupe-b').active)

    def test_run_dream_unparseable_is_safe(self):
        self._orig_llm = self.dream._dream_llm
        self.dream._dream_llm = lambda prompt: 'I refuse to emit JSON'
        try:
            r = self.dream.run_dream('d_mem')
        finally:
            self.dream._dream_llm = self._orig_llm
        self.assertIn('error', r)
        self.assertEqual(MemoryFragment.objects.filter(active=True).count(), 4)

    def test_backup_creates_and_prunes(self):
        import os
        import sqlite3 as sq
        import tempfile
        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, 'src.sqlite3')
        sq.connect(src).execute('CREATE TABLE t (x)').connection.commit()
        backups = os.path.join(tmp, 'backups')
        os.makedirs(backups)
        for i in range(20):
            open(os.path.join(backups, f'db-2020-01-{i+1:02d}.sqlite3'), 'w').close()
        dest = self.dream.backup_database(src_path=src, dest_dir=backups)
        self.assertTrue(os.path.exists(dest))
        remaining = [f for f in os.listdir(backups) if f.startswith('db-')]
        self.assertEqual(len(remaining), self.dream.BACKUP_KEEP)

    def test_nightly_tick_gates_on_hour_day_and_activity(self):
        import datetime as dt
        import os
        import time

        from django.conf import settings
        marker = os.path.join(settings.DATA_STORE_DIR, self.dream._MARKER_FILE)
        if os.path.exists(marker):
            os.remove(marker)
        fired = []
        self._orig_safe = self.dream._safe_maintenance
        self.dream._safe_maintenance = lambda only_devices=None: fired.append(only_devices)
        try:
            wrong_hour = dt.datetime(2026, 7, 30, self.dream.DREAM_HOUR + 1, 0)
            self.assertFalse(self.dream.nightly_tick(wrong_hour))
            # right hour but zero conversation activity -> skip (and no maintenance)
            right = dt.datetime(2026, 7, 30, self.dream.DREAM_HOUR, 5)
            self.assertFalse(self.dream.nightly_tick(right))
            # next night, with activity -> fires for the active device only
            ChatTranscript.objects.create(device=MoxieDevice.objects.get(device_id='d_mem'),
                                          role='user', text='hi moxie')
            night2 = dt.datetime(2026, 7, 31, self.dream.DREAM_HOUR, 5)
            self.assertTrue(self.dream.nightly_tick(night2))
            self.assertFalse(self.dream.nightly_tick(night2), 'must not fire twice same day')
            time.sleep(0.3)
            self.assertEqual(fired, [['d_mem']])
        finally:
            self.dream._safe_maintenance = self._orig_safe
            if os.path.exists(marker):
                os.remove(marker)


class DirectorTests(TestCase):
    def setUp(self):
        import datetime as dt
        self.dt = dt
        from . import director as director_mod
        self.director = director_mod

    def make_volley(self, director_cfg=None, nickname='George'):
        from types import SimpleNamespace
        return SimpleNamespace(config={'child_pii': {'nickname': nickname}},
                               persist_data={'director': director_cfg or {}},
                               device_id='d_dir', local_data={})

    def make_session(self, assistant_lines=(), volleys=10):
        from types import SimpleNamespace
        hist = [{'role': 'assistant', 'content': l} for l in assistant_lines]
        return SimpleNamespace(_history=hist, total_volleys=volleys)

    def noon(self):
        return self.dt.datetime(2026, 7, 30, 12, 0)

    def test_style_rules_always_present_and_configurable(self):
        out = self.director.build_directives(self.make_volley(), self.make_session(), self.noon())
        self.assertIn('at most once every few replies', out)
        cfg = {'style': {'rules': ['Custom rule only.']}}
        out = self.director.build_directives(self.make_volley(cfg), self.make_session(), self.noon())
        self.assertIn('Custom rule only.', out)
        self.assertNotIn('at most once', out)

    def test_directives_can_be_disabled_per_device(self):
        cfg = {'style': {'enabled': False}, 'repetition': {'enabled': False},
               'time_of_day': {'enabled': False}, 'session_arc': {'enabled': False},
               'topic_dwell': {'enabled': False}}
        out = self.director.build_directives(self.make_volley(cfg), self.make_session(), self.noon())
        self.assertEqual(out, '')

    def test_repetition_flags_formulaic_replies(self):
        lines = ['That sounds great, Magic Davey! What next?',
                 'That sounds amazing, Magic Davey! How fun!',
                 'That sounds terrific, Magic Davey! Wow!',
                 'That sounds wonderful, Magic Davey! Neat!']
        out = self.director.build_directives(self.make_volley(), self.make_session(lines), self.noon())
        self.assertIn('Quality check', out)
        self.assertIn('overusing', out)
        self.assertIn('begin the same way', out)

    def test_repetition_quiet_on_varied_replies(self):
        lines = ['Whales sing songs across whole oceans.',
                 'Did you know card fans need dry hands?',
                 'My circuits are tickled by that joke!']
        out = self.director.build_directives(self.make_volley(), self.make_session(lines), self.noon())
        self.assertNotIn('Quality check', out)

    def test_time_of_day_bedtime_and_winddown(self):
        late = self.dt.datetime(2026, 7, 30, 21, 0)
        out = self.director.build_directives(self.make_volley(), self.make_session(), late)
        self.assertIn('past bedtime', out)
        soon = self.dt.datetime(2026, 7, 30, 19, 45)
        out = self.director.build_directives(self.make_volley(), self.make_session(), soon)
        self.assertIn('Bedtime is coming soon', out)
        custom = {'time_of_day': {'bedtime': '22:00'}}
        out = self.director.build_directives(self.make_volley(custom), self.make_session(), late)
        self.assertNotIn('past bedtime', out)

    def test_session_arc_on_long_sessions(self):
        out = self.director.build_directives(self.make_volley(),
                                             self.make_session(volleys=150), self.noon())
        self.assertIn('long chat', out)

    def test_topic_dwell_uses_memory_history(self):
        import time as t
        vec = [1.0, 0.0]
        with memory_mod._topic_lock:
            memory_mod._topic_vectors['d_dir'] = vec
            memory_mod._topic_history['d_dir'] = [(t.monotonic() - 15 * 60, vec),
                                                  (t.monotonic() - 5 * 60, vec),
                                                  (t.monotonic(), vec)]
        try:
            out = self.director.build_directives(self.make_volley(), self.make_session(), self.noon())
            self.assertIn('same topic for a while', out)
        finally:
            with memory_mod._topic_lock:
                memory_mod._topic_vectors.pop('d_dir', None)
                memory_mod._topic_history.pop('d_dir', None)

    def test_custom_directive_registration(self):
        @self.director.directive('magic_test')
        def magic_test(cfg, volley, session, now):
            return 'Custom concern injected.'
        try:
            out = self.director.build_directives(self.make_volley(), self.make_session(), self.noon())
            self.assertIn('Custom concern injected.', out)
        finally:
            self.director.DIRECTIVES.pop('magic_test', None)


class InitDataTests(TestCase):
    def test_factory_data_loads_and_preserves_local_edits(self):
        call_command('init_data')
        self.assertEqual(MoxieSchedule.objects.count(), 3)
        self.assertEqual(SinglePromptChat.objects.count(), 7)
        chat = SinglePromptChat.objects.get(module_id='OPENMOXIE_CHAT', content_id='default')
        chat.prompt = 'You are a robot from the Daddy Robotics Laboratory.'
        chat.save()
        call_command('init_data')  # same source_version -> must not overwrite
        chat.refresh_from_db()
        self.assertIn('Daddy Robotics', chat.prompt)
