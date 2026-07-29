import random

import numpy
from django.core.management import call_command
from django.test import TestCase

from .models import MoxieDevice, MoxieSchedule, PersistentData, SinglePromptChat
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
