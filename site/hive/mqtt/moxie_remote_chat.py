'''
MOXIE REMOTE CHAT - Handle Remote Chat protocol messages from Moxie

Remote Chat (RemoteChatRequest/Response) messages make up the bulk of the remote module
interface for Moxie.  Module/Content IDs that are remote, send inputs and get outputs
using these messages.

Moxie Remote Chat uses a "notify" tracking approach for context, meaning that Moxie notifies
the remote service everytime it says something.  This data is used to accumulate the true
history of the conversation and provides mostly seemless conversation context for the AI,
even when the user provides input in multiple speech windows before hearing a response.
'''
import concurrent.futures
import json
from ..models import SinglePromptChat, MoxieDevice, ChatTranscript, ChatSessionState
from .util import run_db_atomic
from ..automarkup import process as automarkup_process
from ..automarkup import initialize_rules as automarkup_initialize_rules
import logging
from datetime import datetime
from .global_responses import GlobalResponses
from .conversations import ChatSession, SinglePromptDBChatSession
from .volley import Volley

# Turn on to enable global commands in the cloud
_ENABLE_GLOBAL_COMMANDS = True
_LOG_ALL_RCR = False
_LOG_NOTIFY_RCR = True
_PERSIST_TRANSCRIPTS = True
_PERSIST_SESSIONS = True
_SESSION_RESTORE_WINDOW_SECS = 1800
_MAX_WORKER_THREADS = 5

logger = logging.getLogger(__name__)

'''
RemoteChat is the plugin to the MoxieServer that handles all remote module requests.  It
keeps track of the active remote module, creates new ones as needed, and ignores all data
from local modules.  It also manages auto-markup, which renders plaintext into a markup
language with animated gestures.
'''
class RemoteChat:
    _global_responses: GlobalResponses
    def __init__(self, server):
        self._server = server
        self._device_sessions = {}
        self._modules = { }
        self._modules_info = { "modules": [], "version": "openmoxie_v1" }
        self._worker_queue = concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKER_THREADS)
        self._automarkup_rules = automarkup_initialize_rules()
        self._global_responses = GlobalResponses()

    def register_module(self, module_id, content_id, cname):
        self._modules[f'{module_id}/{content_id}'] = cname

    # Gets the remote module info record to share remote modules with Moxie
    def get_modules_info(self):
        return self._modules_info
    
    # Update database backed records.  Query to get all the module/content IDs and update the modules schema and register the modules
    def update_from_database(self):
        new_modules = {}
        mod_map = {}
        for chat in SinglePromptChat.objects.all():
            # one module can support many content IDs, separated by | like openers
            cid_list = chat.content_id.split('|')
            for content_id in cid_list:
                new_modules[f'{chat.module_id}/{content_id}'] = { 'xtor': SinglePromptDBChatSession, 'params': { 'pk': chat.pk } }
                logger.debug(f'Registering {chat.module_id}/{content_id}')
                # Group content IDs under module IDs
                if chat.module_id in mod_map:
                    mod_map[chat.module_id].append(content_id)
                else:
                    mod_map[chat.module_id] = [ content_id ]
        # Models/content IDs into the module info schema - bare bones mandatory fields only
        mlist = []
        for mod in mod_map.keys():
            modinfo = { "info": { "id": mod }, "rules": "RANDOM", "source": "REMOTE_CHAT", "content_infos": [] }
            for cid in mod_map[mod]:
                modinfo["content_infos"].append({ "info": { "id": cid } })
            mlist.append(modinfo)
        self._modules_info["modules"] = mlist
        self._modules = new_modules
        self._global_responses.update_from_database()
    
    # Handle GLOBAL patterns, available inside (almost) any module
    def check_global(self, volley):
        return self._global_responses.check_global(volley) if _ENABLE_GLOBAL_COMMANDS else None
        
    def on_chat_complete(self, device_id, id, session:ChatSession):
        logger.info(f'Chat Session Complete: {id} {session.has_complete_hook()}')
        if session.has_complete_hook():
            # make a data-only Volley for the completion hook
            volley = Volley({}, device_id=device_id, data_only=True, robot_data=self._server.robot_data().get_volley_data(device_id), local_data=session.local_data)
            self._worker_queue.submit(self.run_complete_hook, device_id, session, volley)

    # Run a session completion hook, then flush any persistent data it wrote so it survives
    # a server restart before the robot disconnects
    def run_complete_hook(self, device_id, session:ChatSession, volley:Volley):
        try:
            session.complete_hook(volley)
        finally:
            self._server.robot_data().save_persist(device_id)

    # Get the current or a new session for this device for this module/content ID pair
    def active_session_data(self, device_id):
        if device_id in self._device_sessions:
            return self._device_sessions[device_id]['session'].local_data
        return None
            
    # Get the current or a new session for this device for this module/content ID pair
    def get_session(self, device_id, id, maker) -> ChatSession:
        # each device has a single session only for now
        if device_id in self._device_sessions:
            if self._device_sessions[device_id]['id'] == id:
                return self._device_sessions[device_id]['session']
            else:
                self.on_chat_complete(device_id, self._device_sessions[device_id]['id'], self._device_sessions[device_id]['session'])

        # new session needed
        new_session = { 'id': id, 'session': maker['xtor'](**maker['params']) }
        self._device_sessions[device_id] = new_session
        if _PERSIST_SESSIONS:
            self.restore_session_state(device_id, id, new_session['session'])
        return new_session['session']

    # Snapshot the device's live session to the DB (runs on worker pool) so a server
    # restart mid-conversation resumes instead of forgetting
    def save_session_state(self, device_id):
        try:
            rec = self._device_sessions.get(device_id)
            if not rec:
                return
            sess = rec['session']
            local = {}
            for k, v in getattr(sess, '_local_data', {}).items():
                try:
                    json.dumps(v)
                    local[k] = v
                except (TypeError, ValueError):
                    pass
            run_db_atomic(self.write_session_state, device_id, rec['id'],
                          list(getattr(sess, '_history', [])),
                          getattr(sess, '_total_volleys', 0), local)
        except Exception as e:
            logger.warning(f'Failed to save session state: {e}')

    def write_session_state(self, device_id, session_id, history, volleys, local):
        device = MoxieDevice.objects.filter(device_id=device_id).first()
        if device:
            ChatSessionState.objects.update_or_create(
                device=device, defaults={'session_id': session_id, 'history': history,
                                         'total_volleys': volleys, 'local_data': local})

    def restore_session_state(self, device_id, id, session):
        try:
            state = run_db_atomic(self.read_session_state, device_id)
            if state and state['session_id'] == id:
                session._history = state['history']
                session._total_volleys = state['total_volleys']
                session._local_data.update(state['local_data'])
                logger.info(f'Restored session for {device_id} ({id}, {state["total_volleys"]} volleys)')
        except Exception as e:
            logger.warning(f'Failed to restore session state: {e}')

    def read_session_state(self, device_id):
        from django.utils import timezone
        rec = ChatSessionState.objects.filter(device__device_id=device_id).first()
        if rec and (timezone.now() - rec.updated).total_seconds() <= _SESSION_RESTORE_WINDOW_SECS:
            return {'session_id': rec.session_id, 'history': rec.history,
                    'total_volleys': rec.total_volleys, 'local_data': rec.local_data}
        return None

    # Get's a chat session object for use in the web chat
    def get_web_session_for_module(self, device_id, module_id, content_id):
        id = module_id + '/' + content_id
        maker = self._modules.get(id)
        return self.get_session(device_id, id, maker) if maker else None

    def get_web_session_global_response(self, volley: Volley):
        global_functor = self.check_global(volley)
        if global_functor:
            resp = global_functor()
            if isinstance(resp, str):
                return resp
            else:
                return volley.debug_response_string()
        return None

    # Markup text      
    def make_markup(self, text, mood_and_intensity = None):
        return automarkup_process(text, self._automarkup_rules, mood_and_intensity=mood_and_intensity)

    # Get the next response to a chat
    def create_session_response(self, device_id, sess:ChatSession, volley: Volley):
        sess.handle_volley(volley)
        if 'markup' not in volley.response['output']:
            # if we don't have markup, create it
            text = volley.response['output']['text']
            volley.set_output(text, self.make_markup(text))

        if _LOG_ALL_RCR:
            logger.info(f"RemoteChatResponse\n{volley.response}")
        self._server.send_command_to_bot_json(device_id, 'remote_chat', volley.response)
    
    # Produce / execute a global response
    def global_response(self, device_id, functor):
        resp = functor()
        output = resp.get('output')
        if output.get('text') and not output.get('markup'):
            # Run automarkup on any text-only responses
            output['markup'] = self.make_markup(output['text'])
        self._server.send_command_to_bot_json(device_id, 'remote_chat', resp)
        pass

    def log_notify(self, rcr):
        moxie_speech = rcr.get('speech')
        for el in rcr.get('extra_lines', []):
            if el.get('context_type') == 'input':
                logger.info(f"-- USER: {el.get('text')}")    
        if moxie_speech:
            logger.info(f"-- MOXIE: {moxie_speech} [{rcr.get('module_id')}/{rcr.get('content_id')}]")

    # Persist the spoken lines from a notify record as transcript rows (runs on worker pool)
    def save_transcript(self, device_id, rcr):
        try:
            lines = []
            for el in rcr.get('extra_lines', []):
                if el.get('context_type') == 'input' and el.get('text'):
                    lines.append(('user', el['text']))
            speech = rcr.get('speech')
            if speech and 'animation:' not in speech and 'silent:' not in speech:
                lines.append(('moxie', speech))
            if lines:
                run_db_atomic(self.write_transcript_rows, device_id,
                              rcr.get('module_id', ''), rcr.get('content_id', ''), lines)
                # fold the spoken turn into the rolling topic vector (memory v3 retrieval);
                # already off the response path, and a no-op if embeddings are unavailable
                from ..memory import update_topic_vector
                update_topic_vector(device_id, ' '.join(text for _, text in lines))
        except Exception as e:
            logger.warning(f'Failed to persist transcript: {e}')

    def write_transcript_rows(self, device_id, module_id, content_id, lines):
        device = MoxieDevice.objects.filter(device_id=device_id).first()
        if device:
            for role, text in lines:
                ChatTranscript.objects.create(device=device, module_id=module_id,
                                              content_id=content_id, role=role, text=text)

    # Entry point where all RemoteChatRequests arrive
    def handle_request(self, device_id, rcr, volley_data):
        if _LOG_ALL_RCR:
            logger.info(f"RemoteChatRequest\n{rcr}")
        id = rcr.get('module_id', '') + '/' + rcr.get('content_id', '')
        cmd = rcr.get('command')
        if _LOG_NOTIFY_RCR and cmd == 'notify':
            self.log_notify(rcr)
        if _PERSIST_TRANSCRIPTS and cmd == 'notify':
            # notify carries what was actually spoken (either side), the ground truth transcript
            self._worker_queue.submit(self.save_transcript, device_id, rcr)

        maker = self._modules.get(id)
        if maker:
            # THIS IS THE PATH FOR REMOTE CONTENT - MODULE/CONTENT HOSTED IN OPENMOXIE
            logger.debug(f'Handling RCR:{cmd} for {id}') 
            sess = self.get_session(device_id, id, maker)
            if cmd == 'notify':
                volley = Volley(rcr, device_id=device_id, robot_data=volley_data, local_data=sess.local_data, data_only=True)
                sess.ingest_notify(volley)
                if _PERSIST_SESSIONS:
                    # notify carries the ground-truth history; snapshot after each one
                    self._worker_queue.submit(self.save_session_state, device_id)
            else:
                volley = Volley(rcr, device_id=device_id, robot_data=volley_data, local_data=sess.local_data)
                if not self.handled_global(device_id, volley):
                    self._worker_queue.submit(self.create_session_response, device_id, sess, volley)
        else:
            # THIS IS THE PATH FOR MOXIE ON-BOARD CONTENT
            session_reset = False
            if device_id in self._device_sessions:
                session = self._device_sessions.pop(device_id, None)
                self.on_chat_complete(device_id, session['id'], session['session'])
                session_reset = True
            if cmd != 'notify':
                volley = Volley(rcr, device_id=device_id, robot_data=volley_data)
                if not self.handled_global(device_id, volley):
                    logger.debug(f'Ignoring request for other module: {id} SessionReset:{session_reset}')
                    # Rather than ignoring these, we return a generic FALLBACK response
                    fbline = "I'm sorry. Can  you repeat that?"
                    volley.set_output(fbline, fbline, output_type='FALLBACK')
                    self._server.send_command_to_bot_json(device_id, 'remote_chat', volley.response)
    
    def handled_global(self, device_id, volley):
        global_functor = self.check_global(volley)
        if global_functor:
            logger.debug(f'Global response inside {id}')
            self._worker_queue.submit(self.global_response, device_id, global_functor)
            return True
        return False
