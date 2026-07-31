from .moxie_zmq_handler import ZMQHandler
from .protos.embodied.perception.audio.zmqSTT_pb2 import zmqSTTRequest,zmqSTTResponse
import soundfile as sf
import numpy as np
import base64
import io
import json
import queue
import threading
import time
import logging
import concurrent.futures
from .ai_factory import create_openai, get_openai_key

LOG_WAV=False
OPENAI_MODEL='gpt-transcribe'      # OpenAI's recommended transcription model (July 2026)
FALLBACK_MODEL='whisper-1'         # battle-tested fallback if the primary rejects a request

# Streaming mode: transcribe DURING speech via gpt-live-transcribe over WebSocket,
# sending PARTIAL results to the robot as deltas arrive and FINAL right after
# end-of-speech. Audio is always buffered in parallel; any streaming failure or a
# missed final falls back to the batch path above, so streaming can never be worse.
STT_STREAMING=True
STREAM_MODEL='gpt-live-transcribe'
STREAM_URL='wss://api.openai.com/v1/realtime?intent=transcription'
STREAM_FINAL_TIMEOUT=2.5           # seconds after EOS before batch fallback
PARTIAL_MIN_INTERVAL=0.3           # throttle PARTIAL responses to the robot

logger = logging.getLogger(__name__)

def now_ms():
    return time.time_ns() // 1_000_000


def resample_16k_to_24k(pcm_bytes):
    # linear interpolation 16kHz -> 24kHz mono int16 (the realtime API's expected rate)
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(samples) == 0:
        return b''
    idx = np.linspace(0, len(samples) - 1, int(len(samples) * 1.5))
    return np.interp(idx, np.arange(len(samples)), samples).astype(np.int16).tobytes()


'''
A thin synchronous client for the realtime transcription WebSocket. One instance per
utterance. Tolerant event parsing: any *.delta event extends the running transcript,
any *completed/*done transcription event finalizes it.
'''
class LiveStream:
    def __init__(self, prompt, on_partial, on_final):
        import websocket
        self._on_partial = on_partial
        self._on_final = on_final
        self._accum = ''
        self._done = False
        self._ws = websocket.create_connection(
            STREAM_URL,
            header=[f'Authorization: Bearer {get_openai_key()}'],
            timeout=5)
        # GA protocol shape (probed live 2026-07-31): session.type=transcription with
        # nested audio.input config; turn_detection off - the robot's VAD decides EOS
        self._ws.send(json.dumps({
            'type': 'session.update',
            'session': {
                'type': 'transcription',
                'audio': {'input': {
                    'format': {'type': 'audio/pcm', 'rate': 24000},
                    'transcription': {'model': STREAM_MODEL, 'prompt': prompt},
                    'turn_detection': None,
                }}}}))
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def send_audio(self, pcm_16k_bytes):
        audio = base64.b64encode(resample_16k_to_24k(pcm_16k_bytes)).decode()
        self._ws.send(json.dumps({'type': 'input_audio_buffer.append', 'audio': audio}))

    def commit(self):
        self._ws.send(json.dumps({'type': 'input_audio_buffer.commit'}))

    def close(self):
        try:
            self._ws.close()
        except Exception:
            pass

    def _read_loop(self):
        try:
            while not self._done:
                evt = json.loads(self._ws.recv())
                etype = evt.get('type', '')
                if etype.endswith('.delta') and 'transcription' in etype:
                    self._accum += evt.get('delta', '')
                    self._on_partial(self._accum)
                elif ('transcription' in etype and
                      (etype.endswith('.completed') or etype.endswith('.done'))):
                    self._done = True
                    self._on_final(evt.get('transcript') or self._accum)
                elif etype == 'error':
                    logger.warning(f'LiveStream error event: {evt}')
        except Exception as e:
            if not self._done:
                logger.debug(f'LiveStream reader ended: {e}')


'''
An STT Session is a stream of contiguous audio coming out of the Robot's voice activity detector (VAD). This
is a very simple implementation tuned to OpenAI Whisper.  Their API doesn't support streaming, so we simply
accumulate the audio frames, then transcribe them when complete.
'''
class STTSession:
    def __init__(self, parent, device_id, session_id, streaming=None):
        self._parent = parent
        self._device_id = device_id
        self._session_id = session_id
        self._stream_bytes = bytearray()
        self._start_ts = None
        self._streaming = STT_STREAMING if streaming is None else streaming
        self._chunk_queue = queue.Queue()
        self._live_started = False
        self._live = None
        self._live_failed = False
        self._final_text = None
        self._final_evt = threading.Event()
        self._last_partial = 0.0

    def on_request(self, req):
        # future ref, this is technically wrong in the design, this ts is realtime on robot, not audio timestamp
        if not self._start_ts:
            self._start_ts = req.timestamp
        self._stream_bytes += req.audio_content   # always buffered: the fallback source
        if self._streaming and req.audio_content:
            self._chunk_queue.put(bytes(req.audio_content))
            if not self._live_started:
                self._live_started = True
                self._parent.submit_stream_worker(self._stream_worker)
        return len(self._stream_bytes)

    # runs on its own worker thread: connect, forward chunks, commit at EOS sentinel
    def _stream_worker(self):
        try:
            self._live = LiveStream(self._parent.bias_prompt(self._device_id),
                                    self._on_partial, self._on_final)
        except Exception as e:
            logger.warning(f'LiveStream connect failed ({e}); will use batch fallback')
            self._live_failed = True
            return
        try:
            while True:
                chunk = self._chunk_queue.get()
                if chunk is None:
                    self._live.commit()
                    return
                self._live.send_audio(chunk)
        except Exception as e:
            logger.warning(f'LiveStream send failed ({e}); will use batch fallback')
            self._live_failed = True
            self._live.close()

    def _on_partial(self, text):
        now = time.monotonic()
        if now - self._last_partial < PARTIAL_MIN_INTERVAL:
            return
        self._last_partial = now
        resp = zmqSTTResponse()
        resp.uuid = self._session_id
        resp.type = resp.ResponseType.PARTIAL
        resp.timestamp = now_ms()
        resp.speech = text
        self._parent.zmq_reply(self._device_id, resp)

    def _on_final(self, text):
        self._final_text = text
        self._final_evt.set()

    def perform(self):
        # streaming first: EOS sentinel -> await the live final briefly; fall back to batch
        if self._streaming and self._live_started and not self._live_failed:
            self._chunk_queue.put(None)
            if self._final_evt.wait(timeout=STREAM_FINAL_TIMEOUT) and self._final_text is not None:
                resp = zmqSTTResponse()
                resp.uuid = self._session_id
                resp.type = resp.ResponseType.FINAL
                resp.timestamp = now_ms()
                resp.speech = self._final_text
                duration_ms = len(self._stream_bytes) // 32
                resp.start_timestamp = self._start_ts or now_ms()
                resp.end_timestamp = resp.start_timestamp + duration_ms
                logger.info(f'STT-LIVE-FINAL: {self._final_text}')
                self._parent.zmq_reply(self._device_id, resp)
                if self._live:
                    self._live.close()
                return
            logger.warning('LiveStream final missed the deadline; using batch fallback')
            if self._live:
                self._live.close()
        logger.info(f'Processing session_id {self._session_id} with {len(self._stream_bytes)} bytes')
        buffer = io.BytesIO()
        sf.write(
            buffer,  # File-like object (None for bytes)
            np.frombuffer(self._stream_bytes, dtype=np.int16),
            16000,
            format='WAV',
            subtype='PCM_16'  # 16-bit PCM
            )
        wav_bytes = buffer.getvalue()
        # Create proto response, send regardless
        resp = zmqSTTResponse()
        resp.uuid = self._session_id
        resp.type = resp.ResponseType.FINAL
        resp.timestamp = now_ms()

        try:
            client = create_openai()
            prompt = self._parent.bias_prompt(self._device_id)
            transcript, last_error = None, None
            for model in (OPENAI_MODEL, FALLBACK_MODEL):
                try:
                    transcript = client.audio.transcriptions.create(
                        file=('speech.wav', wav_bytes),
                        model=model,
                        response_format='json',
                        prompt=prompt)
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f'STT model {model} failed: {e}')
            if transcript is None:
                raise last_error
            resp.speech = transcript.text
            # timestamps from the PCM stream itself: 16kHz 16-bit mono = 32 bytes/ms
            duration_ms = len(self._stream_bytes) // 32
            resp.start_timestamp = self._start_ts or now_ms()
            resp.end_timestamp = resp.start_timestamp + duration_ms
            logger.info(f'STT-FINAL: {transcript.text}')
        except Exception as e:
            logger.warning(f'Exception handling openAI request: {e}')
            resp.error_code = 66
            resp.error_message = str(e)

        # send response to device
        self._parent.zmq_reply(self._device_id, resp)

        if LOG_WAV:
            logfile = f'{self._session_id}.wav'
            with open(logfile, 'wb') as f:
                f.write(wav_bytes)
                logger.info(f'Wrote WAV data to {logfile}')

'''
This is the handler for all Speech data packets.  By default, the Robot uses stt:4, which begins sending
audio data during session to be transcribed.  If Robot is using stt:0, no STT packets will arrive here.
This is also very simple.  We create unique sessions for each device_id / session pair, pass them all the
data inline, and when a session hits end-of-speech, queue transcription to run in the background.
'''
class STTHandler(ZMQHandler):

    def __init__(self, server):
        super().__init__(server)
        self._sessions = {}
        self._worker_queue = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self._stream_workers = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def submit_stream_worker(self, fn):
        self._stream_workers.submit(fn)

    # Context biasing: the child's name plus their personal vocabulary from memory
    # fragments (stuffed animals, pets, tricks) - benchmarks show this is the largest
    # STT quality lever for family names (5/10 -> 9/10 correct).
    def bias_prompt(self, device_id):
        words = []
        try:
            cfg = self._server.robot_data().get_config(device_id)
            nick = (cfg.get('child_pii') or {}).get('nickname')
            if nick:
                words.append(nick)
        except Exception:
            pass
        try:
            from ..memory import stt_lexicon
            lex = stt_lexicon(device_id)
            if lex:
                words.append(lex)
        except Exception:
            pass
        base = 'A child talking with their robot friend Moxie.'
        return base + ((' Vocabulary: ' + ', '.join(words) + '.') if words else '')

    def handle_zmq(self, device_id, protoname, protodata):
        req = zmqSTTRequest()
        req.ParseFromString(protodata)
        sesskey = ( device_id, req.uuid )
        if sesskey not in self._sessions:
            self._sessions[sesskey] = STTSession(self, sesskey[0], sesskey[1])
        total_sess_bytes = self._sessions[sesskey].on_request(req)
        # every time we reach EOS, we background it for work
        logger.debug(f'ZMQ Speech VAD: {req.vad} TotalBytes: {total_sess_bytes}')
        if req.vad == req.VADState.END_OF_SPEECH:
            logger.info(f'Session reached END OF SPEECH')
            # session is done, do the work
            sess = self._sessions.pop(sesskey)
            self._worker_queue.submit(sess.perform)
