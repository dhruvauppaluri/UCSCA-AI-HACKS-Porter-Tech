"""Explicit EarPods microphone and MacBook speaker routing."""
import queue
import threading
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

class MacAudioInterface(DefaultAudioInterface):
    @staticmethod
    def devices(p):
        devices = [p.get_device_info_by_index(i) for i in range(p.get_device_count())]
        def find(name, channels):
            matches = [d for d in devices if name in d['name'] and d[channels] > 0]
            if len(matches) != 1:
                raise RuntimeError('Connect/select exactly one ' + name + ' device')
            return int(matches[0]['index'])
        return find('EarPods', 'maxInputChannels'), find('MacBook Pro Speakers', 'maxOutputChannels')

    def start(self, input_callback):
        self.input_callback = input_callback
        self.output_queue = queue.Queue()
        self.should_stop = threading.Event()
        self.p = self.pyaudio.PyAudio()
        self.in_stream = self.out_stream = None
        self.output_thread = None
        try:
            mic, speaker = self.devices(self.p)
            self.in_stream = self.p.open(format=self.pyaudio.paInt16, channels=1, rate=16000, input=True, input_device_index=mic, stream_callback=self._in_callback, frames_per_buffer=self.INPUT_FRAMES_PER_BUFFER)
            self.out_stream = self.p.open(format=self.pyaudio.paInt16, channels=1, rate=16000, output=True, output_device_index=speaker, frames_per_buffer=self.OUTPUT_FRAMES_PER_BUFFER)
            self.output_thread = threading.Thread(target=self._output_thread)
            self.output_thread.start()
            print('Audio: EarPods microphone -> MacBook Pro Speakers', flush=True)
        except Exception:
            self.stop()
            raise

    def stop(self):
        if not hasattr(self, 'should_stop'):
            return
        self.should_stop.set()
        if self.output_thread:
            self.output_thread.join(timeout=2)
            self.output_thread = None
        for name in ('in_stream', 'out_stream'):
            stream = getattr(self, name, None)
            if stream:
                stream.close()
                setattr(self, name, None)
        if self.p:
            self.p.terminate()
            self.p = None
