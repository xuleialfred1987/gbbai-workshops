export class Recorder {
  onDataAvailable: (buffer: Iterable<number>) => void;

  private audioContext: AudioContext | null = null;

  private mediaStream: MediaStream | null = null;

  private mediaStreamSource: MediaStreamAudioSourceNode | null = null;

  private workletNode: AudioWorkletNode | null = null;

  private isWorkletLoaded = false;

  public constructor(onDataAvailable: (buffer: Iterable<number>) => void) {
    this.onDataAvailable = onDataAvailable;
  }

  async start(stream: MediaStream) {
    try {
      if (this.audioContext) {
        await this.stop();
      }

      this.audioContext = new AudioContext({ sampleRate: 24000 });

      this.isWorkletLoaded = false;

      if (!this.isWorkletLoaded) {
        try {
          await this.audioContext.audioWorklet.addModule('/scripts/audio-processor-worklet.js');
          this.isWorkletLoaded = true;
        } catch (error) {
          console.error('Failed to load audio processor worklet:', error);
          throw new Error('Failed to load audio processing module');
        }
      }

      this.mediaStream = stream;
      this.mediaStreamSource = this.audioContext.createMediaStreamSource(this.mediaStream);

      if (!this.isWorkletLoaded) {
        throw new Error('Audio worklet not loaded properly');
      }

      try {
        this.workletNode = new AudioWorkletNode(this.audioContext, 'audio-processor-worklet');
        this.workletNode.port.onmessage = (event) => {
          if (event.data && event.data.buffer) {
            this.onDataAvailable(event.data.buffer);
          }
        };

        this.mediaStreamSource.connect(this.workletNode);
        this.workletNode.connect(this.audioContext.destination);
      } catch (error) {
        console.error('Error creating audio worklet node:', error);
        await this.stop();
        throw new Error('Failed to initialize audio processing');
      }
    } catch (error) {
      console.error('Error starting recorder:', error);
      await this.stop();
      throw error;
    }
  }

  async stop() {
    try {
      if (this.mediaStream) {
        this.mediaStream.getTracks().forEach((track) => track.stop());
        this.mediaStream = null;
      }

      if (this.workletNode) {
        try {
          this.workletNode.disconnect();
        } catch (e) {
          console.warn('Error disconnecting worklet node:', e);
        }
        this.workletNode = null;
      }

      if (this.mediaStreamSource) {
        try {
          this.mediaStreamSource.disconnect();
        } catch (e) {
          console.warn('Error disconnecting media stream source:', e);
        }
        this.mediaStreamSource = null;
      }

      if (this.audioContext && this.audioContext.state !== 'closed') {
        try {
          await this.audioContext.close();
        } catch (e) {
          console.warn('Error closing audio context:', e);
        }
        this.audioContext = null;
      }
    } catch (error) {
      console.error('Error during recorder cleanup:', error);
    }
  }

  setMuted(muted: boolean) {
    if (this.mediaStream) {
      this.mediaStream.getAudioTracks().forEach((track) => {
        track.enabled = !muted;
      });
    }
  }
}
