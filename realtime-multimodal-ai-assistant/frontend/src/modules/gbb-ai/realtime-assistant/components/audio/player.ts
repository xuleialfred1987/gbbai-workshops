export class Player {
  private audioContext: AudioContext | null = null;
  
  private playbackNode: AudioWorkletNode | null = null;

  private isWorkletLoaded = false;

  async init(sampleRate: number) {
    try {
      // Clean up any existing resources
      await this.cleanup();

      this.audioContext = new AudioContext({ sampleRate });

      // Only load worklet if not already loaded
      if (!this.isWorkletLoaded) {
        try {
          await this.audioContext.audioWorklet.addModule('/scripts/audio-playback-worklet.js');
          this.isWorkletLoaded = true;
        } catch (error) {
          console.error('Failed to load audio playback worklet:', error);
          throw new Error('Failed to load audio playback module');
        }
      }

      try {
        this.playbackNode = new AudioWorkletNode(this.audioContext, 'audio-playback-worklet');
        this.playbackNode.connect(this.audioContext.destination);
      } catch (error) {
        console.error('Error creating playback node:', error);
        await this.cleanup();
        throw new Error('Failed to initialize audio playback');
      }
    } catch (error) {
      console.error('Error initializing player:', error);
      await this.cleanup();
      throw error;
    }
  }

  play(buffer: Int16Array) {
    if (this.playbackNode && this.audioContext) {
      try {
        // Resume AudioContext if it's suspended (browser autoplay policy)
        if (this.audioContext.state === 'suspended') {
          this.audioContext.resume();
        }
        this.playbackNode.port.postMessage(buffer);
      } catch (error) {
        console.error('Error during audio playback:', error);
      }
    } else {
      console.warn('Attempted to play audio without initialized player');
    }
  }

  stop() {
    if (this.playbackNode) {
      try {
        this.playbackNode.port.postMessage(null);
      } catch (error) {
        console.error('Error stopping playback:', error);
      }
    }
  }

  async cleanup() {
    try {
      if (this.playbackNode) {
        try {
          this.playbackNode.disconnect();
        } catch (e) {
          console.warn('Error disconnecting playback node:', e);
        }
        this.playbackNode = null;
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
      console.error('Error during player cleanup:', error);
    }
  }

  getSampleRate(): number | null {
    return this.audioContext?.sampleRate ?? null;
  }
}
