export const getRtConfiguration = (initialResourceName: string) => ({
  'rt-Deployment': initialResourceName,
  'rt-Temperature': '0.7',
  'rt-Voice choice': 'Alloy',
  'rt-Voice host': 'gpt-realtime',
  'rt-Max response': '1000',
  'rt-Disable audio': false,
  'rt-System message': 'You are an assistant designed to help people perform tasks such as answering questions',
  'rt-VAD Threshold': '0.55',
  'rt-VAD Prefix Padding (ms)': '180',
  'rt-VAD Silence Duration (ms)': '220',
});
