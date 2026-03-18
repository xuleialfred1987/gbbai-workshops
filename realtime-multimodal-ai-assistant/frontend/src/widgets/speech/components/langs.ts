export interface SpeechOption {
  lang: string;
  voice: string;
  style: string;
}

export interface VoiceOption {
  name: string;
  value: string;
  avatar: string;
  styles: string[];
}

export const COUNTRIES = [
  { name: 'English', icon: 'flagpack:us', value: 'en-US' },
  { name: 'Chinese Simplified', icon: 'flagpack:cn', value: 'zh-Hans' },
  { name: 'Chinese Traditional', icon: 'flagpack:cz', value: 'zh-Hant' },
  { name: 'Tamil', icon: 'flagpack:dk', value: 'ta' },
  { name: 'Telugu', icon: 'flagpack:nl', value: 'te' },
  { name: 'Hindi', icon: 'flagpack:in', value: 'hi' },
  { name: 'Gujarati', icon: 'flagpack:in', value: 'gu' },
  { name: 'Kannada', icon: 'flagpack:in', value: 'kn' },
  { name: 'Marathi', icon: 'flagpack:in', value: 'mr' },
  { name: 'Urdu', icon: 'flagpack:in', value: 'ur' },
];

export const SOURCE_LANG_COUNTRIES = [
  { name: 'English', icon: 'flagpack:us', displayName: 'EN', value: 'en-us' },
  { name: 'Chinese Simplified', icon: 'flagpack:cn', displayName: 'ZH', value: 'zh-CN' },
  { name: 'Taiwanese Mandarin', icon: 'flagpack:cz', displayName: 'ZH', value: 'zh-TW' },
  { name: 'Tamil', icon: 'flagpack:dk', displayName: 'TA', value: 'ta' },
  { name: 'Telugu', icon: 'flagpack:nl', displayName: 'TE', value: 'te' },
  { name: 'Hindi', icon: 'flagpack:in', displayName: 'HI', value: 'hi' },
  { name: 'Gujarati', icon: 'flagpack:in', displayName: 'GU', value: 'gu' },
  { name: 'Kannada', icon: 'flagpack:in', displayName: 'KN', value: 'kn' },
  { name: 'Marathi', icon: 'flagpack:in', displayName: 'MR', value: 'mr' },
  { name: 'Urdu', icon: 'flagpack:in', displayName: 'UR', value: 'ur' },
  { name: 'Korean', icon: 'flagpack:kr', displayName: 'KR', value: 'ko-kr' },
];

export const TARGET_LANG_COUNTRIES = [
  { name: 'English', icon: 'flagpack:us', value: 'en' },
  { name: 'Chinese Simplified', icon: 'flagpack:cn', value: 'zh-Hans' },
  { name: 'Chinese Traditional', icon: 'flagpack:cz', value: 'zh-tw' },
  { name: 'Tamil', icon: 'flagpack:dk', value: 'ta' },
  { name: 'Telugu', icon: 'flagpack:nl', value: 'te' },
  { name: 'Hindi', icon: 'flagpack:in', value: 'hi' },
  { name: 'Gujarati', icon: 'flagpack:in', value: 'gu' },
  { name: 'Kannada', icon: 'flagpack:in', value: 'kn' },
  { name: 'Marathi', icon: 'flagpack:in', value: 'mr' },
  { name: 'Urdu', icon: 'flagpack:in', value: 'ur' },
];

export const SOURCE_LANG_VOICES: Record<string, VoiceOption[]> = {
  'Chinese Simplified': [
    {
      name: 'Yunxi',
      value: 'zh-CN-YunxiNeural',
      avatar: '/assets/avatars/speech/zh-CN-YunxiNeural.png',
      styles: [
        'Default',
        'Narration - relaxed',
        'Embarrassed',
        'Fearful',
        'Cheerful',
        'Disgruntled',
        'Serious',
        'Angry',
        'Sad',
        'Depressed',
        'Chat',
        'Assistant',
        'Newscast',
      ],
    },
    {
      name: 'Xiaochen',
      value: 'zh-CN-Xiaochen:DragonHDLatestNeural',
      avatar: '/assets/avatars/speech/placeholder-female.png',
      styles: [
        'Default',
      ],
    },
    {
      name: 'Xiaoxiao',
      value: 'zh-CN-XiaoxiaoNeural',
      avatar: '/assets/avatars/speech/zh-CN-XiaoxiaoNeural.png',
      styles: [
        'Default',
        'Assistant',
        'Chat',
        'Customer service',
        'Newscast',
        'Affectionate',
        'Angry',
        'Calm',
        'Cheerful',
        'Disgruntled',
        'Fearful',
        'Gentle',
        'Lyrical',
        'Sad',
        'Serious',
        'Poetry - reading',
        'Friendly',
        'Chat - casual',
        'Whispering',
        'Sorry',
        'Excited',
      ],
    },
    {
      name: 'Xiaoxiao Multilingual',
      value: 'zh-CN-XiaoxiaoMultilingualNeural',
      avatar: '/assets/avatars/speech/placeholder-female.png',
      styles: [
        'Default',
        'Affectionate',
        'Cheerful',
        'Empathetic',
        'Excited',
        'Poetry - reading',
        'Sorry',
        'Story',
      ],
    },
    {
      name: 'Yunye',
      value: 'zh-CN-YunyeNeural',
      avatar: '/assets/avatars/speech/zh-CN-YunyeNeural.png',
      styles: [
        'Default',
        'Embarrassed',
        'Calm',
        'Fearful',
        'Cheerful',
        'Disgruntled',
        'Serious',
        'Angry',
        'Sad',
      ],
    },
    {
      name: 'Yunjian',
      value: 'zh-CN-YunjianNeural',
      avatar: '/assets/avatars/speech/placeholder-male.png',
      styles: [
        'Default',
        'Narration - excited',
        'Sports commentary',
        'Sports - excited',
        'Angry',
        'Disgruntled',
        'Cheerful',
        'Sad',
        'Serious',
        'Depressed',
        'Documentary - narration',
      ],
    },
  ],
  'Taiwanese Mandarin': [
    {
      name: 'HsiaoChen',
      value: 'zh-TW-HsiaoChenNeural',
      avatar: '/assets/avatars/speech/placeholder-female.png',
      styles: ['Default'],
    },
    {
      name: 'YunJhe',
      value: 'zh-TW-YunJheNeural',
      avatar: '/assets/avatars/speech/placeholder-male.png',
      styles: ['Default'],
    },
    {
      name: 'HsiaoYu',
      value: 'zh-TW-HsiaoYuNeural',
      avatar: '/assets/avatars/speech/placeholder-female.png',
      styles: ['Default'],
    },
  ],
  English: [
    {
      name: 'Ava Multilingual',
      value: 'en-US-AvaMultilingualNeural',
      avatar: '/assets/avatars/speech/placeholder-female.png',
      styles: ['Default'],
    },
    {
      name: 'Andrew Multilingual',
      value: 'en-US-AndrewMultilingualNeural',
      avatar: '/assets/avatars/speech/placeholder-male.png',
      styles: ['Default', 'Empathetic', 'Relieved'],
    },
  ],
  Hindi: [
    {
      name: 'Aarav',
      value: 'hi-IN-AaravNeural',
      avatar: '/assets/avatars/speech/placeholder-male.png',
      styles: ['Default'],
    },
    {
      name: 'Ananya',
      value: 'hi-IN-AnanyaNeural',
      avatar: '/assets/avatars/speech/placeholder-female.png',
      styles: ['Default'],
    },
  ],
  Urdu: [
    {
      name: 'Gul',
      value: 'ur-IN-GulNeural',
      avatar: '/assets/avatars/speech/placeholder-female.png',
      styles: ['Default'],
    },

    {
      name: 'Salman',
      value: 'ur-IN-SalmanNeural',
      avatar: '/assets/avatars/speech/placeholder-male.png',
      styles: ['Default'],
    },
  ],
  Tamil: [
    {
      name: 'Pallavi',
      value: 'ta-IN-PallaviNeural',
      avatar: '/assets/avatars/speech/placeholder-female.png',
      styles: ['Default'],
    },
    {
      name: 'Valluvar',
      value: 'ta-IN-ValluvarNeural',
      avatar: '/assets/avatars/speech/placeholder-male.png',
      styles: ['Default'],
    },
  ],
  Marathi: [
    {
      name: 'Aarohi',
      value: 'mr-IN-AarohiNeural',
      avatar: '/assets/avatars/speech/placeholder-female.png',
      styles: ['Default'],
    },
    {
      name: 'Manohar',
      value: 'mr-IN-ManoharNeural',
      avatar: '/assets/avatars/speech/placeholder-male.png',
      styles: ['Default'],
    },
  ],
  Telugu: [
    {
      name: 'Shruti',
      value: 'te-IN-ShrutiNeural',
      avatar: '/assets/avatars/speech/placeholder-female.png',
      styles: ['Default'],
    },
    {
      name: 'Manohar',
      value: 'te-IN-MohanNeural',
      avatar: '/assets/avatars/speech/placeholder-male.png',
      styles: ['Default'],
    },
  ],
  Gujarati: [
    {
      name: 'Dhwani',
      value: 'gu-IN-DhwaniNeural',
      avatar: '/assets/avatars/speech/placeholder-female.png',
      styles: ['Default'],
    },
    {
      name: 'Niranjan',
      value: 'gu-IN-NiranjanNeural',
      avatar: '/assets/avatars/speech/placeholder-male.png',
      styles: ['Default'],
    },
  ],
  Kannada: [
    {
      name: 'Sapna',
      value: 'kn-IN-SapnaNeural',
      avatar: '/assets/avatars/speech/placeholder-female.png',
      styles: ['Default'],
    },
    {
      name: 'Gagan',
      value: 'kn-IN-GaganNeural',
      avatar: '/assets/avatars/speech/placeholder-male.png',
      styles: ['Default'],
    },
  ],
  Korean: [
    {
      name: 'Sun-Hi',
      value: 'ko-KR-SunHiNeural',
      avatar: '/assets/avatars/speech/placeholder-female.png',
      styles: ['Default'],
    },
    {
      name: 'InJoon',
      value: 'ko-KR-InJoonNeural',
      avatar: '/assets/avatars/speech/placeholder-male.png',
      styles: ['Default'],
    },
  ],
};
