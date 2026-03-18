// ----------------------------------------------------------------------

export type AssetType = 'prompts' | 'modeling' | 'keyframes' | 'voices' | 'videos';

export type AssetStatus = 'active' | 'inactive' | 'draft';

// Asset Types with translation keys
export const AssetTypeKeys = {
  PROMPTS: 'prompts',
  MODELING: 'modeling',
  KEYFRAMES: 'keyframes',
  VOICES: 'voices',
  VIDEOS: 'videos',
} as const;

export const AssetTypes = ['prompts', 'modeling', 'keyframes', 'voices', 'videos'];

// Base Asset Interface (shared properties)
export interface BaseAsset {
  id: string;
  name: string;
  description: string;
  category: string;
  subcategory?: string;
  status: AssetStatus;
  tags: string[];
  createdAt: Date | string;
  updatedAt: Date | string;
  createdBy: {
    id: string;
    name: string;
    avatar?: string;
  };
  content?: string;
  thumbnail?: string;
  isPublic: boolean;
  usageCount: number;
  projectId: string; // Required project ID
  isFavorite?: boolean; // Optional favorite flag
}

// Asset-specific interfaces
export interface PromptAsset extends BaseAsset {
  type: 'prompts';
  metadata?: {
    tone?: 'formal' | 'casual' | 'friendly' | 'professional';
    complexity?: 'simple' | 'intermediate' | 'advanced';
    targetAudience?: string[];
    industry?: string;
    [key: string]: any;
  };
}

export interface ModelingAsset extends BaseAsset {
  type: 'modeling';
  prompt: string; // Required prompt used to generate the modeling image
  imageUrl?: string; // Link to the specific modeling image
  thumbnailUrl?: string; // Link to the thumbnail version of the image (smaller/optimized)
  modelProvider?: 'openai' | 'gemini' | 'flux'; // AI model provider used to generate the image
  metadata?: {
    gender?: 'male' | 'female';
    modelType?: 'people' | 'nature' | 'objects' | 'architecture';
    occupation?: 'student' | 'teacher' | 'nurse' | 'lawyer' | 'engineer' | 'artist';
    age?: 'kid' | 'teenager' | 'adult' | 'elderly';
    ethnicity?: 'asian' | 'american' | 'european' | 'african' | 'indian';
    dressStyle?:
      | 'white_skirt'
      | 'black_skirt'
      | 'suit'
      | 'hoodie'
      | 'dress'
      | 'jeans'
      | 't_shirt'
      | 'blouse'
      | 'jacket'
      | 'sweater';
    bodyPart?: 'face' | 'whole_body' | 'upper_body' | 'lower_body' | 'hands' | 'portrait';
    dimensions?: {
      width: number;
      height: number;
      depth?: number;
    };
    renderQuality?: 'low' | 'medium' | 'high' | 'ultra';
    [key: string]: any;
  };
}

export interface KeyframeAsset extends BaseAsset {
  type: 'keyframes';
  prompt?: string;
  imageUrl?: string; // Link to the storyboard or keyframe preview image
  thumbnailUrl?: string;
  modelProvider?: string;
  metadata?: {
    style?: 'realistic' | 'cartoon' | 'abstract' | 'minimalist';
    mood?: 'happy' | 'dramatic' | 'calm' | 'energetic' | 'mysterious';
    setting?: 'indoor' | 'outdoor' | 'urban' | 'nature' | 'fantasy';
    action?: 'static' | 'movement' | 'transition' | 'effect';
    duration?: number; // in seconds
    frameRate?: number;
    contentType?:
      | 'content_type_commercial'
      | 'content_type_explainer'
      | 'content_type_tutorial'
      | 'content_type_entertainment'
      | 'content_type_documentary'; // Video content type
    targetAudience?:
      | 'target_audience_children'
      | 'target_audience_teenagers'
      | 'target_audience_adults'
      | 'target_audience_seniors'
      | 'target_audience_general'; // Target demographic
    resolution?: {
      width: number;
      height: number;
    };
    [key: string]: any;
  };
}

export interface VoiceAsset extends BaseAsset {
  type: 'voices';
  voiceUrl?: string; // Link to the voice audio file
  metadata?: {
    use?: 'narration' | 'dialogue' | 'advertisement' | 'tutorial' | 'audiobook' | 'podcast' | 'announcement' | 'character_voice' | 'background_voice' | 'onomatopoeia' | 'background_music' | 'other';
    gender?: 'male' | 'female' | 'neutral';
    age?: 'child' | 'young_adult' | 'adult' | 'elderly';
    accent?: 'american' | 'british' | 'australian' | 'neutral';
    tone?: 'formal' | 'casual' | 'friendly' | 'authoritative';
    mood?: 'happy' | 'sad' | 'excited' | 'calm' | 'angry' | 'neutral';
    language?: 'en-US' | 'zh-CN' | 'es-ES' | 'hi-IN' | 'ar-SA'; // Top 5 most common languages: English, Chinese, Spanish, Hindi, Arabic
    sampleRate?: number;
    bitRate?: number;
    audioFormat?: 'mp3' | 'wav' | 'flac';
    [key: string]: any;
  };
}

export interface VideoAsset extends BaseAsset {
  type: 'videos';
  subcategory?: string;
  taskStatus?: 'submitted' | 'processing' | 'succeed' | 'failed'; // Status of the video generation task
  videoUrl?: string; // Link to the video file
  thumbnailUrl?: string; // Link to the video thumbnail
  modelProvider?: 'openai' | 'gemini' | 'kling'; // AI model provider used to generate the image
  metadata?: {
    style?: 'realistic' | 'cartoon' | 'abstract' | 'minimalist' | 'cinematic';
    mood?: 'happy' | 'dramatic' | 'calm' | 'energetic' | 'mysterious' | 'inspiring';
    setting?: 'indoor' | 'outdoor' | 'urban' | 'nature' | 'fantasy' | 'studio';
    genre?: 'commercial' | 'documentary' | 'tutorial' | 'entertainment' | 'promotional';
    duration?: number; // in seconds
    frameRate?: number;
    resolution?: {
      width: number;
      height: number;
    };
    videoFormat?: 'mp4' | 'webm' | 'mov' | 'avi';
    quality?: 'low' | 'medium' | 'high' | 'ultra' | '4k';
    targetAudience?:
      | 'target_audience_children'
      | 'target_audience_teenagers'
      | 'target_audience_adults'
      | 'target_audience_seniors'
      | 'target_audience_general';
    [key: string]: any;
  };
}

// Discriminated Union - Main Asset Type
export type Asset = PromptAsset | ModelingAsset | KeyframeAsset | VoiceAsset | VideoAsset;

// Asset Categories by Type
export const AssetCategoriesByType = {
  prompts: {
    keys: {
      ALL: 'all',
      MARKETING: 'marketing',
      CREATIVE: 'creative',
      TECHNICAL: 'technical',
      EDUCATIONAL: 'educational',
    },
    // values: ['all', 'creative', 'marketing', 'technical', 'educational'],
    values: ['all', 'creative', 'marketing'],
    options: {
      creative: [
        'all',
        'storytelling',
        'copywriting',
        'brainstorming',
        'ideation',
        'dialogue_generation',
        'character_development',
        'scene_description',
        'storyboard_creation',
        'character_modeling',
        'world_building',
        'plot_development',
        'script_writing',
        'narrative_design',
      ],
      marketing: ['all', 'social_media', 'email', 'advertising', 'content'],
      // technical: ['all', 'documentation', 'tutorials', 'api_guides', 'specifications'],
      // educational: ['all', 'courses', 'tutorials', 'training', 'workshops'],
    },
  },
  modeling: {
    keys: {
      ALL: 'all',
      TYPE: 'type',
      GENDER: 'gender',
      OCCUPATION: 'occupation',
      AGE: 'age',
      ETHNICITY: 'ethnicity',
      DRESS_STYLE: 'dressStyle',
      BODY_PART: 'bodyPart',
    },
    values: ['all', 'type', 'gender', 'occupation', 'age', 'ethnicity', 'dressStyle', 'bodyPart'],
    options: {
      type: ['all', 'people', 'nature', 'objects', 'architecture'],
      gender: ['all', 'male', 'female'],
      occupation: ['all', 'student', 'teacher', 'nurse', 'lawyer', 'engineer', 'artist'],
      age: ['all', 'kid', 'teenager', 'adult', 'elderly'],
      ethnicity: ['all', 'asian', 'american', 'european', 'african', 'indian'],
      dressStyle: [
        'all',
        'white_skirt',
        'black_skirt',
        'suit',
        'hoodie',
        'dress',
        'jeans',
        't_shirt',
        'blouse',
        'jacket',
        'sweater',
      ],
      bodyPart: ['all', 'face', 'whole_body', 'upper_body', 'lower_body', 'hands', 'portrait'],
    },
  },
  keyframes: {
    keys: {
      ALL: 'all',
      STYLE: 'style',
      MOOD: 'mood',
      SETTING: 'setting',
      ACTION: 'action',
      CONTENT_TYPE: 'content_type',
      TARGET_AUDIENCE: 'target_audience',
    },
    values: ['all', 'style', 'mood', 'setting', 'action', 'content_type', 'target_audience'],
    options: {
      style: ['all', 'realistic', 'cartoon', 'abstract', 'minimalist'],
      mood: ['all', 'happy', 'dramatic', 'calm', 'energetic', 'mysterious'],
      setting: ['all', 'indoor', 'outdoor', 'urban', 'nature', 'fantasy'],
      action: ['all', 'static', 'movement', 'transition', 'effect'],
      content_type: [
        'all',
        'content_type_commercial',
        'content_type_explainer',
        'content_type_tutorial',
        'content_type_entertainment',
        'content_type_documentary',
      ],
      target_audience: [
        'all',
        'target_audience_children',
        'target_audience_teenagers',
        'target_audience_adults',
        'target_audience_seniors',
        'target_audience_general',
      ],
    },
  },
  voices: {
    keys: {
      ALL: 'all',
      USE: 'use',
      GENDER: 'gender',
      AGE: 'age',
      ACCENT: 'accent',
      TONE: 'tone',
      MOOD: 'mood',
      LANGUAGE: 'language',
    },
    values: ['all', 'use', 'gender', 'age', 'accent', 'tone', 'mood', 'language'],
    options: {
      use: ['all', 'narration', 'dialogue', 'advertisement', 'tutorial', 'audiobook', 'podcast', 'announcement', 'character_voice', 'background_voice', 'onomatopoeia', 'background_music', 'other'],
      gender: ['all', 'male', 'female', 'neutral'],
      age: ['all', 'child', 'young_adult', 'adult', 'elderly'],
      accent: ['all', 'american', 'british', 'australian', 'neutral'],
      tone: ['all', 'formal', 'casual', 'friendly', 'authoritative'],
      mood: ['all', 'happy', 'sad', 'excited', 'calm', 'angry', 'neutral'],
      language: ['all', 'en-US', 'zh-CN', 'es-ES', 'hi-IN', 'ar-SA'],
    },
  },
  videos: {
    keys: {
      ALL: 'all',
      STYLE: 'style',
      MOOD: 'mood',
      SETTING: 'setting',
      GENRE: 'genre',
      QUALITY: 'quality',
      TARGET_AUDIENCE: 'target_audience',
    },
    values: ['all', 'style', 'mood', 'setting', 'genre', 'quality', 'target_audience'],
    options: {
      style: ['all', 'realistic', 'cartoon', 'abstract', 'minimalist', 'cinematic'],
      mood: ['all', 'happy', 'dramatic', 'calm', 'energetic', 'mysterious', 'inspiring'],
      setting: ['all', 'indoor', 'outdoor', 'urban', 'nature', 'fantasy', 'studio'],
      genre: ['all', 'commercial', 'documentary', 'tutorial', 'entertainment', 'promotional'],
      quality: ['all', 'low', 'medium', 'high', 'ultra', '4k'],
      target_audience: [
        'all',
        'target_audience_children',
        'target_audience_teenagers',
        'target_audience_adults',
        'target_audience_seniors',
        'target_audience_general',
      ],
    },
  },
};

export type AssetTypeCategory = (typeof AssetTypes)[number];

// ----------------------------------------------------------------------
// Create/Update Data Types

export interface BaseCreateAssetData {
  name: string;
  description: string;
  category: string;
  tags: string[];
  content?: string;
  thumbnail?: string;
  isPublic?: boolean;
}

export interface CreatePromptAssetData extends BaseCreateAssetData {
  type: 'prompts';
  metadata?: PromptAsset['metadata'];
}

export interface CreateModelingAssetData extends BaseCreateAssetData {
  type: 'modeling';
  prompt: string; // Required prompt used to generate the modeling image
  imageUrl?: string;
  metadata?: ModelingAsset['metadata'];
}

export interface CreateKeyframeAssetData extends BaseCreateAssetData {
  type: 'keyframes';
  imageUrl?: string;
  metadata?: KeyframeAsset['metadata'];
}

export interface CreateVoiceAssetData extends BaseCreateAssetData {
  type: 'voices';
  voiceUrl?: string;
  metadata?: VoiceAsset['metadata'];
}

export interface CreateVideoAssetData extends BaseCreateAssetData {
  type: 'videos';
  videoUrl?: string;
  thumbnailUrl?: string;
  metadata?: VideoAsset['metadata'];
}

export type CreateAssetData =
  | CreatePromptAssetData
  | CreateModelingAssetData
  | CreateKeyframeAssetData
  | CreateVoiceAssetData
  | CreateVideoAssetData;

export interface UpdateAssetData extends Partial<Omit<BaseCreateAssetData, 'type'>> {
  status?: AssetStatus;
  metadata?: Record<string, any>;
}

// ----------------------------------------------------------------------

export interface AssetFilters {
  query: string;
  category: string;
  type: AssetType | 'all';
  status: AssetStatus | 'all';
  tags: string[];
  author?: string;
}
