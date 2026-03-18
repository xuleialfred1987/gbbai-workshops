// ----------------------------------------------------------------------

export type IVideoFilters = {
  publish: string;
};

export type IVideoItem = {
  id: string;
  title: string;
  tags: string[];
  videoId: string;
  coverId: string;
  publish: string;
  content: string;
  coverUrl: string;
  summary: string;
  videoUrl: string;
  indexName: string;
  totalViews: number;
  totalShares: number;
  description: string;
  totalComments: number;
  totalFavorites: number;
  metaKeywords: string[];
  metaDescription: string;
  sampleQuestions: IVideoSampleQuestion[];
  createdAt: Date;
  chapters: IVideoChapter[];
  author: {
    name: string;
    avatarUrl: string;
  };
  competitors?: string;
  script?: string;
};

export type IVideoChapter = {
  title: string;
  time: string;
  status: string;
  scenes: { title: string; description: string }[];
  endTime: string;
  snapshot: string;
  description: string;
};

export type IVideoSampleQuestion = {
  title: string;
  prompt: string;
};

export type IVideoClip = {
  id: string;
  title: string;
  video_id: string;
  videoUrl: string;
  start_time: string;
  end_time: string;
  scene_theme: string;
};
