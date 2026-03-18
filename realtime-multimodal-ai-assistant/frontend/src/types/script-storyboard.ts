// ----------------------------------------------------------------------

export type User = {
  id: string;
  name: string;
  avatar: string;
};

export type Storyboard = {
  id: string;
  name: string;
  content: string;
  tags: string[];
  createdBy: User[];
  createdAt: string;
  updatedAt: string;
  thumbnail?: string;
  duration?: number;
  frameCount?: number;
  title?: string;
  authors?: string;
  description?: string;
};

export type Script = {
  id: string;
  projectId: string;
  name: string;
  content: string;
  tags: string[];
  createdBy: User[];
  createdAt: string;
  updatedAt: string;
  storyboards: Storyboard[];
  status?: 'draft' | 'in_progress' | 'completed' | 'archived';
  category?: string;
  duration?: number;
  language?: string;
};

export type ScriptStoryboard = Omit<Script, 'projectId'>;
