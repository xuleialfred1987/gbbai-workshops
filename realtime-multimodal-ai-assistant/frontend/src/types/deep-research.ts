// ----------------------------------------------------------------------
// Research Types
// ----------------------------------------------------------------------

export interface ResearchAuthor {
  photoURL?: string;
  [key: string]: any;
}

export interface ResearchItem {
  id: string;
  title: string;
  author: ResearchAuthor;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  created: string;
  lastModified: string;
  content: string;
  tags: string[];
  timeConsumption: number;
  cover?: string;
  messages?: Array<{
    role: 'user' | 'assistant';
    content: string;
  }>;
}

export interface ResearchListItem {
  id: string;
  title: string;
  author: ResearchAuthor;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  created: string;
  lastModified: string;
  content: string;
  tags: string[];
  timeConsumption: number;
  cover?: string;
}

export interface ResearchMessage {
  role: 'user' | 'assistant';
  content: string;
}

// ----------------------------------------------------------------------
// API Payload Types
// ----------------------------------------------------------------------

export interface CreateResearchPayload {
  id?: string;
  title: string;
  author: ResearchAuthor;
  status?: 'queued' | 'processing' | 'completed' | 'failed';
  content?: string;
  tags?: string[];
  timeConsumption?: number;
  cover?: string;
}

export interface CreateResearchJobPayload {
  messages: ResearchMessage[];
  research_id?: string;
  metadata?: Record<string, any>;
}

export interface PaginationInfo {
  page_size: number;
  continuation_token: string | null;
  has_more: boolean;
  count: number;
}

// ----------------------------------------------------------------------
// API Response Types
// ----------------------------------------------------------------------

export interface ResearchListResponse {
  items: ResearchListItem[];
  pagination: PaginationInfo;
  success: boolean;
}

export interface ResearchResponse {
  item: ResearchItem;
  success: boolean;
  message?: string;
}

export interface CreateResearchResponse {
  message: string;
  item: ResearchItem;
  success: boolean;
}

export interface CreateResearchJobResponse {
  message: string;
  job_id: string;
  success: boolean;
}

export interface ApiResponse {
  message: string;
  success: boolean;
}

// ----------------------------------------------------------------------
// Thread Message Types
// ----------------------------------------------------------------------

export interface TextAnnotation {
  end_index: number;
  start_index: number;
  text: string;
  type: 'url_citation' | 'file_citation' | string;
  url_citation?: {
    title: string;
    url: string;
  };
  file_citation?: {
    file_id: string;
    quote?: string;
  };
}

export interface MessageContent {
  text?: {
    annotations: TextAnnotation[];
    value: string;
  };
  type: 'text' | 'image_file' | 'image_url';
  image_file?: {
    file_id: string;
  };
  image_url?: {
    url: string;
  };
}

export interface ThreadMessage {
  id: string;
  object: 'thread.message';
  created_at: number;
  thread_id: string;
  role: 'user' | 'assistant';
  content: MessageContent[];
  assistant_id?: string;
  run_id?: string;
  metadata: Record<string, any>;
  attachments?: any[];
}

export interface ResearchChatsResponse {
  item: ResearchItem & {
    threadId?: string;
    threadMessages?: ThreadMessage[];
  };
  success: boolean;
}

export interface DeepResearchMessage {
  id: string;
  body: string;
  role: 'user' | 'assistant';
  createdAt: Date;
  model?: string;
  annotations?: TextAnnotation[];
  attachments?: Array<{
    name: string;
    file_id?: string;
    url?: string;
  }>;
  metadata?: Record<string, any>;
}