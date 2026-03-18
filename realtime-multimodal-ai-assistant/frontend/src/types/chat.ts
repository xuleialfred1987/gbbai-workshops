// ----------------------------------------------------------------------

export const IAgentMsgPrefix = '[AGENT]';
export const IPlannerMsgPrefix = '[SYS]';
export const ISystemMsgPrefix = '(SYS)';

export type IChatMode = 'open-chat' | 'rag' | 'function-calling';
export type ISenderRole = 'assistant' | 'user';

export type Contact = {
  id: string;
  name: string;
  username: string;
  avatar: string;
  address: string;
  phone: string;
  email: string;
  lastActivity: Date | string | number;
  status: string;
  position: string;
};

export type SendTextFuncProps = {
  content: string;
  senderId: string;
  mode: string;
  sources?: any[];
  function_calls?: any[];
  msgId?: string;
  attachments?: any[];
  uuid?: string;
  timestamp?: string;
  buttonPrompt?: string;
  thoughts?: any[];
  status?: string;
  chatMode?: string;
  senderRole?: ISenderRole;
};

export type Conversation = {
  id: string;
  participants: Participant[];
  type: string;
  unreadCount: number;
  messages: Message[];
};

export type ICaption = {
  timestamp: string | number;
  caption: string;
};

export type IToolDefinition = {
  name: string;
  description: Participant[];
  parameters: { type: string; properties: object; required: string[] }[];
};

export type IConfiguration = {
  'open-chat-Deployment': string;
  'open-chat-Past messages included': string;
  'open-chat-Temperature': string;
  'open-chat-Top P': string;
  'open-chat-Max response': string;
  'open-chat-Stop sequence': string;
  'open-chat-Frequency penalty': string;
  'open-chat-Presence penalty': string;
  'open-chat-System message': string;
  'open-chat-Should stream': boolean;

  'rag-Deployment': string;
  'rag-Past messages included': string;
  'rag-Temperature': string;
  'rag-Top P': string;
  'rag-Max response': string;
  'rag-Stop sequence': string;
  'rag-Frequency penalty': string;
  'rag-Presence penalty': string;
  'rag-Retrieve count': string;
  'rag-System message': string;
  'rag-Should stream': boolean;
  'rag-Retrieval mode': string;
  'rag-Use semantic ranker': boolean;

  'function-calling-Deployment': string;
  'function-calling-Past messages included': string;
  'function-calling-Temperature': string;
  'function-calling-Top P': string;
  'function-calling-Max response': string;
  'function-calling-Stop sequence': string;
  'function-calling-Frequency penalty': string;
  'function-calling-Presence penalty': string;
  'function-calling-System message': string;
  'function-calling-Should stream': boolean;

  // Additinal configurations
  selectedIndex?: string;
  selectedTools?: string[];

  [key: string]: any;
};

export type IRtConfiguration = {
  'rt-Deployment': string;
  'rt-Temperature': string;
  'rt-Voice choice': string;
  'rt-Voice host': string;
  'rt-Max response': string;
  'rt-Disable audio': boolean;
  'rt-System message': string;
  'rt-VAD Threshold': string;
  'rt-VAD Prefix Padding (ms)': string;
  'rt-VAD Silence Duration (ms)': string;

  [key: string]: any;
};

export type SendMessage = {
  conversationId?: string;
  messageId: string;
  message: string;
  contentType: 'text' | 'image';
  sources: { label: string; url: string }[];
  function_calls?: {
    funcName: string;
    funcArgs?: string;
    results: any;
    status?: 'running' | 'completed' | 'error';
    callId?: string;
    previousItemId?: string;
  }[];
  createdAt: Date | string | number;
  senderId: string;
  mode: string;
  chatMode: string;
  ddb_uuid?: string;
  log_timestamp?: string;
  attachments?: { type: string; url: string }[];
  thoughts?: any[];
  realtimeItemId?: string;
};

export type ChatbotConfigStruct = {
  name: string;
  type: string;
  description: string;
  model: ChatbotModel;
};

export type ChatbotModel = {
  alias: string;
  endpoint: string;
  model: string;
  pipeline: string;
  time: number;
  source: string;
};

export type ChatbotResponseAttachment = {
  file_name: string;
  file_location: string;
  key_name?: string;
  bucket_name?: string;
};

export type ChatbotResponse = {
  Answer: string;
  Buttons: MessageButton[];
  Attachments: ChatbotResponseAttachment[];
  Type: string;
  SimilarScore: number;
  SimilarText: string;
  NeedFeedback: boolean;
  IsApi: boolean;
  ApiInfo: ChatbotResponseApi;
  Topic: string;
};

export type ChatbotResponseApi = {
  RequestMethod: string;
  RequestUri: string;
  RequestBody: string;
  Parameters: [{ name: string; defaultValue: string }];
  ResponseTemplate: string;
  OutputFields: string[];
};

export type IChatAttachment = {
  name: string;
  size: number;
  type: string;
  path: string;
  preview: string;
  createdAt: Date;
  modifiedAt: Date;
};

export type IChatMessage = {
  // id: string;
  // body: string;
  // createdAt: Date;
  // senderId: string;
  // contentType: string;
  // attachments: IChatAttachment[];

  id: string;
  body: string;
  mode: string;
  createdAt: Date;
  senderId: string;
  chatMode: string;
  contentType: 'text' | 'image' | 'video';
  sources: { label: string; url: string }[];
  //
  query?: string;
  status?: string;
  cost?: GptCost[];
  agents?: Agent[];
  thoughts?: any[];
  ddb_uuid?: string;
  avatarUrl?: string;
  attachments?: any[];
  files?: any[];
  avatarName?: string;
  log_timestamp?: string;
  senderRole?: ISenderRole;
  buttons?: MessageButton[];
  agentMessages?: AgentMessage[];
  videos?: { snippet: string; url: string }[];
  function_calls?: {
    funcName: string;
    funcArgs?: string;
    results: any;
    status?: 'running' | 'completed' | 'error';
    callId?: string;
    previousItemId?: string;
  }[];
};

export type IAgentFleet = {
  id: string;
  type: string;
  unreadCount: number;
  messages: IChatMessage[];
  participants: IChatParticipant[];
  // for non-autogen mode
  framework?: 'autogen' | 'none';
  planner?: string;
};

export type IChatParticipant = {
  id: string;
  name: string;
  avatarUrl: string;
  lastActivity: Date;
  status: string;
};

export type IChatConversation = {
  id: string;
  type: string;
  unreadCount: number;
  messages: IChatMessage[];
  participants: IChatParticipant[];
  // for non-autogen mode
  framework?: 'autogen' | 'none';
  planner?: string;
  // for group chat
  title?: string;
  description?: string;
};

export type IChatConversations = {
  byId: Record<string, IChatConversation>;
  allIds: string[];
};

export type Participant = {
  id: string;
  name: string;
  username: string;
  avatarUrl: string;
  address?: string;
  phone?: string;
  email?: string;
  lastActivity?: Date | string | number;
  status?: 'online' | 'offline' | 'away' | 'busy';
  position?: string;
  type?: string;
  model?: ChatbotModel;
  description?: string;
};

export type Agent = {
  name: string;
  avatar: string;
  description: string;
};

export type GptCost = {
  name: string;
  items: Record<string, string | number>;
};

export type AgentMessage = {
  name: string;
  agentId?: string;
  content: string;
  role: string;
  createdAt: Date | string;
  status?: string;
  //
  function_calls?: {
    funcName: string;
    funcArgs?: string;
    results: any;
    status?: 'running' | 'completed' | 'error';
    callId?: string;
    previousItemId?: string;
  }[];
};

export type TextMessage = {
  id: string;
  body: string;
  mode: string;
  createdAt: Date;
  senderId: string;
  chatMode: string;
  contentType: 'text' | 'image' | 'video';
  sources: { label: string; url: string }[];
  //
  query?: string;
  status?: string;
  cost?: GptCost[];
  agents?: Agent[];
  thoughts?: any[];
  ddb_uuid?: string;
  avatarUrl?: string;
  attachments?: any[];
  senderRole?: ISenderRole;
  files?: any[]; // for agent generated files
  avatarName?: string;
  log_timestamp?: string;
  buttons?: MessageButton[];
  agentMessages?: AgentMessage[];
  videos?: { snippet: string; url: string }[];
  function_calls?: {
    funcName: string;
    funcArgs?: string;
    results: any;
    status?: 'running' | 'completed' | 'error';
    callId?: string;
    previousItemId?: string;
  }[];
  ruleChecks?: any[]; // PCB rule check results
  realtimeItemId?: string;
};

export type MessageButton = {
  button_id: string;
  button_content: string;
  button_prompt: string;
};

export const enum RetrievalMode {
  Text = 'text',
  Hybrid = 'hybrid',
  Vectors = 'vectors',
}

export type ChatAppRequest = {
  session_state: any;
  messages: ResponseMessage[];
  //
  stream?: boolean;
  context?: ChatAppRequestContext;
};

export type ChatAppRequestContext = {
  functions?: any;
  overrides?: ChatAppRequestOverrides;
};

export type ChatAppRequestOverrides = {
  top?: number;
  max_tokens?: number;
  temperature?: number;
  retrieval_mode?: string;
  prompt_template?: string;
  semantic_ranker?: boolean;
  exclude_category?: string;
  semantic_captions?: boolean;
  prompt_template_prefix?: string;
  prompt_template_suffix?: string;
  use_oid_security_filter?: boolean;
  use_groups_security_filter?: boolean;
  suggest_followup_questions?: boolean;
};

export type ChatAppResponse = {
  choices: ResponseChoice[];
};

export type ChatAppResponseOrError = {
  choices?: ResponseChoice[];
  error?: string;
};

export type ResponseChoice = {
  index: number;
  session_state: any;
  finish_reason: string;
  message: ResponseMessage;
  context: ResponseContext;
};

export type ResponseContext = {
  data_points: string[];
  thoughts: string | null;
  followup_questions: string[] | null;
};

export type ResponseMessage = {
  content: string | any[];
  role: string;
  name?: string;
  function_call?: { name: string; arguments: string };
  attachments?: any[];
  sources?: any[];
};

export type IVideoClip = {
  id: string;
  content: string;
  summary: string;
  end_time: string;
  start_time: string;
  key_objects: string;
  onCaptureFrame: (time: string) => Promise<void>;
  search_score: number;
  search_reranker_score: number;
};

export type Message = TextMessage;
