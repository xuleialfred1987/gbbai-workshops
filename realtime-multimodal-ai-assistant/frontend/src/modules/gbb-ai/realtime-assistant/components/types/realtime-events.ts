export type GroundingFile = {
  id: string;
  name: string;
  content: string;
};

export type HistoryItem = {
  id: string;
  transcript: string;
  groundingFiles: GroundingFile[];
};

export type SessionUpdateCommand = {
  type: 'session.update';
  session: {
    turn_detection?: {
      type: 'server_vad' | 'none';
    };
    input_audio_transcription?: {
      model: 'whisper-1';
    };
  };
};

export type InputTextCommand = {
  type: 'conversation.item.create';
  item: {
    type: 'message';
    role: 'user' | 'assistant' | 'system';
    content: [
      {
        type: 'input_text';
        text: string;
      },
    ];
  };
};

export type InputAudioBufferAppendCommand = {
  type: 'input_audio_buffer.append';
  audio: string;
};

export type ConversationItemCreated = {
  type: 'conversation.item.created';
  event_id: string;
  item: {
    id: string;
    object: string;
    type: string;
    status: string;
    role: 'user' | 'assistant' | 'system';
    content?: any[];
  };
  previous_item_id?: string;
};

export type ConversationItemTruncateCommand = {
  type: 'conversation.item.truncate';
  item_id: string;
  content_index: number;
  audio_end_ms: number;
};

export type ConversationItemDeleteCommand = {
  type: 'conversation.item.delete';
  item_id: string;
};

export type ResponseCancelCommand = {
  type: 'response.cancel';
};

export type InputAudioBufferClearCommand = {
  type: 'input_audio_buffer.clear';
};

export type Message = {
  type: string;
};

export type WebSocketErrorPayload = {
  type: 'error';
  event_id: string | null;
  error: {
    code: string;
    message: string;
    param: string | null;
    type: string;
  };
};

export type ResponseAudioDelta = {
  type: 'response.audio.delta';
  delta: string;
};

export type ResponseAudioTranscriptDelta = {
  type: 'response.audio_transcript.delta';
  response_id: string;
  event_id: string;
  item_id: string;
  delta: string;
};

export type ResponseInputAudioTranscriptionCompleted = {
  type: 'conversation.item.input_audio_transcription.completed';
  event_id: string;
  item_id: string;
  content_index: number;
  transcript: string;
};

export type ResponseDone = {
  type: 'response.done';
  event_id: string;
  response: {
    id: string;
    output: { id: string; content?: { transcript: string; type: string }[] }[];
  };
};

export type ExtensionMiddleTierToolResponse = {
  type: 'extension.middle_tier_tool_response';
  previous_item_id?: string;
  tool_name: string;
  tool_result: string;
  call_id?: string;
};

export type ExtensionMiddleTierToolStatus = {
  type: 'extension.middle_tier_tool_status';
  previous_item_id?: string;
  tool_name: string;
  call_id?: string;
  status: 'running';
};

export type ToolResult = {
  sources: { chunk_id: string; title: string; chunk: string }[];
};