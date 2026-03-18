import os
import asyncio
import base64
import json
import logging
from collections import defaultdict
from typing import Union, Optional, Any, Dict, List, Callable, Awaitable
from aiohttp import web
from azure.core.credentials import AzureKeyCredential, TokenCredential

# Azure VoiceLive SDK imports
from azure.ai.voicelive.aio import connect
try:
    from azure.ai.voicelive.models import (
        RequestSession,
        ServerVad,
        AzureStandardVoice,
        Modality,
        AudioInputTranscriptionOptions,
        InputAudioFormat,
        OutputAudioFormat,
        ServerEventType,
        FunctionCallOutputItem,
        ResponseFunctionCallItem,
        AudioEchoCancellation,
        AudioNoiseReduction,
    )
except ImportError:
    from azure.ai.voicelive.models import (
        RequestSession,
        ServerVad,
        AzureStandardVoice,
        Modality,
        AudioInputTranscriptionOptions,
        InputAudioFormat,
        OutputAudioFormat,
        ServerEventType,
        FunctionCallOutputItem,
        ResponseFunctionCallItem,
    )

    AudioEchoCancellation = None
    AudioNoiseReduction = None

from realtime_tools import ToolResult, ToolResultDirection
from prompts import build_opening_greeting_instruction

logger = logging.getLogger("VOICE-LIVE-HANDLER")


def _voice_family(voice: Optional[str]) -> str:
    normalized_voice = (voice or "").strip()
    if not normalized_voice:
        return "unknown"
    if ":" in normalized_voice or "-" in normalized_voice:
        return "azure"
    return "openai"


def _build_voice_config(voice: str) -> Union[AzureStandardVoice, str]:
    if _voice_family(voice) == "azure":
        logger.info("Using Azure voice: %s", voice)
        return AzureStandardVoice(name=voice, type="azure-standard")

    logger.info("Using OpenAI-style voice: %s", voice)
    return voice


def _build_audio_cleanup_config() -> Dict[str, Any]:
    session_kwargs: Dict[str, Any] = {}

    if AudioEchoCancellation is not None:
        try:
            session_kwargs["input_audio_echo_cancellation"] = AudioEchoCancellation()
            logger.info("Enabled VoiceLive input audio echo cancellation")
        except Exception as exc:
            logger.warning("Failed to configure VoiceLive echo cancellation: %s", exc)
    else:
        logger.info(
            "VoiceLive SDK does not expose AudioEchoCancellation; skipping echo cancellation config"
        )

    if AudioNoiseReduction is not None:
        try:
            session_kwargs["input_audio_noise_reduction"] = AudioNoiseReduction(
                type="azure_deep_noise_suppression"
            )
            logger.info("Enabled VoiceLive deep noise suppression")
        except Exception as exc:
            logger.warning("Failed to configure VoiceLive noise reduction: %s", exc)
    else:
        logger.info(
            "VoiceLive SDK does not expose AudioNoiseReduction; skipping noise reduction config"
        )

    return session_kwargs


class WebSocketAudioBridge:
    """
    Bridges audio data between client WebSocket and VoiceLive connection.

    In a web application:
    - Audio capture happens in the browser (client-side)
    - Audio playback happens in the browser (client-side)  
    - Server just forwards audio data between client and VoiceLive API
    """

    def __init__(self, voice_live_connection, client_ws: web.WebSocketResponse):
        self.voice_live_connection = voice_live_connection
        self.client_ws = client_ws
        self.is_active = False
        self.client_audio_chunk_count = 0
        self._forward_task: Optional[asyncio.Task] = None
        self._client_audio_queue: asyncio.Queue[str] = asyncio.Queue()
        logger.info(
            "WebSocketAudioBridge initialized for web-based audio streaming")

    async def start(self):
        """Start the audio bridge."""
        self.is_active = True
        self._forward_task = asyncio.create_task(
            self._forward_client_audio_loop())
        logger.info(
            "Audio bridge started - ready to forward audio between client and VoiceLive")

    async def stop(self):
        """Stop the audio bridge."""
        self.is_active = False
        if self._forward_task and not self._forward_task.done():
            self._forward_task.cancel()
            try:
                await self._forward_task
            except asyncio.CancelledError:
                pass
        self._forward_task = None
        logger.info("Audio bridge stopped")

    async def _forward_client_audio_loop(self):
        try:
            while True:
                audio_base64 = await self._client_audio_queue.get()
                try:
                    await self.voice_live_connection.input_audio_buffer.append(audio=audio_base64)
                finally:
                    self._client_audio_queue.task_done()
        except asyncio.CancelledError:
            raise

    async def wait_for_audio_flush(self):
        await self._client_audio_queue.join()

    async def forward_client_audio_to_voicelive(self, audio_base64: str):
        """Queue audio from client for forwarding to VoiceLive."""
        if self.is_active:
            try:
                self.client_audio_chunk_count += 1
                await self._client_audio_queue.put(audio_base64)
            except Exception as e:
                message = str(e)
                if "closing transport" in message.lower():
                    logger.info(
                        "Skipping audio forward because VoiceLive transport is closing")
                    self.is_active = False
                else:
                    logger.error(f"Error forwarding audio to VoiceLive: {e}")

    async def forward_voicelive_audio_to_client(self, audio_data: bytes):
        """Forward audio from VoiceLive to client."""
        if self.is_active and not self.client_ws.closed:
            try:
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                await self.client_ws.send_json({
                    "type": "response.audio.delta",
                    "response_id": "resp_001",
                    "item_id": "item_001",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": audio_base64
                })
            except Exception as e:
                logger.error(f"Error forwarding audio to client: {e}")

    async def cleanup(self):
        """Clean up resources."""
        await self.stop()
        logger.info("Audio bridge cleaned up")


class VoiceLiveWebSocketHandler:
    """
    WebSocket handler that integrates Azure VoiceLive SDK with the existing web application.
    This provides a bridge between the web interface and the voice live API.
    """

    def __init__(
        self,
        endpoint: str,
        credential: Union[AzureKeyCredential, TokenCredential],
        model: str,
        voice: str,
        instructions: str,
        tools: List[Dict[str, Any]] = None,
        tool_handlers: Optional[Dict[str, Callable[[
            Any], Awaitable[ToolResult]]]] = None,
    ):
        self.endpoint = endpoint
        self.credential = credential
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.tools = tools or []
        self.tool_handlers: Dict[str, Callable[[Any],
                                               Awaitable[ToolResult]]] = tool_handlers or {}
        self.active_connections: Dict[str, Any] = {}
        self._pending_function_calls: Dict[str, Dict[str, Any]] = {}
        self._function_call_arguments: Dict[str, str] = defaultdict(str)

    async def _request_opening_greeting(self, connection) -> None:
        try:
            from azure.ai.voicelive.models import UserMessageItem, InputTextContentPart

            await connection.conversation.item.create(
                item=UserMessageItem(
                    role='user',
                    content=[
                        InputTextContentPart(
                            text=build_opening_greeting_instruction(),
                        )
                    ],
                )
            )
            await connection.response.create()
            logger.info("Requested opening greeting response")
        except Exception as exc:
            logger.error("Failed to request opening greeting: %s", exc)

    async def _setup_session(self, connection):
        """Configure the VoiceLive session for audio conversation."""
        logger.info("Setting up voice conversation session...")
        logger.info(f"Voice choice: {self.voice}")

        voice_config = _build_voice_config(self.voice)

        # Create strongly typed turn detection configuration
        turn_detection_config = ServerVad(
            threshold=0.55,
            prefix_padding_ms=180,
            silence_duration_ms=220,
            create_response=True,
            interrupt_response=True,
        )

        # Enable audio transcription
        transcription_config = AudioInputTranscriptionOptions(
            model="whisper-1"
        )
        audio_cleanup_config = _build_audio_cleanup_config()

        # Create strongly typed session configuration
        logger.info(
            f"Voice config type: {type(voice_config)}, value: {voice_config}")
        logger.info("VoiceLive system prompt in use:\n%s", self.instructions)

        session_config = RequestSession(
            modalities=[Modality.TEXT, Modality.AUDIO],
            instructions=self.instructions,
            voice=voice_config,
            input_audio_format=InputAudioFormat.PCM16,
            output_audio_format=OutputAudioFormat.PCM16,
            turn_detection=turn_detection_config,
            input_audio_transcription=transcription_config,
            tools=self.tools if self.tools else None,
            **audio_cleanup_config,
        )

        logger.info("Session config created, sending to VoiceLive...")
        try:
            await connection.session.update(session=session_config)
            logger.info("Session configuration sent successfully")
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            logger.error(f"Session config: {session_config}")
            raise

    async def _execute_tool_call(
        self,
        call_id: str,
        tool_name: Optional[str],
        arguments_str: str,
        connection,
        ws: web.WebSocketResponse,
        pending_metadata: Dict[str, Any],
        connection_id: str,
    ) -> None:
        """Execute a tool call emitted by VoiceLive and forward results."""
        pending_metadata = pending_metadata or {}
        resolved_tool_name = tool_name or pending_metadata.get("name")

        if not resolved_tool_name:
            logger.warning(
                "Received function call without a name (call_id=%s)", call_id)
            return

        handler = self.tool_handlers.get(resolved_tool_name)
        tool_result: ToolResult

        try:
            parsed_args = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse arguments for tool '%s': %s",
                resolved_tool_name,
                exc,
            )
            tool_result = ToolResult(
                json.dumps({
                    "error": f"Invalid arguments for tool '{resolved_tool_name}'.",
                    "details": str(exc),
                }),
                ToolResultDirection.TO_CLIENT,
            )
        else:
            if handler is None:
                logger.error(
                    "Tool '%s' was invoked but no handler is registered.",
                    resolved_tool_name,
                )
                tool_result = ToolResult(
                    json.dumps({
                        "error": f"Tool '{resolved_tool_name}' is not available.",
                    }),
                    ToolResultDirection.TO_CLIENT,
                )
            else:
                try:
                    if resolved_tool_name == "transfer_to_live_agent":
                        parsed_args["_session_context"] = {
                            "conversation_transcript": self.active_connections.get(connection_id, {}).get(
                                "conversation_transcript",
                                [],
                            )
                        }
                    handler_result = handler(parsed_args)
                    if asyncio.iscoroutine(handler_result):
                        handler_result = await handler_result
                except Exception as exc:
                    logger.exception(
                        "Error executing tool '%s'", resolved_tool_name)
                    tool_result = ToolResult(
                        json.dumps({
                            "error": f"Tool '{resolved_tool_name}' execution failed.",
                            "details": str(exc),
                        }),
                        ToolResultDirection.TO_CLIENT,
                    )
                else:
                    if isinstance(handler_result, ToolResult):
                        tool_result = handler_result
                    elif isinstance(handler_result, str):
                        tool_result = ToolResult(
                            handler_result,
                            ToolResultDirection.TO_SERVER,
                        )
                    else:
                        tool_result = ToolResult(
                            json.dumps(handler_result),
                            ToolResultDirection.TO_SERVER,
                        )

        model_output = (
            tool_result.to_text()
            if tool_result.destination == ToolResultDirection.TO_SERVER
            else ""
        )

        try:
            output_item = FunctionCallOutputItem(
                call_id=call_id,
                output=model_output,
                type="function_call_output",
            )
            await connection.conversation.item.create(item=output_item)
        except Exception:
            logger.exception(
                "Failed to send tool output for '%s' to VoiceLive", resolved_tool_name)
        else:
            try:
                await connection.response.create()
            except Exception as exc:
                logger.error(
                    "Failed to request follow-up response after tool '%s': %s",
                    resolved_tool_name,
                    exc,
                )

        client_payload = {
            "type": "extension.middle_tier_tool_response",
            "tool_name": resolved_tool_name,
            "tool_result": tool_result.to_text(),
            "call_id": call_id,
        }
        if previous := pending_metadata.get("item_id"):
            client_payload["previous_item_id"] = previous

        await ws.send_json(client_payload)

    async def _handle_voice_live_event(self, event, ws: web.WebSocketResponse, audio_bridge, connection, connection_id: str):
        """Handle different types of events from VoiceLive."""
        logger.debug(f"Received VoiceLive event: {event.type}")

        if event.type == ServerEventType.RESPONSE_OUTPUT_ITEM_ADDED:
            response_item = getattr(event, "item", None)
            if isinstance(response_item, ResponseFunctionCallItem):
                call_id = response_item.call_id
                self._pending_function_calls[call_id] = {
                    "name": response_item.name,
                    "item_id": getattr(response_item, "id", None),
                    "response_id": getattr(event, "response_id", None),
                    "output_index": getattr(event, "output_index", None),
                }
                initial_arguments = getattr(response_item, "arguments", None)
                if initial_arguments:
                    self._function_call_arguments[call_id] += initial_arguments
                logger.info(
                    "Registered pending tool call '%s' (call_id=%s)",
                    response_item.name,
                    call_id,
                )
                await ws.send_json({
                    "type": "extension.middle_tier_tool_status",
                    "tool_name": response_item.name,
                    "call_id": call_id,
                    "status": "running",
                    "previous_item_id": getattr(response_item, "id", None),
                })
                return

        if event.type == ServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA:
            self._function_call_arguments[event.call_id] += event.delta
            return

        if event.type == ServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
            pending = self._pending_function_calls.pop(event.call_id, {})
            accumulated_arguments = self._function_call_arguments.pop(
                event.call_id, "")
            final_arguments = event.arguments or accumulated_arguments
            await self._execute_tool_call(
                call_id=event.call_id,
                tool_name=getattr(event, "name", None),
                arguments_str=final_arguments,
                connection=connection,
                ws=ws,
                pending_metadata=pending,
                connection_id=connection_id,
            )
            return

        # Convert event to the legacy client event shape.
        if hasattr(event, 'as_dict'):
            event_data = event.as_dict()
        else:
            # Fallback to manual serialization
            event_data = {
                "type": event.type.lower().replace('_', '.'),
                **{k: v for k, v in vars(event).items() if not k.startswith('_')}
            }

        logger.debug(f"Forwarding event: {event_data}")

        # Special handling for certain event types
        if event.type == ServerEventType.SESSION_UPDATED:
            logger.info("Session ready")
            connection_state = self.active_connections.get(connection_id, {})
            if not connection_state.get("opening_greeting_sent"):
                connection_state["opening_greeting_sent"] = True
                self.active_connections[connection_id] = connection_state
                await self._request_opening_greeting(connection)
            # Forward the original session event
            await ws.send_json(event_data)

        elif event.type == ServerEventType.RESPONSE_AUDIO_DELTA:
            # Only forward the original event - don't duplicate audio!
            logger.debug("Processing audio delta")
            await ws.send_json(event_data)

        elif event.type in [
            ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED,
            ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED,
            ServerEventType.RESPONSE_CREATED,
            ServerEventType.RESPONSE_AUDIO_DONE,
            ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA,
            ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED,
            ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DELTA,
            ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DONE,
            ServerEventType.RESPONSE_DONE,
            ServerEventType.CONVERSATION_ITEM_CREATED,
        ]:
            # Forward these events as-is to the client.
            logger.debug(f"Forwarding event: {event.type}")
            await ws.send_json(event_data)

        elif event.type == ServerEventType.ERROR:
            logger.error(f"❌ VoiceLive error: {event.error.message}")

            # Send error to client
            await ws.send_json({
                "type": "error",
                "error": {
                    "type": "server_error",
                    "code": "voice_live_error",
                    "message": event.error.message
                }
            })

            if "maximum duration" in event.error.message.lower():
                raise RuntimeError(event.error.message)

        elif event.type == ServerEventType.CONVERSATION_ITEM_CREATED:
            logger.debug(f"Conversation item created: {event.item.id}")

            # Send conversation item created to client
            await ws.send_json({
                "type": "conversation.item.created",
                "previous_item_id": None,
                "item": {
                    "id": event.item.id,
                    "object": "realtime.item"
                }
            })

        else:
            logger.debug(f"Unhandled event type: {event.type}")

    async def _handle_client_message(self, message: dict, connection, audio_bridge, connection_id: str):
        """Handle messages from the client WebSocket."""
        msg_type = message.get('type')

        if msg_type == "session.update":
            # Apply supported session settings from the client to the live VoiceLive session.
            session_data = message.get('session', {})
            session_update = {}

            if 'instructions' in session_data:
                if self.instructions:
                    logger.info(
                        "Ignoring client instructions because server instructions are configured"
                    )
                else:
                    self.instructions = session_data['instructions']
                    session_update['instructions'] = self.instructions

            if 'voice' in session_data and session_data['voice']:
                requested_voice = str(session_data['voice']).strip()
                if requested_voice and requested_voice != self.voice:
                    current_family = _voice_family(self.voice)
                    requested_family = _voice_family(requested_voice)

                    if self.voice and current_family != requested_family:
                        logger.warning(
                            "Ignoring incompatible live voice update without reconnect: %s (%s) -> %s (%s)",
                            self.voice,
                            current_family,
                            requested_voice,
                            requested_family,
                        )
                    else:
                        logger.info(
                            "Applying client voice update: %s -> %s",
                            self.voice,
                            requested_voice,
                        )
                        self.voice = requested_voice
                        session_update['voice'] = _build_voice_config(self.voice)
                elif requested_voice:
                    session_update['voice'] = _build_voice_config(self.voice)

            turn_detection = session_data.get('turn_detection')
            if isinstance(turn_detection, dict) and turn_detection.get('type') == 'server_vad':
                silence_duration_ms = int(
                    turn_detection.get('silence_duration_ms', 220))
                session_update['turn_detection'] = {
                    'type': 'server_vad',
                    'threshold': float(turn_detection.get('threshold', 0.55)),
                    'prefix_padding_ms': int(turn_detection.get('prefix_padding_ms', 180)),
                    'silence_duration_ms': silence_duration_ms,
                    'create_response': True,
                    'interrupt_response': True,
                }

            input_audio_transcription = session_data.get(
                'input_audio_transcription')
            if isinstance(input_audio_transcription, dict):
                session_update['input_audio_transcription'] = {
                    'model': input_audio_transcription.get('model', 'whisper-1')
                }

            if session_update:
                logger.debug(
                    'Applying VoiceLive session update: %s', session_update)
                try:
                    await connection.session.update(session=session_update)
                except Exception as e:
                    logger.error(f"Failed to apply session update: {e}")

        elif msg_type == "input_audio_buffer.append":
            # Handle incoming audio data from client browser
            audio_data = message.get('audio', '')
            if audio_data:
                try:
                    # Forward audio from client to VoiceLive through audio bridge
                    await audio_bridge.forward_client_audio_to_voicelive(audio_data)
                except Exception as e:
                    logger.error(f"Error forwarding audio: {e}")

        elif msg_type == "input_audio_buffer.commit":
            # Commit the audio buffer (trigger processing)
            try:
                await audio_bridge.wait_for_audio_flush()
                await connection.input_audio_buffer.commit()
            except Exception as e:
                logger.error(f"Error committing audio buffer: {e}")

        elif msg_type == "conversation.item.create":
            # Handle text messages and other conversation items by converting the
            # legacy client format into the VoiceLive wire format.
            logger.debug(f"Creating conversation item: {message}")
            try:
                # Extract the item data from the client message.
                item_data = message.get('item', {})
                item_type = item_data.get('type', 'message')
                item_role = item_data.get('role', 'user')
                item_content = item_data.get('content', [])

                # Convert legacy content payloads into VoiceLive content parts.
                if item_type == 'message' and item_role == 'user':
                    from azure.ai.voicelive.models import UserMessageItem, InputTextContentPart

                    # Convert content parts
                    voice_live_content = []
                    for content_part in item_content:
                        if content_part.get('type') == 'input_text':
                            text_content = InputTextContentPart(
                                text=content_part.get('text', ''))
                            voice_live_content.append(text_content)
                        elif content_part.get('type') == 'input_image':
                            # Mirror legacy behavior by forwarding the original image payload.
                            image_part = json.loads(json.dumps(content_part))
                            has_image_data = any(
                                image_part.get(key)
                                for key in ("image_url", "image_base64", "image_data")
                            )

                            if not has_image_data:
                                logger.warning(
                                    "Received input_image content without image data; skipping")
                                continue

                            voice_live_content.append(image_part)
                            logger.info(
                                "Forwarding image content to VoiceLive (detail=%s)",
                                image_part.get("image_url", {}).get("detail")
                                if isinstance(image_part.get("image_url"), dict)
                                else None,
                            )
                        else:
                            logger.debug(
                                f"Skipping unsupported content type: {content_part.get('type')}")

                    # Ensure we have at least one content part
                    if not voice_live_content:
                        logger.warning(
                            "No processable content found in message, skipping item creation")
                        return

                    # Create VoiceLive UserMessageItem
                    user_message = UserMessageItem(
                        role='user', content=voice_live_content)
                    logger.debug(
                        f"Created UserMessageItem: {user_message.as_dict()}")

                    # Create the conversation item using the VoiceLive format
                    await connection.conversation.item.create(item=user_message)

                else:
                    logger.warning(
                        f"Unsupported item type/role: {item_type}/{item_role}")

            except Exception as e:
                logger.error(f"Error creating conversation item: {e}")
                logger.error(f"Original message: {message}")
                logger.error(f"Item data: {item_data}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

        elif msg_type == "extension.transcript.update":
            transcript = message.get('transcript')
            if isinstance(transcript, list):
                connection_state = self.active_connections.get(connection_id)
                if connection_state is not None:
                    connection_state['conversation_transcript'] = transcript

        elif msg_type == "response.create":
            # Create a new response
            try:
                response_config = message.get('response', {})
                await connection.response.create(**response_config)
            except Exception as e:
                logger.error(f"Error creating response: {e}")

        elif msg_type == "response.cancel":
            # Cancel current response
            try:
                await connection.response.cancel()
            except Exception as e:
                logger.error(f"Error canceling response: {e}")

        elif msg_type == "input_audio_buffer.clear":
            # Clear the audio buffer
            try:
                await audio_bridge.wait_for_audio_flush()
                await connection.input_audio_buffer.clear()
            except Exception as e:
                logger.error(f"Error clearing audio buffer: {e}")

        else:
            logger.debug(f"Unhandled client message type: {msg_type}")

    async def handle_websocket(self, request: web.Request):
        """Main WebSocket handler for VoiceLive integration."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        connection_id = f"conn_{id(ws)}"
        logger.info(f"New VoiceLive WebSocket connection: {connection_id}")

        try:
            # Connect to VoiceLive WebSocket API
            async with connect(
                endpoint=self.endpoint,
                credential=self.credential,
                model=self.model,
                connection_options={
                    "max_msg_size": 10 * 1024 * 1024,
                    "heartbeat": 20,
                    "timeout": 60,
                },
            ) as voice_live_connection:

                # Initialize audio bridge
                audio_bridge = WebSocketAudioBridge(voice_live_connection, ws)

                # Store connection info
                self.active_connections[connection_id] = {
                    'ws': ws,
                    'voice_live_connection': voice_live_connection,
                    'audio_bridge': audio_bridge,
                    'opening_greeting_sent': False,
                    'conversation_transcript': [],
                }

                # Setup session
                await self._setup_session(voice_live_connection)
                await audio_bridge.start()

                logger.info(f"VoiceLive connection ready: {connection_id}")

                async def handle_client_messages():
                    """Handle messages from client WebSocket."""
                    try:
                        async for msg in ws:
                            if msg.type == web.WSMsgType.TEXT:
                                try:
                                    message = json.loads(msg.data)
                                    await self._handle_client_message(
                                        message,
                                        voice_live_connection,
                                        audio_bridge,
                                        connection_id,
                                    )
                                except json.JSONDecodeError as e:
                                    logger.error(
                                        f"Invalid JSON from client: {e}")
                            elif msg.type == web.WSMsgType.ERROR:
                                logger.error(
                                    f'WebSocket error: {ws.exception()}')
                                break
                            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED):
                                break
                    finally:
                        logger.info(
                            "Client WebSocket loop ended for VoiceLive connection: %s",
                            connection_id,
                        )

                async def handle_voice_live_events():
                    """Handle events from VoiceLive connection."""
                    try:
                        async for event in voice_live_connection:
                            await self._handle_voice_live_event(
                                event, ws, audio_bridge, voice_live_connection, connection_id)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error(f"Error processing VoiceLive events: {e}")

                client_task = asyncio.create_task(handle_client_messages())
                voice_task = asyncio.create_task(handle_voice_live_events())

                done, pending = await asyncio.wait(
                    {client_task, voice_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if client_task in done:
                    logger.info(
                        "Client WebSocket task completed first for VoiceLive connection: %s",
                        connection_id,
                    )
                if voice_task in done:
                    logger.warning(
                        "VoiceLive event task completed first for connection: %s",
                        connection_id,
                    )

                for task in done:
                    exc = task.exception()
                    if exc and not isinstance(exc, asyncio.CancelledError):
                        logger.error("VoiceLive session task failed: %s", exc)

                audio_bridge.is_active = False
                await voice_live_connection.close(code=1000, reason="Client session ended")

                for task in pending:
                    task.cancel()

                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

                if not ws.closed:
                    await ws.close()

        except Exception as e:
            logger.error(f"VoiceLive connection error: {e}")
            if not ws.closed:
                await ws.send_json({
                    "type": "error",
                    "error": {
                        "type": "connection_error",
                        "code": "voice_live_connection_failed",
                        "message": str(e)
                    }
                })

        finally:
            # Cleanup
            if connection_id in self.active_connections:
                audio_bridge = self.active_connections[connection_id]['audio_bridge']
                await audio_bridge.cleanup()
                del self.active_connections[connection_id]

            logger.info(
                f"VoiceLive WebSocket connection closed: {connection_id}")

        return ws


class VoiceLiveMiddleTier:
    """
    Voice Live middle tier for Azure VoiceLive SDK sessions.
    This class manages VoiceLive connections and tool execution state.
    """

    def __init__(
        self,
        credentials: Union[AzureKeyCredential, TokenCredential],
        endpoint: str,
        model: str,
        voice_choice: Optional[str] = None,
    ):
        self.endpoint = endpoint
        self.model = model
        self.voice_choice = voice_choice or "alloy"
        self.credentials = credentials

        # Server-enforced configuration
        self.system_message: Optional[str] = None
        self.temperature: Optional[float] = None
        self.max_tokens: Optional[int] = None
        self.tools: List[Dict[str, Any]] = []
        self.tool_handlers: Dict[str,
                                 Callable[[Any], Awaitable[ToolResult]]] = {}

        # WebSocket handler
        self._ws_handler: Optional[VoiceLiveWebSocketHandler] = None

    def add_tool(
        self,
        tool_schema: Dict[str, Any],
        handler: Optional[Callable[[Any], Awaitable[ToolResult]]] = None,
    ):
        """Add or update a tool to be available during voice conversations."""
        tool_name = tool_schema.get("name")

        if tool_name:
            for idx, existing_schema in enumerate(self.tools):
                if existing_schema.get("name") == tool_name:
                    self.tools[idx] = tool_schema
                    break
            else:
                self.tools.append(tool_schema)

            if handler is not None:
                self.tool_handlers[tool_name] = handler
        else:
            self.tools.append(tool_schema)

    def _get_ws_handler(self) -> VoiceLiveWebSocketHandler:
        """Get or create the WebSocket handler."""
        if not self._ws_handler:
            self._ws_handler = VoiceLiveWebSocketHandler(
                endpoint=self.endpoint,
                credential=self.credentials,
                model=self.model,
                voice=self.voice_choice,
                instructions=self.system_message or "You are a helpful assistant.",
                tools=self.tools,
                tool_handlers=self.tool_handlers,
            )
        else:
            # Update the handler with current settings
            self._ws_handler.instructions = self.system_message or "You are a helpful assistant."
            self._ws_handler.tools = self.tools
            self._ws_handler.voice = self.voice_choice
            self._ws_handler.tool_handlers = self.tool_handlers

        return self._ws_handler

    def attach_to_app(self, app: web.Application, path: str):
        """Attach the VoiceLive handler to the web application."""
        handler = self._get_ws_handler()
        app.router.add_get(path, handler.handle_websocket)
        logger.info(f"VoiceLive handler attached to {path}")


# Export the main classes and functions
__all__ = [
    "VoiceLiveMiddleTier",
    "VoiceLiveWebSocketHandler",
    "WebSocketAudioBridge",
]
