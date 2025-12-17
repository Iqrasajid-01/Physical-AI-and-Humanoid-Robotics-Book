---
title: Whisper Integration for Voice Commands
sidebar_label: 21 - Whisper Integration for Voice Commands
---

# Whisper Integration for Voice Commands

## Learning Objectives

By the end of this chapter, you will be able to:
- Integrate OpenAI Whisper for speech recognition in robotic systems
- Implement real-time voice command processing pipelines
- Design voice command grammars and recognition strategies
- Optimize Whisper models for robotic applications
- Handle voice command ambiguities and error recovery
- Evaluate voice command recognition performance in robotics

## Introduction

Voice command integration enables robots to understand and respond to natural spoken language, providing a more intuitive and accessible interaction paradigm. OpenAI's Whisper model has revolutionized speech recognition with its robust performance across different accents, languages, and acoustic conditions. This chapter explores the integration of Whisper with robotic systems to enable voice-controlled robot operation.

Whisper's ability to handle diverse audio conditions makes it particularly valuable for robotics applications where acoustic environments can vary significantly. The integration involves real-time audio processing, speech-to-text conversion, natural language understanding, and command execution.

## Core Concepts

### Whisper Model Architecture

Whisper is built on a Transformer architecture with:
- **Encoder**: Processes audio spectrograms
- **Decoder**: Generates text transcriptions
- **Multilingual Capability**: Supports multiple languages
- **Robustness**: Handles various acoustic conditions

### Voice Command Processing Pipeline

The typical pipeline includes:
- **Audio Capture**: Recording voice commands from microphones
- **Preprocessing**: Noise reduction and audio enhancement
- **Speech Recognition**: Converting speech to text with Whisper
- **Natural Language Understanding**: Interpreting command meaning
- **Command Execution**: Translating commands to robot actions

### Real-time Processing Considerations

- **Latency**: Minimizing delay between speech and action
- **Throughput**: Processing audio streams in real-time
- **Resource Usage**: Efficient use of computational resources
- **Reliability**: Consistent performance in varying conditions

### Voice Command Design

- **Grammar Structure**: Defining acceptable command formats
- **Vocabulary Limitations**: Managing recognition accuracy
- **Ambiguity Resolution**: Handling unclear commands
- **Feedback Mechanisms**: Confirming command understanding

## Architecture Diagram

![Flow Diagram](/img/ch21-ad.svg)

<!-- ```mermaid
graph TB
    subgraph "Audio Input"
        A[Microphone Array]
        B[Audio Preprocessing]
        C[Noise Reduction]
    end

    subgraph "Whisper Processing"
        D[Audio Spectrogram]
        E[Whisper Encoder]
        F[Whisper Decoder]
        G[Text Transcription]
    end

    subgraph "Command Processing"
        H[Natural Language Parser]
        I[Command Interpreter]
        J[Action Planner]
    end

    subgraph "Robot Control"
        K[Navigation Commands]
        L[Manipulation Commands]
        M[Interaction Commands]
    end

    subgraph "Feedback System"
        N[Audio Feedback]
        O[Visual Feedback]
        P[Status Indication]
    end

    A -/-> B
    B -/-> C
    C -/-> D
    D -/-> E
    E -/-> F
    F -/-> G
    G -/-> H
    H -/-> I
    I /-/-> J
    J -/-> K
    J -/-> L
    J -/-> M
    K /-/-> N
    L -/-/> N
    M -/-> N
    N -/-> P
    O -/-> P
``` -->

## Flow Diagram

![Flow Diagram](/img/ch21-flow.svg)

<!-- ```mermaid
sequenceDiagram
    participant User as Human User
    participant Mic as Microphone
    participant Whisper as Whisper ASR
    participant NLU as NLU System
    participant Robot as Robot
    participant Feedback as Feedback System

    User->>Mic: Speak command
    Mic->>Whisper: Audio stream
    Whisper->>NLU: Transcribed text
    NLU->>Robot: Structured command
    Robot->>Robot: Execute action
    Robot->>Feedback: Action status
    Feedback->>User: Audio/visual confirmation
    User->>User: Wait for completion
    Robot->>Feedback: Completion status
    Feedback->>User: Task completed confirmation
``` -->

## Code Example: Whisper Integration for Voice Commands

Here's an example implementation of Whisper integration with a robotic system:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import AudioData
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from std_msgs.msg import Header
import numpy as np
import torch
import whisper
import pyaudio
import wave
import threading
import queue
import time
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class VoiceCommand:
    """Represents a recognized voice command"""
    text: str
    confidence: float
    timestamp: float
    interpreted_action: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class WhisperVoiceCommandNode(Node):
    """
    Node that integrates Whisper for voice command recognition
    """
    def __init__(self):
        super().__init__('whisper_voice_command_node')

        # Initialize parameters
        self.declare_parameter('model_size', 'base')  # tiny, base, small, medium, large
        self.declare_parameter('processing_rate', 2.0)  # Process every 2 seconds
        self.declare_parameter('enable_gpu_processing', True)
        self.declare_parameter('audio_chunk_size', 1024)
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('command_timeout', 5.0)
        self.declare_parameter('min_confidence', 0.7)

        # Get parameters
        self.model_size = self.get_parameter('model_size').value
        self.processing_rate = self.get_parameter('processing_rate').value
        self.enable_gpu_processing = self.get_parameter('enable_gpu_processing').value
        self.audio_chunk_size = self.get_parameter('audio_chunk_size').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.command_timeout = self.get_parameter('command_timeout').value
        self.min_confidence = self.get_parameter('min_confidence').value

        # Initialize Whisper model
        self.whisper_model = None
        self._initialize_whisper_model()

        # Audio processing
        self.audio_buffer = queue.Queue(maxsize=100)  # Store audio chunks
        self.recording = False
        self.audio_lock = threading.Lock()

        # Command processing
        self.command_queue = queue.Queue(maxsize=10)
        self.active_command: Optional[VoiceCommand] = None
        self.command_timeout_timer = None

        # Create publishers
        self.command_pub = self.create_publisher(String, '/robot/command', 10)
        self.status_pub = self.create_publisher(String, '/whisper/status', 10)
        self.feedback_pub = self.create_publisher(String, '/whisper/feedback', 10)

        # Create subscribers
        self.audio_sub = self.create_subscription(
            AudioData, '/microphone/audio_raw', self.audio_callback, 10)
        self.wake_word_sub = self.create_subscription(
            String, '/whisper/wake_word', self.wake_word_callback, 10)

        # Create timers
        self.processing_timer = self.create_timer(
            1.0 / self.processing_rate, self.process_audio)
        self.monitoring_timer = self.create_timer(1.0, self.monitor_system)

        # Processing statistics
        self.processed_commands = 0
        self.recognition_errors = 0

        self.get_logger().info(
            f'Whisper Voice Command Node initialized with {self.model_size} model'
        )

    def _initialize_whisper_model(self):
        """
        Initialize the Whisper model
        """
        try:
            # Determine device
            device = 'cuda' if self.enable_gpu_processing and torch.cuda.is_available() else 'cpu'
            self.get_logger().info(f'Loading Whisper {self.model_size} model on {device}')

            # Load model
            self.whisper_model = whisper.load_model(self.model_size).to(device)

            # Test model availability
            dummy_audio = np.zeros(int(self.sample_rate * 1.0))  # 1 second of silence
            result = self.whisper_model.transcribe(dummy_audio)
            self.get_logger().info('Whisper model loaded successfully')

        except Exception as e:
            self.get_logger().error(f'Failed to load Whisper model: {e}')
            self.whisper_model = None

    def audio_callback(self, msg: AudioData):
        """
        Handle incoming audio data
        """
        try:
            # Convert audio data to numpy array
            # Assuming the audio data is in int16 format
            audio_array = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Add to audio buffer
            if not self.audio_buffer.full():
                self.audio_buffer.put({
                    'audio': audio_array,
                    'timestamp': time.time(),
                    'header': msg.header
                })
                self.get_logger().debug(f'Audio chunk added to buffer, size: {self.audio_buffer.qsize()}')
            else:
                # Buffer full, drop oldest
                try:
                    self.audio_buffer.get_nowait()
                    self.audio_buffer.put({
                        'audio': audio_array,
                        'timestamp': time.time(),
                        'header': msg.header
                    })
                except queue.Empty:
                    pass

        except Exception as e:
            self.get_logger().error(f'Error in audio callback: {e}')

    def wake_word_callback(self, msg: String):
        """
        Handle wake word detection
        """
        try:
            wake_word = msg.data.lower()
            self.get_logger().info(f'Wake word detected: {wake_word}')

            # Start recording for command
            self.start_recording_command()

        except Exception as e:
            self.get_logger().error(f'Error in wake word callback: {e}')

    def start_recording_command(self):
        """
        Start recording a voice command
        """
        try:
            self.recording = True
            self.get_logger().info('Started recording voice command')

            # Publish status
            status_msg = String()
            status_msg.data = 'recording_voice_command'
            self.status_pub.publish(status_msg)

            # Set timeout for command
            if self.command_timeout_timer:
                self.command_timeout_timer.cancel()

            self.command_timeout_timer = self.create_timer(
                self.command_timeout, self.command_timeout_callback)

        except Exception as e:
            self.get_logger().error(f'Error starting recording: {e}')

    def command_timeout_callback(self):
        """
        Handle command timeout
        """
        try:
            self.recording = False
            self.get_logger().info('Command recording timeout')

            # Publish timeout status
            status_msg = String()
            status_msg.data = 'command_timeout'
            self.status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error in timeout callback: {e}')

    def process_audio(self):
        """
        Process audio buffer for voice commands
        """
        if not self.whisper_model or not self.recording:
            return

        try:
            # Collect audio from buffer
            audio_chunks = []
            buffer_size = self.audio_buffer.qsize()

            # Get all available audio chunks
            for _ in range(buffer_size):
                try:
                    chunk_data = self.audio_buffer.get_nowait()
                    audio_chunks.append(chunk_data['audio'])
                except queue.Empty:
                    break

            if not audio_chunks:
                return

            # Concatenate audio chunks
            full_audio = np.concatenate(audio_chunks)

            # Check if we have enough audio (at least 1 second)
            min_samples = int(self.sample_rate * 0.5)  # 0.5 seconds minimum
            if len(full_audio) < min_samples:
                return

            self.get_logger().debug(f'Processing {len(full_audio)/self.sample_rate:.2f}s of audio')

            # Transcribe using Whisper
            with self.audio_lock:
                result = self.whisper_model.transcribe(full_audio, fp16=False)

            # Check confidence (Whisper doesn't provide confidence, so we'll use alternative measures)
            transcription = result['text'].strip()

            if transcription and len(transcription) > 3:  # Basic validity check
                confidence = self._estimate_confidence(transcription, full_audio)

                if confidence >= self.min_confidence:
                    # Create voice command
                    voice_command = VoiceCommand(
                        text=transcription,
                        confidence=confidence,
                        timestamp=time.time()
                    )

                    # Interpret the command
                    interpreted_command = self.interpret_command(transcription)
                    voice_command.interpreted_action = interpreted_command

                    # Add to command queue
                    if not self.command_queue.full():
                        self.command_queue.put(voice_command)
                        self.get_logger().info(f'Recognized command: "{transcription}" (confidence: {confidence:.2f})')

                        # Publish feedback
                        feedback_msg = String()
                        feedback_msg.data = f'Command recognized: {transcription}'
                        self.feedback_pub.publish(feedback_msg)

                        # Stop recording after successful recognition
                        self.recording = False
                        if self.command_timeout_timer:
                            self.command_timeout_timer.cancel()

                        # Update statistics
                        self.processed_commands += 1
                    else:
                        self.get_logger().warn('Command queue is full, dropping command')
                else:
                    self.get_logger().debug(f'Command below confidence threshold: {transcription}')
            else:
                self.get_logger().debug('Empty or invalid transcription')

        except Exception as e:
            self.recording = False
            self.get_logger().error(f'Error in audio processing: {e}')
            self.recognition_errors += 1

    def _estimate_confidence(self, text: str, audio: np.ndarray) -> float:
        """
        Estimate confidence of transcription (simplified approach)
        """
        # In a real implementation, you might use more sophisticated methods
        # For now, we'll use simple heuristics

        # Check for common filler words that might indicate low confidence
        filler_words = ['um', 'uh', 'ah', 'you know', 'like']
        filler_count = sum(1 for word in filler_words if word in text.lower())

        # Check audio quality (simplified)
        audio_energy = np.mean(np.abs(audio))
        has_speech = audio_energy > 0.01  # Threshold for speech detection

        # Calculate confidence score
        confidence = 0.9  # Base confidence

        if filler_count > 0:
            confidence -= 0.1 * filler_count

        if not has_speech:
            confidence -= 0.2

        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))

    def interpret_command(self, text: str) -> Optional[str]:
        """
        Interpret natural language command into robot action
        """
        text_lower = text.lower()

        # Define command patterns
        command_patterns = {
            r'move forward|go forward|move ahead|go ahead': 'move_forward',
            r'move backward|go backward|move back|go back': 'move_backward',
            r'turn left|rotate left': 'turn_left',
            r'turn right|rotate right': 'turn_right',
            r'stop|halt|cease': 'stop',
            r'pick up|grasp|take|get|lift': 'grasp_object',
            r'put down|release|drop': 'release_object',
            r'approach|go to|move to': 'approach_object',
            r'help|what can you do': 'show_help',
            r'hello|hi|hey': 'greet',
        }

        for pattern, action in command_patterns.items():
            if re.search(pattern, text_lower):
                return action

        # If no specific pattern matches, return a generic command
        return 'unknown_command'

    def monitor_system(self):
        """
        Monitor system status and performance
        """
        try:
            status_msg = String()
            status_msg.data = f'Whisper Active: {self.whisper_model is not None}, ' \
                             f'Recording: {self.recording}, ' \
                             f'Commands: {self.processed_commands}, ' \
                             f'Errors: {self.recognition_errors}, ' \
                             f'Buffer Size: {self.audio_buffer.qsize()}'

            self.status_pub.publish(status_msg)

            self.get_logger().debug(f'System Status: {status_msg.data}')

        except Exception as e:
            self.get_logger().error(f'Error in monitoring: {e}')

    def destroy_node(self):
        """
        Clean up resources when node is destroyed
        """
        self.get_logger().info('Cleaning up Whisper Voice Command Node')
        super().destroy_node()


class VoiceCommandGrammar:
    """
    Defines grammar and validation for voice commands
    """
    def __init__(self):
        self.valid_actions = [
            'move_forward', 'move_backward', 'turn_left', 'turn_right', 'stop',
            'grasp_object', 'release_object', 'approach_object', 'navigate_to',
            'find_object', 'follow_me', 'wait', 'come_here'
        ]

        self.object_keywords = [
            'cup', 'box', 'ball', 'table', 'chair', 'door', 'person', 'object'
        ]

        self.location_keywords = [
            'kitchen', 'living room', 'bedroom', 'office', 'hallway', 'here', 'there'
        ]

    def validate_command(self, command_text: str) -> Dict[str, Any]:
        """
        Validate and parse a voice command
        """
        result = {
            'valid': False,
            'action': None,
            'object': None,
            'location': None,
            'confidence': 0.0,
            'parsed_text': command_text
        }

        # Parse the command
        words = command_text.lower().split()

        # Extract action
        for word in words:
            if word in ['move', 'go', 'turn', 'rotate', 'stop', 'pick', 'grasp', 'take', 'put', 'release']:
                # Map to valid action
                if word in ['move', 'go']:
                    if any(w in words for w in ['forward', 'ahead']):
                        result['action'] = 'move_forward'
                    elif any(w in words for w in ['backward', 'back']):
                        result['action'] = 'move_backward'
                    elif any(w in words for w in ['left', 'right']):
                        result['action'] = 'turn_left' if 'left' in words else 'turn_right'
                elif word in ['stop', 'halt']:
                    result['action'] = 'stop'
                elif word in ['pick', 'grasp', 'take']:
                    result['action'] = 'grasp_object'
                elif word in ['put', 'release', 'drop']:
                    result['action'] = 'release_object'

        # Extract object
        for word in words:
            if word in self.object_keywords:
                result['object'] = word
                break

        # Extract location
        for word in words:
            if word in self.location_keywords:
                result['location'] = word
                break

        # Validate action
        if result['action'] in self.valid_actions:
            result['valid'] = True
            result['confidence'] = 0.8  # Assume 80% confidence for valid commands

        return result


class AdvancedWhisperProcessor:
    """
    Advanced Whisper processor with additional features
    """
    def __init__(self, model_size: str = 'base', enable_gpu: bool = True):
        self.model_size = model_size
        self.enable_gpu = enable_gpu
        self.model = None
        self.grammar = VoiceCommandGrammar()

        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the Whisper model with additional features
        """
        try:
            device = 'cuda' if self.enable_gpu and torch.cuda.is_available() else 'cpu'
            self.model = whisper.load_model(self.model_size).to(device)

            # Additional features
            self.language_detection_enabled = True
            self.speech_activity_detection = True

            print(f"Advanced Whisper processor initialized with {self.model_size} model on {device}")

        except Exception as e:
            print(f"Failed to initialize Whisper model: {e}")
            self.model = None

    def transcribe_with_context(self, audio: np.ndarray, context: str = "") -> Dict[str, Any]:
        """
        Transcribe audio with contextual information
        """
        if not self.model:
            return {'text': '', 'confidence': 0.0, 'valid': False}

        try:
            # For now, we'll use the basic transcription
            # In a real implementation, you might use the context to improve recognition
            result = self.model.transcribe(audio)

            # Validate with grammar
            validation = self.grammar.validate_command(result['text'])

            return {
                'text': result['text'],
                'confidence': validation['confidence'],
                'valid': validation['valid'],
                'action': validation['action'],
                'object': validation['object'],
                'location': validation['location']
            }

        except Exception as e:
            print(f"Transcription error: {e}")
            return {'text': '', 'confidence': 0.0, 'valid': False}


def create_audio_stream_processor():
    """
    Create a real-time audio stream processor (for standalone use)
    """
    class AudioStreamProcessor:
        def __init__(self, callback: callable):
            self.callback = callback
            self.chunk = 1024
            self.format = pyaudio.paInt16
            self.channels = 1
            self.rate = 16000
            self.p = pyaudio.PyAudio()

            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )

            self.recording = False

        def start_recording(self):
            self.recording = True
            threading.Thread(target=self._record_audio).start()

        def stop_recording(self):
            self.recording = False

        def _record_audio(self):
            while self.recording:
                data = self.stream.read(self.chunk)
                audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self.callback(audio_array)

        def __del__(self):
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

    return AudioStreamProcessor


def main(args=None):
    """
    Main function for Whisper voice command node
    """
    rclpy.init(args=args)

    try:
        whisper_node = WhisperVoiceCommandNode()

        # Example: Simulate some voice commands
        def simulate_voice_command():
            commands = [
                "Move forward",
                "Turn left",
                "Grasp the cup",
                "Go to the kitchen"
            ]
            # In a real system, this would come from actual voice input
            pass

        rclpy.spin(whisper_node)

    except KeyboardInterrupt:
        pass
    finally:
        if 'whisper_node' in locals():
            whisper_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Advanced Whisper Integration Example

Here's an example of advanced Whisper integration with additional features:

```python
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import json
from typing import Callable, Awaitable


class RealTimeWhisperController:
    """
    Real-time Whisper controller for voice commands
    """
    def __init__(self, model_size: str = 'base', sample_rate: int = 16000):
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Audio processing
        self.audio_buffer = []
        self.is_listening = False
        self.listening_thread = None

        # Command processing
        self.command_callbacks = []
        self.status_callbacks = []

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the Whisper model
        """
        try:
            import whisper
            self.model = whisper.load_model(self.model_size)
            print(f"Real-time Whisper controller initialized with {self.model_size} model")
        except ImportError:
            print("Whisper not available, using mock implementation")
            self.model = None

    def start_listening(self):
        """
        Start listening for voice commands
        """
        if not self.is_listening:
            self.is_listening = True
            self.listening_thread = threading.Thread(target=self._listening_loop)
            self.listening_thread.start()

    def stop_listening(self):
        """
        Stop listening for voice commands
        """
        self.is_listening = False
        if self.listening_thread:
            self.listening_thread.join()

    def _listening_loop(self):
        """
        Main listening loop
        """
        while self.is_listening:
            # In a real implementation, this would continuously capture audio
            # For simulation, we'll just sleep and process any buffered audio
            time.sleep(0.1)

            # Process buffered audio if available
            if len(self.audio_buffer) > int(self.sample_rate * 2):  # 2 seconds of audio
                self._process_audio_buffer()

    def _process_audio_buffer(self):
        """
        Process the accumulated audio buffer
        """
        if not self.model or len(self.audio_buffer) == 0:
            return

        try:
            # Convert to numpy array
            audio_data = np.array(self.audio_buffer)

            # Process with Whisper (in a separate thread to avoid blocking)
            future = self.executor.submit(self._transcribe_audio, audio_data)
            result = future.result(timeout=5.0)  # 5 second timeout

            if result and result['text'].strip():
                # Notify command callbacks
                for callback in self.command_callbacks:
                    callback(result)

                # Update status
                status = {
                    'status': 'command_recognized',
                    'command': result['text'],
                    'timestamp': time.time()
                }
                for callback in self.status_callbacks:
                    callback(status)

            # Clear buffer after processing
            self.audio_buffer = []

        except Exception as e:
            print(f"Error processing audio buffer: {e}")
            status = {
                'status': 'processing_error',
                'error': str(e),
                'timestamp': time.time()
            }
            for callback in self.status_callbacks:
                callback(status)

    def _transcribe_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper
        """
        if not self.model:
            return {'text': '', 'confidence': 0.0}

        try:
            result = self.model.transcribe(audio_data)
            return {
                'text': result['text'],
                'confidence': 0.9,  # Whisper doesn't provide confidence, assume high
                'language': result.get('language', 'unknown')
            }
        except Exception as e:
            print(f"Transcription error: {e}")
            return {'text': '', 'confidence': 0.0}

    def add_command_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Add a callback for recognized commands
        """
        self.command_callbacks.append(callback)

    def add_status_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Add a callback for status updates
        """
        self.status_callbacks.append(callback)

    def process_audio_chunk(self, audio_chunk: np.ndarray):
        """
        Process an incoming audio chunk
        """
        # Add to buffer
        self.audio_buffer.extend(audio_chunk.tolist())

        # Limit buffer size to prevent memory issues
        max_buffer_size = int(self.sample_rate * 10)  # 10 seconds max
        if len(self.audio_buffer) > max_buffer_size:
            self.audio_buffer = self.audio_buffer[-max_buffer_size:]


class VoiceCommandManager:
    """
    Manages voice commands and their execution
    """
    def __init__(self):
        self.active_commands = {}
        self.command_history = []
        self.max_history = 50

    def register_command(self, command_id: str, command_data: Dict[str, Any]):
        """
        Register a new voice command
        """
        self.active_commands[command_id] = {
            'data': command_data,
            'timestamp': time.time(),
            'status': 'registered'
        }

        # Add to history
        self.command_history.append({
            'id': command_id,
            'data': command_data,
            'timestamp': time.time()
        })

        # Limit history size
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]

    def execute_command(self, command_id: str) -> bool:
        """
        Execute a registered command
        """
        if command_id not in self.active_commands:
            return False

        command = self.active_commands[command_id]
        command['status'] = 'executing'

        # In a real implementation, this would execute the command
        # For simulation, we'll just update the status
        time.sleep(0.1)  # Simulate execution time

        command['status'] = 'completed'
        command['completion_time'] = time.time()

        return True

    def get_command_status(self, command_id: str) -> str:
        """
        Get the status of a command
        """
        if command_id in self.active_commands:
            return self.active_commands[command_id]['status']
        return 'unknown'

    def get_recent_commands(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent commands from history
        """
        return self.command_history[-limit:]


def create_voice_command_config():
    """
    Create configuration for voice command system
    """
    config = {
        # Whisper parameters
        'model_size': 'base',
        'enable_gpu': True,
        'sample_rate': 16000,
        'audio_chunk_size': 1024,

        # Processing parameters
        'processing_rate': 2.0,  # Process every 2 seconds
        'command_timeout': 5.0,
        'min_confidence': 0.7,

        # Audio parameters
        'noise_threshold': 0.01,
        'silence_duration': 1.0,
        'max_command_duration': 10.0,

        # Recognition parameters
        'language': 'en',
        'suppress_tokens': [-1],
        'temperature': 0.0,

        # Robot integration
        'command_topic': '/robot/command',
        'status_topic': '/whisper/status',
        'feedback_topic': '/whisper/feedback',

        # Debug parameters
        'enable_logging': True,
        'log_level': 'INFO',
        'enable_profiling': False
    }

    return config


def main_advanced():
    """
    Main function for advanced Whisper integration
    """
    print("Advanced Whisper Voice Command System")

    # Create controller
    controller = RealTimeWhisperController(model_size='base')

    # Create command manager
    command_manager = VoiceCommandManager()

    # Add command callback
    def command_callback(result):
        print(f"Recognized command: {result['text']}")

        # Register the command
        command_id = f"cmd_{int(time.time())}"
        command_manager.register_command(command_id, result)

        # Execute the command
        success = command_manager.execute_command(command_id)
        print(f"Command execution {'successful' if success else 'failed'}")

    controller.add_command_callback(command_callback)

    # Add status callback
    def status_callback(status):
        print(f"Status update: {status['status']}")

    controller.add_status_callback(status_callback)

    # Start listening
    controller.start_listening()

    # Simulate some audio processing
    try:
        for i in range(10):
            # Simulate adding audio chunks
            dummy_audio = np.random.normal(0, 0.01, 1600)  # 0.1 seconds of audio
            controller.process_audio_chunk(dummy_audio)
            time.sleep(0.5)

        # Print recent commands
        recent_commands = command_manager.get_recent_commands()
        print(f"\nRecent commands: {len(recent_commands)}")
        for cmd in recent_commands:
            print(f"  - {cmd['data'].get('text', 'unknown')}")

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        controller.stop_listening()


if __name__ == "__main__":
    main_advanced()
```

## Step-by-Step Practical Tutorial

### Implementing Whisper Voice Command Integration

1. **Install Whisper and required dependencies**:
   ```bash
   pip3 install openai-whisper torch torchaudio pyaudio
   # On some systems, you might need additional packages:
   sudo apt update
   sudo apt install portaudio19-dev python3-pyaudio
   ```

2. **Create a Whisper integration package**:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python whisper_integration_examples --dependencies rclpy std_msgs sensor_msgs geometry_msgs sensor_msgs
   ```

3. **Navigate to the package directory**:
   ```bash
   cd whisper_integration_examples
   ```

4. **Create the main module directory**:
   ```bash
   mkdir whisper_integration_examples
   touch whisper_integration_examples/__init__.py
   ```

5. **Create the Whisper integration implementation** (`whisper_integration_examples/whisper_integration.py`):
   ```python
   # Use the Whisper integration code examples above
   ```

6. **Create a configuration file** (`config/whisper_config.yaml`):
   ```yaml
   whisper_voice_command_node:
     ros__parameters:
       # Whisper model parameters
       model_size: "base"  # tiny, base, small, medium, large
       enable_gpu_processing: true
       sample_rate: 16000
       audio_chunk_size: 1024

       # Processing parameters
       processing_rate: 2.0
       command_timeout: 5.0
       min_confidence: 0.7

       # Audio parameters
       noise_threshold: 0.01
       silence_duration: 1.0

       # Topic configuration
       audio_topic: "/microphone/audio_raw"
       command_topic: "/robot/command"
       status_topic: "/whisper/status"
       feedback_topic: "/whisper/feedback"

       # Debug parameters
       enable_logging: true
       log_level: "INFO"
       enable_profiling: false
   ```

7. **Create launch directory**:
   ```bash
   mkdir launch
   ```

8. **Create a launch file** (`launch/whisper_integration_example.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory
   import os


   def generate_launch_description():
       # Declare launch arguments
       use_sim_time = LaunchConfiguration('use_sim_time', default='false')
       enable_gpu = LaunchConfiguration('enable_gpu', default='true')

       # Get package share directory
       pkg_share = get_package_share_directory('whisper_integration_examples')
       config_file = os.path.join(pkg_share, 'config', 'whisper_config.yaml')

       return LaunchDescription([
           # Declare launch arguments
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='false',
               description='Use simulation time if true'),
           DeclareLaunchArgument(
               'enable_gpu',
               default_value='true',
               description='Enable GPU processing'),

           # Whisper voice command node
           Node(
               package='whisper_integration_examples',
               executable='whisper_integration_examples.whisper_integration',
               name='whisper_voice_command_node',
               parameters=[
                   config_file,
                   {'use_sim_time': use_sim_time},
                   {'enable_gpu_processing': enable_gpu}
               ],
               output='screen'
           )
       ])
   ```

9. **Update setup.py**:
   ```python
   from setuptools import setup
   import os
   from glob import glob

   package_name = 'whisper_integration_examples'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
           (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='User',
       maintainer_email='user@example.com',
       description='Whisper integration examples for voice commands',
       license='Apache-2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'whisper_integration_node = whisper_integration_examples.whisper_integration:main',
           ],
       },
   )
   ```

10. **Build the package**:
    ```bash
    cd ~/ros2_ws
    colcon build --packages-select whisper_integration_examples
    ```

11. **Source the workspace**:
    ```bash
    source install/setup.bash
    ```

12. **Run the Whisper integration example**:
    ```bash
    ros2 launch whisper_integration_examples whisper_integration_example.launch.py enable_gpu:=true
    ```

13. **Test with simulated audio input**:
    ```bash
    # In another terminal, you would publish audio data
    # For testing, you can simulate by publishing text commands to check the pipeline
    ros2 topic pub /microphone/audio_raw sensor_msgs/msg/AudioData "data: [72, 101, 108, 108, 111]"  # "Hello" in ASCII
    ```

14. **Monitor the system status**:
    ```bash
    ros2 topic echo /whisper/status
    ros2 topic echo /whisper/feedback
    ```

## Summary

This chapter covered the integration of OpenAI Whisper for voice command recognition in robotic systems. We explored the architecture of Whisper models, implementation of real-time voice command processing pipelines, and techniques for optimizing voice recognition for robotics applications.

Whisper integration enables robots to understand natural spoken language, providing an intuitive interaction paradigm. The examples demonstrated how to build robust voice command systems that can handle various acoustic conditions and provide reliable command recognition for robotic control.

## Mini-Quiz

1. What is the primary purpose of Whisper in robotics applications?
   - A) Image recognition
   - B) Speech recognition and transcription
   - C) Motion planning
   - D) Path finding

2. Which parameter controls the size of the Whisper model?
   - A) model_type
   - B) model_size
   - C) model_version
   - D) model_scale

3. What is an important consideration for real-time voice command processing?
   - A) High storage requirements
   - B) Low latency processing
   - C) Complex visualization
   - D) Multiple displays

4. Which ROS message type is typically used for audio data?
   - A) AudioStream
   - B) SoundData
   - C) AudioData
   - D) VoiceData

5. What should be considered when designing voice command grammars?
   - A) Only complex sentences
   - B) Vocabulary limitations and ambiguity resolution
   - C) Only technical terms
   - D) Only long commands

**Answers**: 1-B, 2-B, 3-B, 4-C, 5-B