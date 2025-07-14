from typing import List, Tuple, Optional

from llguidance import LLInterpreter
from llguidance.hf import from_tokenizer as llg_from_tokenizer
import numpy as np

from exo import DEBUG


class BufferedOutput:
  stop_sequences: List[str]
  max_tokens: int
  eos_token_id: int
  buffer_char_size: int

  _token_count: int = 0
  buffer: List[Tuple[int, str]]
  
  # Keep track of all tokens generated so far for stop sequence detection
  _all_text: str = ""
  
  # Tokens that have been sent to the user
  _sent_token_count: int = 0

  is_finished: bool = False
  finish_reason: Optional[str] = None

  # Grammar for output structural generation
  guidance_interpreter: Optional[LLInterpreter] = None

  def __init__(
    self,
    max_tokens: int,
    eos_token_id: int,
    stop_sequences: List[str],
    tokenizer,
    grammar_definition: Optional[str] = None,
  ):
    self.buffer = []
    self._all_text = ""
    self._sent_token_count = 0
    # Keep a larger buffer to avoid losing context - at least 100 chars or 10x the stop sequence length
    self.buffer_char_size = max(100, 10 * max(len(stop_sequence) for stop_sequence in stop_sequences)) if len(
      stop_sequences) > 0 else 100
    self.max_tokens = max_tokens
    self.eos_token_id = eos_token_id
    self.stop_sequences = stop_sequences
    self.tokenizer = tokenizer

    # If we are generating structured responses initialize the guidance
    if grammar_definition:
      print(f"Initializing guidance with grammar definition {grammar_definition}")
      self.initialize_guidance(grammar_definition)

  def initialize_guidance(self, grammar_definition: str):
    try:
      self.guidance_interpreter = LLInterpreter(
        llg_from_tokenizer(self.tokenizer, n_vocab=self.tokenizer.vocab_size),
        grammar_definition,
        # These can't be enabled with how we are currently constructing the tokenizer
        enable_ff_tokens=False,
        enable_backtrack=False,
        log_level=2
      )

      self.guidance_interpreter.start_without_prompt()
    except Exception as e:
      if DEBUG >= 2: print(f"Failed to initialize guidance interpreter for grammar definition {grammar_definition}: {e}")
      raise Exception(f"Failed to initialize guidance interpreter: {e}")

  def append(self, token: int):
    # Validate token against guidance interpreter if we are doing guided generation
    if self.guidance_interpreter:
      valid = self.guidance_interpreter.commit_token(token)
      if not valid:
        raise ValueError(f"Schema violation at token {token} ('{self.tokenizer.decode([token])}')")

    # Store the complete text before adding the new token
    old_text = self._all_text
    
    decoded_token = self.tokenizer.decode([token])
    self.buffer.append((token, decoded_token))
    self._all_text += decoded_token
    self._token_count += 1

    if token == self.eos_token_id:
      self.is_finished = True
      self.finish_reason = "stop"
    elif self._token_count >= self.max_tokens:
      self.is_finished = True
      self.finish_reason = "length"
    elif self.guidance_interpreter and self.guidance_interpreter.has_pending_stop():
      self.is_finished = True

      grammar_stop_reason = self.guidance_interpreter.stop_reason()
      if grammar_stop_reason == "EndOfSentence" or grammar_stop_reason == "NoExtension":
        self.finish_reason = "stop"
      elif grammar_stop_reason == "MaxTokensTotal" or grammar_stop_reason == "MaxTokensParser":
        self.finish_reason = "length"
      else:
        self.finish_reason = grammar_stop_reason
    elif len(self.stop_sequences) > 0:
      # Check if any stop sequence appeared after adding the new token
      self.check_new_stop_sequences(old_text)

  def assembled_text(self) -> str:
    return "".join([text for _, text in self.buffer])

  def check_new_stop_sequences(self, old_text: str):
    """Check if a stop sequence appeared after adding the latest token."""
    new_text = self._all_text
    
    for stop_sequence in self.stop_sequences:
      # Check all positions where stop sequence appears in new text
      start = 0
      while True:
        pos = new_text.find(stop_sequence, start)
        if pos == -1:
          break
        
        # Check if this occurrence ends after the old text length
        # This means it was completed by the new token
        stop_end = pos + len(stop_sequence)
        if stop_end > len(old_text):
          if DEBUG >= 2: print(f"Stop sequence '{stop_sequence}' detected at position {pos}")
          
          # Truncate the buffer to remove everything from the stop sequence onwards
          char_count = 0
          tokens_to_keep = 0
          
          for i, (token_id, token_text) in enumerate(self.buffer):
            if char_count >= pos:
              break
            char_count += len(token_text)
            tokens_to_keep = i + 1
          
          # If the stop sequence starts in the middle of a token, we need to handle that
          if char_count > pos and tokens_to_keep > 0:
            # The stop sequence starts within the last kept token
            # For clean output, remove the entire token that contains the start of the stop sequence
            # (e.g., remove "?<" entirely when stop sequence is "<|im_end|>")
            self.buffer = self.buffer[:tokens_to_keep-1]
          else:
            # Stop sequence starts at a token boundary
            self.buffer = self.buffer[:tokens_to_keep]
          
          # Update _all_text to match the actual buffer content
          self._all_text = "".join([text for _, text in self.buffer])
          
          # Adjust _sent_token_count if we've truncated tokens that were already marked as sent
          if self._sent_token_count > len(self.buffer):
            self._sent_token_count = len(self.buffer)
          
          self.is_finished = True
          self.finish_reason = "stop"
          return
        
        start = pos + 1  # Continue searching for more occurrences


  def token_count(self) -> int:
    return self._token_count

  def next_tokens(self) -> List[int]:
    if self.is_finished:
      # Return all remaining tokens if finished (that haven't been sent yet)
      tokens = [token for token, _ in self.buffer[self._sent_token_count:]]
      self._sent_token_count = len(self.buffer)
      return tokens
    
    # Calculate how many tokens we can safely send
    # We need to hold back tokens that might be part of a stop sequence
    safe_to_send = self._calculate_safe_tokens()

    if safe_to_send > self._sent_token_count:
      # Send the tokens that are safe
      tokens = [token for token, _ in self.buffer[self._sent_token_count:safe_to_send]]
      old_sent_count = self._sent_token_count
      self._sent_token_count = safe_to_send

      # Pop old tokens from buffer to keep memory usage reasonable
      # Only pop tokens that have already been sent
      while len(self.buffer) > self.buffer_char_size and old_sent_count > 0:
        self.buffer.pop(0)
        self._sent_token_count -= 1
        old_sent_count -= 1
      
      return tokens
    
    return []
  
  def _calculate_safe_tokens(self) -> int:
    """Calculate how many tokens can be safely sent without risk of being part of a stop sequence."""
    if not self.stop_sequences or len(self.buffer) == 0:
      return len(self.buffer)
    
    # Get the assembled text from tokens we haven't sent yet
    unsent_text = "".join([text for _, text in self.buffer[self._sent_token_count:]])

    # Check if the end of unsent text could be the start of any stop sequence
    for stop_seq in self.stop_sequences:
      # Check all possible prefix lengths
      for prefix_len in range(1, min(len(stop_seq), len(unsent_text)) + 1):
        if unsent_text.endswith(stop_seq[:prefix_len]):
          # We found a potential prefix, need to hold back tokens
          # Find the safe cutoff point before this potential prefix
          safe_text_len = len(unsent_text) - prefix_len
          if safe_text_len <= 0:
            return self._sent_token_count  # Can't send any more tokens
          
          # Find which token corresponds to this safe length
          char_count = 0
          for i in range(self._sent_token_count, len(self.buffer)):
            _, text = self.buffer[i]
            if char_count + len(text) > safe_text_len:
              return i
            char_count += len(text)
    
    # No potential stop sequences found, safe to send all tokens
    return len(self.buffer)

  def get_token_mask(self) -> Optional[np.ndarray]:
    if self.guidance_interpreter:
      mask, _ = self.guidance_interpreter.compute_mask()
      if mask is not None:
        return np.array(list(mask), dtype="int32")

    return None
