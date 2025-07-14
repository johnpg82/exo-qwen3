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
    self.buffer_char_size = max(len(stop_sequence) for stop_sequence in stop_sequences) if len(
      stop_sequences) > 0 else 0
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

    # Store the text before adding the new token
    old_text = self.assembled_text()
    
    decoded_token = self.tokenizer.decode([token])
    self.buffer.append((token, decoded_token))
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
    new_text = self.assembled_text()
    
    for stop_sequence in self.stop_sequences:
      # Check if stop sequence wasn't in old text but is in new text
      if stop_sequence not in old_text and stop_sequence in new_text:
        if DEBUG >= 2: print(f"Stop sequence '{stop_sequence}' newly appeared in text")
        
        # Find where the stop sequence starts
        stop_idx = new_text.index(stop_sequence)
        
        # Truncate the buffer to remove everything from the stop sequence onwards
        char_count = 0
        tokens_to_keep = 0
        
        for i, (token_id, token_text) in enumerate(self.buffer):
          if char_count >= stop_idx:
            break
          char_count += len(token_text)
          tokens_to_keep = i + 1
        
        # If the stop sequence starts in the middle of a token, we need to handle that
        if char_count > stop_idx and tokens_to_keep > 0:
          # The stop sequence starts within the last kept token
          last_token_id, last_token_text = self.buffer[tokens_to_keep - 1]
          overlap = char_count - stop_idx
          truncated_text = last_token_text[:-overlap]
          
          # Update the buffer
          self.buffer = self.buffer[:tokens_to_keep-1]
          if truncated_text:  # Only add if there's text remaining
            self.buffer.append((last_token_id, truncated_text))
        else:
          # Stop sequence starts at a token boundary
          self.buffer = self.buffer[:tokens_to_keep]
        
        self.is_finished = True
        self.finish_reason = "stop"
        return

  # Keep the old method for compatibility but it's not actively used
  def attempt_to_match_stop_sequences(self):
    # This method is kept for backward compatibility
    # The actual stop sequence detection now happens in check_new_stop_sequences
    pass

  def token_count(self) -> int:
    return self._token_count

  def next_tokens(self) -> List[int]:
    if self.is_finished:
      # Return all remaining tokens if finished
      tokens = [token for token, _ in self.buffer]
      self.buffer = []
      return tokens
    elif len(self.assembled_text()) >= self.buffer_char_size:
      token, _ = self.buffer.pop(0)
      return [token]

    # Not enough tokens yet
    return []

  def get_token_mask(self) -> Optional[np.ndarray]:
    if self.guidance_interpreter:
      mask, _ = self.guidance_interpreter.compute_mask()
      if mask is not None:
        return np.array(list(mask), dtype="int32")

    return None
