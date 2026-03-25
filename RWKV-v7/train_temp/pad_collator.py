from typing import Any
import torch
from torch.nn.utils.rnn import pad_sequence


class PadCollator:
    """Pads (and optionally truncates) batch to the longest sequence.

    Applies proper masking and label ignore index.

    Attributes:
        pad_token_id (int): The token ID to use for padding.
        max_context (int): The maximum context length to allow.
        ignore_index (int): The label ignore index.

    """

    def __init__(
        self,
        pad_token_id: int = 0,
        max_context: int | None = None,
    ) -> None:
        """Initialize the collator with a padding token ID.

        Args:
            pad_token_id (int, optional): The token ID used for padding. Defaults to 0.
            max_context (Optional[int], optional): The maximum context length to allow.
                Defaults to None.

        """
        self.pad_token_id = pad_token_id
        self.max_context = max_context
        self.ignore_index = -100

    def _truncate(self, seq: list[int]) -> list[int]:
        if self.max_context is None:
            return seq
        return seq[: self.max_context]

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Pad the batch to the maximum sequence length found in the features.

        Args:
            features (List[Dict[str, Any]]): The list of features to pad.

        Returns:
            Dict[str, torch.Tensor]: The padded features.

        """
        if not features:
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "labels": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
            }

        input_tensors = []
        label_tensors = []
        for f in features:
            inp = self._truncate(f["input_ids"])
            lab = self._truncate(f["labels"])
            input_tensors.append(torch.tensor(inp, dtype=torch.long))
            label_tensors.append(torch.tensor(lab, dtype=torch.long))

        input_ids = pad_sequence(
            input_tensors,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            label_tensors,
            batch_first=True,
            padding_value=self.ignore_index,
        )

        attention_mask = (input_ids != self.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }