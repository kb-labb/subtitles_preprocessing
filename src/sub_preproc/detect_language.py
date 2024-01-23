import torch
import transformers
from typing import Optional, Collection, List, Dict

WHISPER_LANGUAGES = [
    "en",
    "zh",
    "de",
    "es",
    "ru",
    "ko",
    "fr",
    "ja",
    "pt",
    "tr",
    "pl",
    "ca",
    "nl",
    "ar",
    "sv",
    "it",
    "id",
    "hi",
    "fi",
    "vi",
    "he",
    "uk",
    "el",
    "ms",
    "cs",
    "ro",
    "da",
    "hu",
    "ta",
    "no",
    "th",
    "ur",
    "hr",
    "bg",
    "lt",
    "la",
    "mi",
    "ml",
    "cy",
    "sk",
    "te",
    "fa",
    "lv",
    "bn",
    "sr",
    "az",
    "sl",
    "kn",
    "et",
    "mk",
    "br",
    "eu",
    "is",
    "hy",
    "ne",
    "mn",
    "bs",
    "kk",
    "sq",
    "sw",
    "gl",
    "mr",
    "pa",
    "si",
    "km",
    "sn",
    "yo",
    "so",
    "af",
    "oc",
    "ka",
    "be",
    "tg",
    "sd",
    "gu",
    "am",
    "yi",
    "lo",
    "uz",
    "fo",
    "ht",
    "ps",
    "tk",
    "nn",
    "mt",
    "sa",
    "lb",
    "my",
    "bo",
    "tl",
    "mg",
    "as",
    "tt",
    "haw",
    "ln",
    "ha",
    "ba",
    "jw",
    "su",
]


def detect_language(
    model: transformers.WhisperForConditionalGeneration,
    tokenizer: transformers.WhisperTokenizer,
    input_features,
    possible_languages: Optional[Collection[str]] = None,
) -> List[Dict[str, float]]:
    # hacky, but all language tokens and only language tokens are 6 characters long
    # except for haw
    language_tokens = [
        t for t in tokenizer.additional_special_tokens if len(t) == 6 or len(t) == 7
    ]
    if possible_languages is not None:
        language_tokens = [t for t in language_tokens if t[2:-2] in possible_languages]
        if len(language_tokens) < len(possible_languages):
            raise RuntimeError(
                f"Some languages in {possible_languages} did not have associated language tokens"
            )

    language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens)

    # 50258 is the token for transcribing
    logits = model(
        input_features,
        decoder_input_ids=torch.tensor([[50258] for _ in range(input_features.shape[0])]).to(
            "cuda:0"
        ),
    ).logits
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[language_token_ids] = False
    logits[:, :, mask] = -float("inf")

    output_probs = logits.softmax(dim=-1).cpu()
    return [
        {
            lang.strip("<|>"): output_probs[input_idx, 0, token_id].item()
            for token_id, lang in zip(language_token_ids, language_tokens)
        }
        for input_idx in range(logits.shape[0])
    ]
