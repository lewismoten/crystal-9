"""Fixed-permutation INT2 container with corresponding-expert-interleaved FP8 scales."""

from __future__ import annotations

import hashlib
import lzma
import zlib
from pathlib import Path

import torch

from crystal9 import TinyMoEPolicy
from packed_int2 import FORMAT as BASE_FORMAT
from packed_int2 import PackedInt2Policy, _encode, _integrity_digest
from packed_int2_fp8_scale_lzma_binary_bitplane_zlib_codes import _bitplane_decode, _bitplane_encode
from packed_int2_fp8_scale_lzma_binary_permuted_bitplane_zlib_codes import _permute_codes
from tools.int2_fp8_scale_expert_interleave_hierarchy_screen import decode_expert_interleave, encode_expert_interleave

MAGIC_PREFIX = b"C9I2EIS"
FORMAT = "crystal-9-packed-int2-fp8-scale-lzma-expert-interleave-fixed-permutation-stream-binary-bitplane-zlib-codes-v1"
DIGEST_BYTES = 32
FIXED_VOCAB_SIZE = 13
FIXED_CODE_PERMUTATION = (0, 3, 1, 2)


def _digest(header: bytes, compressed_scales: bytes, compressed_codes: bytes) -> bytes:
    return hashlib.sha256(header + compressed_scales + compressed_codes).digest()


def _state_layout() -> tuple[list[tuple[str, int]], dict[str, list[tuple[str, int]]]]:
    state = TinyMoEPolicy(FIXED_VOCAB_SIZE).state_dict()
    non_expert, experts = [], {}
    for name, value in state.items():
        if name.startswith("experts."):
            _, _, suffix = name.split(".", 2)
            experts.setdefault(suffix, []).append((name, value.numel()))
        else:
            non_expert.append((name, value.numel()))
    return non_expert, experts


def _raw_sizes() -> tuple[int, int]:
    values = list(TinyMoEPolicy(FIXED_VOCAB_SIZE).state_dict().values())
    return sum((value.numel() + 3) // 4 for value in values), sum(value.numel() for value in values)


def _interleave_scales(named_scales: dict[str, bytes]) -> bytes:
    non_expert, experts = _state_layout()
    parts = [named_scales[name] for name, _ in non_expert]
    for suffix in sorted(experts):
        parts.append(encode_expert_interleave(tuple(named_scales[name] for name, _ in experts[suffix])))
    return b"".join(parts)


def _restore_scales(payload: bytes) -> bytes:
    non_expert, experts = _state_layout()
    offset, named = 0, {}
    for name, count in non_expert:
        named[name] = payload[offset:offset + count]
        offset += count
    for suffix in sorted(experts):
        entries = experts[suffix]
        count = entries[0][1]
        size = len(entries) * count
        streams = decode_expert_interleave(payload[offset:offset + size], len(entries), count)
        named.update({name: stream for (name, _), stream in zip(entries, streams)})
        offset += size
    if offset != len(payload):
        raise ValueError("interleaved scale payload length validation failed")
    return b"".join(named[name] for name in TinyMoEPolicy(FIXED_VOCAB_SIZE).state_dict())


def export_packed_int2_fp8_scale_lzma_expert_interleave_fixed_permutation_stream_binary_bitplane_zlib_codes(source: TinyMoEPolicy, path: str | Path) -> dict:
    if source.embedding.num_embeddings != FIXED_VOCAB_SIZE:
        raise ValueError("expert-interleave packed INT2 requires the fixed Crystal-9 vocabulary")
    raw_codes, named_scales = bytearray(), {}
    for name, value in source.state_dict().items():
        record = _encode(value, torch.float8_e4m3fn)
        raw_codes.extend(record["packed"].contiguous().numpy().tobytes())
        named_scales[name] = record["scales"].contiguous().view(torch.uint8).numpy().tobytes()
    compressed_codes = zlib.compress(_bitplane_encode(_permute_codes(bytes(raw_codes), FIXED_CODE_PERMUTATION)), level=9)
    compressed_scales = lzma.compress(_interleave_scales(named_scales), preset=9)
    digest = _digest(MAGIC_PREFIX, compressed_scales, compressed_codes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MAGIC_PREFIX + digest + compressed_scales + compressed_codes)
    return {"format": FORMAT, "parameter_values": sum(len(value) for value in named_scales.values()), "scale_type": "float8_e4m3fn-lzma-expert-interleave", "code_permutation": FIXED_CODE_PERMUTATION, "raw_code_bytes": len(raw_codes), "compressed_code_bytes": len(compressed_codes), "raw_scale_bytes": sum(len(value) for value in named_scales.values()), "compressed_scale_bytes": len(compressed_scales), "integrity_sha256": digest.hex()}


def _load(path: str | Path) -> tuple[bytes, bytes]:
    payload = Path(path).read_bytes()
    if len(payload) < len(MAGIC_PREFIX) + DIGEST_BYTES or payload[:len(MAGIC_PREFIX)] != MAGIC_PREFIX:
        raise ValueError("invalid expert-interleave packed artifact payload")
    digest = payload[len(MAGIC_PREFIX):len(MAGIC_PREFIX) + DIGEST_BYTES]
    streams = payload[len(MAGIC_PREFIX) + DIGEST_BYTES:]
    decoder = lzma.LZMADecompressor()
    interleaved_scales = decoder.decompress(streams)
    if not decoder.eof or not decoder.unused_data:
        raise ValueError("invalid expert-interleave scale payload")
    compressed_codes = decoder.unused_data
    compressed_scales = streams[:len(streams) - len(compressed_codes)]
    if digest != _digest(MAGIC_PREFIX, compressed_scales, compressed_codes):
        raise ValueError("packed artifact integrity validation failed")
    raw_code_size, raw_scale_size = _raw_sizes()
    raw_scales = _restore_scales(interleaved_scales)
    if len(raw_scales) != raw_scale_size:
        raise ValueError("compressed scale payload length validation failed")
    raw_codes = _permute_codes(_bitplane_decode(zlib.decompress(compressed_codes), raw_code_size), (0, 2, 3, 1))
    return raw_codes, raw_scales


class PackedInt2Fp8ScaleLzmaExpertInterleaveFixedPermutationStreamBinaryBitplaneZlibCodesPolicy:
    def __init__(self, packed: bytes, raw_scales: bytes) -> None:
        prototype = TinyMoEPolicy(FIXED_VOCAB_SIZE)
        tensors, packed_offset, scale_offset = {}, 0, 0
        for name, value in prototype.state_dict().items():
            packed_end = packed_offset + (value.numel() + 3) // 4
            scale_end = scale_offset + value.numel()
            tensors[name] = {"shape": tuple(value.shape), "count": value.numel(), "packed": torch.frombuffer(bytearray(packed[packed_offset:packed_end]), dtype=torch.uint8).clone(), "scales": torch.frombuffer(bytearray(raw_scales[scale_offset:scale_end]), dtype=torch.uint8).view(torch.float8_e4m3fn).clone()}
            packed_offset, scale_offset = packed_end, scale_end
        architecture = {"vocab_size": FIXED_VOCAB_SIZE, "hidden_size": prototype.embedding.embedding_dim, "experts": len(prototype.experts), "heads": prototype.attention.num_heads, "norm_eps": prototype.norm.eps}
        manifest = {"format": BASE_FORMAT, "layout": "complete-scalar-group-int2-research", "architecture": architecture, "parameter_values": sum(value.numel() for value in prototype.parameters()), "scale_type": "float8_e4m3fn", "tensors": tensors}
        manifest["integrity_sha256"] = _integrity_digest(manifest)
        self._runtime = PackedInt2Policy(manifest)

    @classmethod
    def load(cls, path: str | Path):
        return cls(*_load(path))

    def eval(self):
        return self

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._runtime(token_ids)
