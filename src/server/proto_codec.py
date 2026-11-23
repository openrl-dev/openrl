"""Backward-compatible re-export shim over ``server.api.wire``."""

from server.api.wire import (
  PROTO,
  PROTO_CONTENT_TYPE,
  PROTO_RESULT_TYPES,
  ProtoDecodeError,
  decode_forward_backward,
  decode_future_result_json,
  encode_forward_backward_output,
  encode_future_result,
  encode_sample_response,
  forward_backward_output_to_json,
  is_proto_result,
  json_to_forward_backward_proto,
  parse_forward_backward,
  proto_result,
  proto_result_bytes,
  sample_response_to_json,
)

__all__ = [
  "PROTO",
  "PROTO_CONTENT_TYPE",
  "PROTO_RESULT_TYPES",
  "ProtoDecodeError",
  "decode_forward_backward",
  "decode_future_result_json",
  "encode_forward_backward_output",
  "encode_future_result",
  "encode_sample_response",
  "forward_backward_output_to_json",
  "is_proto_result",
  "json_to_forward_backward_proto",
  "parse_forward_backward",
  "proto_result",
  "proto_result_bytes",
  "sample_response_to_json",
]
