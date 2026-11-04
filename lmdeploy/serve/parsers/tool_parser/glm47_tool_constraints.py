# Copyright (c) OpenMMLab. All rights reserved.
"""Token-aware argument-name constraints for GLM automatic tool calls.

Constrain declared top-level names for plain object schemas. This is not full
JSON Schema validation: required names form an unordered prefix before optional
arguments and cannot repeat. Values and optional-name duplicates retain their
existing behavior. Explicitly open or composed schemas stay open. Auto choice
still permits reasoning, ordinary text, zero calls and multiple calls.

Dispatch on control IDs, not their spellings: an ordinary-token literal such
as ``</arg_value>`` in a shell command must remain ordinary argument text.
"""
from __future__ import annotations


def _closed_properties(schema):
    if not isinstance(schema, dict):
        return None
    properties = schema.get('properties')
    if (not isinstance(properties, dict) or not properties
            or schema.get('additionalProperties', False) is not False
            or any(key in schema for key in ('$ref', 'allOf', 'anyOf', 'oneOf', 'patternProperties'))):
        return None
    return tuple(properties)


def _required_argument_format(keys, required, ids):
    import json

    from xgrammar.structural_tag import GrammarFormat

    required = tuple(dict.fromkeys(required))
    if len(required) > 8:
        raise ValueError('GLM auto required-name constraints support at most 8 required fields')
    if any(name not in keys for name in required):
        raise ValueError('Required GLM argument names must be declared properties')
    bit = {name: 1 << index for index, name in enumerate(required)}
    lines = [f'root ::= state-{(1 << len(required)) - 1}', r'ws ::= [ \t-\r]*']
    excluded = ', '.join(str(value) for value in ids.values())
    lines.append(f'value ::= ExcludeToken({excluded})*')
    for index, name in enumerate(keys):
        lines.append(
            f'arg-{index} ::= Token({ids["<arg_key>"]}) ws {json.dumps(name, ensure_ascii=False)} '
            f'ws Token({ids["</arg_key>"]}) ws Token({ids["<arg_value>"]}) '
            f'value Token({ids["</arg_value>"]}) ws')
    # A state records which required names have not appeared yet. Only state 0
    # is nullable. While names are missing, every complete argument clears one
    # missing bit; optional/repeated names cannot postpone required arguments.
    for missing in range(1 << len(required)):
        choices = [f'arg-{index} state-{missing & ~bit.get(name, 0)}'
                   for index, name in enumerate(keys)
                   if (missing & bit.get(name, 0)) or (missing == 0 and name not in bit)]
        if missing == 0:
            choices.append('""')
        lines.append(f'state-{missing} ::= ' + ' | '.join(choices))
    return GrammarFormat(grammar='\n'.join(lines))


def build_auto_tool_format(tools, tokenizer):
    if tokenizer is None or not tools:
        return None
    functions = [tool.function for tool in tools]
    properties = [_closed_properties(function.parameters) for function in functions]
    if not any(keys is not None for keys in properties):
        return None

    from xgrammar.structural_tag import (AnyTokensFormat, ConstStringFormat, OrFormat,
                                        RegexFormat, SequenceFormat, StarFormat,
                                        StructuralTag, TokenDispatchFormat, TokenFormat)

    spellings = ('<tool_call>', '</tool_call>', '<arg_key>', '</arg_key>',
                 '<arg_value>', '</arg_value>')
    vocab = tokenizer.get_added_vocab()
    if not all(spelling in vocab for spelling in spellings):
        raise ValueError('GLM auto tool constraints require dedicated control token IDs')
    ids = {spelling: vocab[spelling] for spelling in spellings}
    boundaries = list(ids.values())

    def token(spelling):
        return TokenFormat(token=ids[spelling])

    def either(elements):
        return elements[0] if len(elements) == 1 else OrFormat(elements=elements)

    bodies = []
    for function, keys in zip(functions, properties):
        key = (either([ConstStringFormat(value=name) for name in keys])
               if keys is not None else AnyTokensFormat(exclude_tokens=boundaries))
        pair = SequenceFormat(elements=[
            token('<arg_key>'), RegexFormat(pattern=r'\s*'), key,
            RegexFormat(pattern=r'\s*'), token('</arg_key>'),
            RegexFormat(pattern=r'\s*'), token('<arg_value>'),
            AnyTokensFormat(exclude_tokens=boundaries), token('</arg_value>'),
            RegexFormat(pattern=r'\s*'),
        ])
        required = function.parameters.get('required', []) if keys is not None else []
        arguments = (_required_argument_format(keys, required, ids) if required
                     else StarFormat(content=pair))
        bodies.append(SequenceFormat(elements=[
            RegexFormat(pattern=r'\s*'), ConstStringFormat(value=function.name),
            RegexFormat(pattern=r'\s*'), arguments, token('</tool_call>'),
        ]))
    return StructuralTag(format=TokenDispatchFormat(
        rules=[(ids['<tool_call>'], either(bodies))], loop=True,
    )).model_dump(mode='json')
