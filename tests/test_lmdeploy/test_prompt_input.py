# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from lmdeploy.model import MODELS
from lmdeploy.serve.processors import MultimodalProcessor


@pytest.mark.parametrize('role', ['system', 'user', 'assistant', 'tool'])
@pytest.mark.parametrize('texts', [
    ['hello', 'world'],
    ['hello', '', 'world'],
    ['</system-reminder>\n', '<system-reminder>'],
    ['<tool_', 'call>'],
])
def test_glm53_text_blocks_match_native_template(hf_media_template, role, texts):
    template = hf_media_template()
    tokenizer = SimpleNamespace(encode=lambda prompt, **kwargs: list(prompt.encode()))
    processor = MultimodalProcessor(tokenizer, template)
    message = dict(role=role, content=[dict(type='text', text=text) for text in texts])
    if role == 'assistant':
        message['reasoning_content'] = ''
    messages = [dict(role='user', content='hi'), message, dict(role='user', content='next')]
    original = deepcopy(messages)

    result = asyncio.run(processor.get_prompt_input(messages, do_preprocess=True))
    expected = template.messages2prompt(messages)

    assert result['prompt'] == expected
    assert result['input_ids'] == tokenizer.encode(expected)
    assert messages == original


def test_other_hf_models_keep_newline_separator(hf_media_template):
    template = hf_media_template()
    template.is_glm5_next = False
    processor = MultimodalProcessor(SimpleNamespace(encode=lambda *args, **kwargs: []), template)
    messages = [dict(role='user', content=[dict(type='text', text='hello'), dict(type='text', text='world')])]

    result = asyncio.run(processor.get_prompt_input(messages, do_preprocess=True))

    assert result['prompt'] == template.messages2prompt([dict(role='user', content='hello\nworld')])


def test_text_separator_uses_selected_adapter_template(hf_media_template, monkeypatch):
    template = hf_media_template()
    adapter_template = hf_media_template()
    adapter_template.is_glm5_next = False
    monkeypatch.setitem(MODELS.module_dict, 'test-text-adapter', lambda: adapter_template)
    processor = MultimodalProcessor(SimpleNamespace(encode=lambda *args, **kwargs: []), template)
    messages = [dict(role='user', content=[dict(type='text', text='hello'), dict(type='text', text='world')])]

    result = asyncio.run(processor.get_prompt_input(
        messages, do_preprocess=True, adapter_name='test-text-adapter'))

    assert result['prompt'] == adapter_template.messages2prompt([dict(role='user', content='hello\nworld')])
