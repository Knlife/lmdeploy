# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from lmdeploy.model import MODELS
from lmdeploy.serve.processors import MultimodalProcessor


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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


@pytest.mark.parametrize('backend', ['pytorch', 'turbomind'])
@pytest.mark.parametrize('new_preprocess', [True, False])
@pytest.mark.parametrize('effort,template_effort,expected', [
    ('low', None, 'Low'),
    ('high', None, 'High'),
    ('max', None, 'Max'),
    (None, None, 'Max'),
    (None, 'low', 'Low'),
    ('low', 'high', 'Low'),
    ('high', 'low', 'High'),
])
def test_multimodal_reasoning_effort(hf_media_template, backend, new_preprocess, effort, template_effort, expected):
    template = hf_media_template()
    template_kwargs = {'clear_thinking': False}
    if template_effort is not None:
        template_kwargs['reasoning_effort'] = template_effort
    original_kwargs = dict(template_kwargs)

    def render(messages, chat_template, tools, chat_template_kwargs, **kwargs):
        return chat_template.messages2prompt(messages, tools=tools, **(chat_template_kwargs or {}))

    async def preprocess(messages, input_prompt=None, **kwargs):
        return input_prompt if new_preprocess else messages

    # Exercise the real dispatcher, media loader, template and both vision
    # preprocessing interfaces without loading a model or running inference.
    encoder = SimpleNamespace(
        _uses_new_preprocess=new_preprocess,
        model=SimpleNamespace(get_input_prompt=render, _turbomind_native_vision=True),
        preprocess=preprocess,
        wrap_for_pytorch=AsyncMock(side_effect=render),
        wrap_for_turbomind=AsyncMock(side_effect=render),
    )
    processor = MultimodalProcessor(None, template, vl_encoder=encoder, backend=backend)
    messages = [dict(role='user', content=[
        dict(type='image_data', image_data=dict(data=Image.new('RGB', (2, 2)))),
        dict(type='text', text='describe'),
    ])]

    result = _run_async(processor.get_prompt_input(
        messages, do_preprocess=True, reasoning_effort=effort, chat_template_kwargs=template_kwargs))

    assert f'Reasoning Effort: {expected}' in result
    assert template_kwargs == original_kwargs
