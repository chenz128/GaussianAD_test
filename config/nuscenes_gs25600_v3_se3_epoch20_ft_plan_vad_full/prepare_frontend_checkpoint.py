"""Extract a weights-only epoch-20 frontend checkpoint for Stage-2 VAD."""

import argparse
from collections import OrderedDict
import os
import os.path as osp

import torch


FORBIDDEN_PREFIXES = (
    'planner_head.',
    'head.planner_offset_head.',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    if not osp.isfile(args.input):
        raise FileNotFoundError(f'source checkpoint not found: {args.input}')
    checkpoint = torch.load(args.input, map_location='cpu')
    source = checkpoint.get('state_dict', checkpoint)
    # Keep an OrderedDict and its private module-version metadata.  SpConv uses
    # this metadata during load to decide whether a kernel-layout conversion is
    # required; rebuilding a plain dict would make valid weights look transposed.
    frontend = OrderedDict(
        (key, value)
        for key, value in source.items()
        if not key.startswith(FORBIDDEN_PREFIXES)
    )
    source_metadata = getattr(source, '_metadata', None)
    if source_metadata is not None:
        frontend._metadata = OrderedDict(
            (key, value)
            for key, value in source_metadata.items()
            if not (
                key == 'planner_head'
                or key.startswith('planner_head.')
                or key == 'head.planner_offset_head'
                or key.startswith('head.planner_offset_head.')
            )
        )

    removed = len(source) - len(frontend)
    if len(source) != 1858 or len(frontend) != 1726 or removed != 132:
        raise RuntimeError(
            'unexpected epoch-20 checkpoint layout: '
            f'source={len(source)}, frontend={len(frontend)}, removed={removed}')
    leaked = [
        key for key in frontend if key.startswith(FORBIDDEN_PREFIXES)
    ]
    if leaked:
        raise RuntimeError(f'old Planner state leaked into frontend: {leaked}')

    output_dir = osp.dirname(osp.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    temporary = f'{args.output}.tmp.{os.getpid()}'
    torch.save(
        {
            'state_dict': frontend,
            'meta': {
                'source': osp.abspath(args.input),
                'source_epoch': checkpoint.get('epoch'),
                'source_global_iter': checkpoint.get('global_iter'),
                'removed_prefixes': FORBIDDEN_PREFIXES,
            },
        },
        temporary,
    )
    os.replace(temporary, args.output)
    print(
        f'created {args.output}: kept={len(frontend)}, removed={removed}, '
        f'source_epoch={checkpoint.get("epoch")}')


if __name__ == '__main__':
    main()
