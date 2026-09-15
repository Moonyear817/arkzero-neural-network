"""Print/save actual module and layer parameter counts for both integrated models."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from network import PolicyValueNetwork
    from network.squad_selector import SquadSelector
    from network.future_event_encoder import EVENT_FEATURE_NAMES, PROGRESSION_FEATURE_NAMES

    result = {}
    lines = ['# 当前双网络逐层参数报告', '', '由实际模型计算；无参数的激活、掩码、池化及拼接均为 0。', '']
    for name, model in (('battle', PolicyValueNetwork()), ('squad', SquadSelector())):
        modules = {key: sum(p.numel() for p in child.parameters()) for key, child in model.named_children()}
        layers = {key: dict(parameters=sum(p.numel() for p in module.parameters(recurse=False)),
                            tensors={k: dict(shape=list(p.shape), parameters=p.numel()) for k,p in module.named_parameters(recurse=False)})
                  for key,module in model.named_modules() if tuple(module.parameters(recurse=False))}
        total = sum(p.numel() for p in model.parameters())
        assert sum(modules.values()) == sum(row['parameters'] for row in layers.values()) == total
        result[name] = dict(total=total, modules=modules, layers=layers)
        lines += [f'## {name}：{total:,}', '', '| 模块 | 参数量 |', '|---|---:|']
        lines += [f'| {key} | {count:,} |' for key,count in modules.items()]
        lines += ['', '| 层 | 参数量 | 权重 / 偏置形状 |', '|---|---:|---|']
        lines += [f'| {key} | {row["parameters"]:,} | '+ '; '.join(f'{k} {v["shape"]}' for k,v in row['tensors'].items())+' |' for key,row in layers.items()]
        lines += ['']
        print(name, f'Total parameters={total:,}', json.dumps(modules, ensure_ascii=False))
    result['total'] = result['battle']['total'] + result['squad']['total']
    result['event_features'] = list(EVENT_FEATURE_NAMES)
    result['progression_features'] = list(PROGRESSION_FEATURE_NAMES)
    lines += [f'**总参数量：{result["total"]:,}**', '']
    output = ROOT / 'outputs/event_architecture'
    output.mkdir(exist_ok=True, parents=True)
    (output/'parameter_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (output/'PARAMETERS.md').write_text('\n'.join(lines))
    print('Combined total parameters:', f'{result["total"]:,}')


if __name__ == '__main__':
    main()
