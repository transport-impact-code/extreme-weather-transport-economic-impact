"""Configuration applies before any input/output operations."""
import argparse
import importlib
import json
from pathlib import Path

STAGES = {'label': 'label_data', 'samples': 'build_samples', 'train': 'train', 'analyze': 'plot_all_multi'}

def main(argv=None):
    parser = argparse.ArgumentParser(description='Original 24-to-12 traffic prediction workflow')
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--config', type=Path, help='JSON sections: label/samples/train/analyze; paths resolve from current directory')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=JSON', help='Override a stage constant; quote string values as JSON')
    parser.add_argument('--show-config', action='store_true', help='Print overrides without importing scientific dependencies')
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig')).get(args.stage, {}) if args.config else {}
    for setting in args.set:
        key, sep, value = setting.partition('=')
        if not sep:
            parser.error('--set requires KEY=JSON')
        try:
            cfg[key] = json.loads(value)
        except json.JSONDecodeError as exc:
            parser.error(str(exc))
    if args.show_config:
        print(json.dumps(cfg, indent=2, ensure_ascii=False)); return
    module = importlib.import_module('.' + STAGES[args.stage], __package__)
    for key, value in cfg.items():
        if not key.isupper() or not hasattr(module, key):
            parser.error('Unknown or unsupported setting: ' + key)
        setattr(module, key, value)
    if args.stage == 'train':
        names = {'CKPT_PATH':'stgnn_weather_multihazard_event_v1_24to12.pth', 'NORM_PATH':'norm_stats_multihazard_event_v1.json', 'LOSS_PATH':'loss_curve_multihazard_event_v1.npz', 'LOSS_PNG':'loss_curve_multihazard_event_v1.png'}
        for key, name in names.items():
            if key not in cfg:
                setattr(module, key, str(Path(module.SAVE_DIR) / name))
        for key in names:
            Path(getattr(module, key)).parent.mkdir(parents=True, exist_ok=True)
    if args.stage == 'analyze':
        if 'MONTH_NPZ' not in cfg:
            module.MONTH_NPZ = str(Path('artifacts/samples') / (module.MONTH + '_samples.npz'))
        if module.MONTH not in module.EVENT_WINDOWS:
            parser.error('MONTH has no original analysis event window')
        module.EVENT_START, module.EVENT_END = map(module.pd.Timestamp, module.EVENT_WINDOWS[module.MONTH])
    output_key = {'label':'OUTPUT_DIR','samples':'SAVE_DIR','train':'SAVE_DIR','analyze':'OUT_DIR'}[args.stage]
    Path(getattr(module, output_key)).mkdir(parents=True, exist_ok=True)
    module.main()

if __name__ == '__main__':
    main()
