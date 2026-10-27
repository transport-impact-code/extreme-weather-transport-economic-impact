"""Explicit stage runner. Overrides only top-level path assignments."""
import argparse, ast, json, os, sys, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def prepare(stage, config, no_plots=False):
    manifest = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    matches = [s for s in manifest["sources"] if s["stage"] == stage]
    if not matches:
        raise ValueError("Source unavailable: " + stage)
    source = ROOT / "scripts" / matches[0]["file"]
    tree = ast.parse(source.read_text(encoding="utf-8"))
    allowed = json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))["paths_by_stage"][stage]
    overrides = config.get("paths_by_stage", {}).get(stage, {})
    for key, value in overrides.items():
        if key not in allowed:
            raise ValueError("Only documented path keys may be overridden: " + key)
        if value is not None and not isinstance(value, (str, list)):
            raise ValueError("Path must be string or list: " + key)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            value = overrides.get(name)
            if value is not None:
                def path_expr(v):
                    return ast.Call(func=ast.Name(id="Path",ctx=ast.Load()), args=[ast.Constant(value=v)], keywords=[])
                node.value = ast.List(elts=[path_expr(v) for v in value],ctx=ast.Load()) if isinstance(value,list) else path_expr(value)
    if no_plots:
        kept=[]
        for node in tree.body:
            if isinstance(node,ast.Import) and any(a.name.startswith("matplotlib") for a in node.names):
                continue
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=="matplotlib" and node.value.func.attr=="use":
                continue
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="PLOT_FIGURES" for t in node.targets):
                node.value=ast.Constant(value=False)
            kept.append(node)
        tree.body=kept
    ast.fix_missing_locations(tree)
    return source, compile(tree, str(source), "exec")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--config", default=str(ROOT / "config.example.json"))
    parser.add_argument("--check", action="store_true", help="Compile only; does not read input datasets")
    parser.add_argument("--no-plots",action="store_true",help="Disable plotting only; calculation parameters are unchanged")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source, code = prepare(args.stage, config, args.no_plots)
    if args.check:
        print("COMPILE ONLY:", source.name)
        return
    os.environ["SCENARIO_DATA_ROOT"] = str(Path(config["data_root"]).resolve())
    sys.path.insert(0,str(ROOT / "scripts"))
    if args.stage == "S1":
        expected = hashlib.sha256((ROOT/"scripts/model.py").read_bytes()).hexdigest()
        explicit = config.get("shared_model_path")
        candidates = [Path(explicit)] if explicit else [ROOT.parent/"traffic_prediction/traffic_prediction/model.py", ROOT.parent/"traffic_prediction/model.py"]
        for candidate in candidates:
            if candidate.exists() and hashlib.sha256(candidate.read_bytes()).hexdigest()==expected:
                sys.path.insert(0,str(candidate.resolve().parent))
                print("Using identical shared traffic_prediction model:",candidate)
                break
        else:
            if explicit:
                raise ValueError("Shared model missing or different from audited model.py")
            print("Using bundled audited model.py")
    exec(code, {"__name__":"__main__", "__file__":str(source)})
if __name__ == "__main__":
    main()
