import ast, json, unittest, importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("runner", ROOT/"run.py")
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
class Smoke(unittest.TestCase):
    def test_compile_all_stages(self):
        config=json.loads((ROOT/"config.example.json").read_text(encoding="utf-8"))
        for item in json.loads((ROOT/"provenance.json").read_text(encoding="utf-8"))["sources"]:
            runner.prepare(item["stage"],config)
    def test_scope_and_windows(self):
        expected={"201903":("2019-03-23 00:00:00","2019-03-27 23:00:00"),"201908":("2019-08-10 00:00:00","2019-08-14 23:00:00"),"201911":("2019-11-19 00:00:00","2019-11-22 23:00:00"),"201912":("2019-12-01 00:00:00","2019-12-08 23:00:00")}
        tree=ast.parse(next((ROOT/"scripts").glob("S2.py")).read_text(encoding="utf-8"))
        values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ("H_NAME","EVENT_WINDOWS_H1","INTENSITY_SCENARIOS")}
        self.assertEqual(values["H_NAME"],"H0")
        self.assertEqual(values["EVENT_WINDOWS_H1"],expected)
        self.assertEqual(values["INTENSITY_SCENARIOS"],["I1","I2","I3"])
    def test_no_algorithm_override(self):
        with self.assertRaises(ValueError): runner.prepare("S4",{"paths_by_stage":{"S4":{"BPR_ALPHA":3}}})
    def test_unknown_stage_fails(self):
        with self.assertRaises(ValueError): runner.prepare("missing",{})
    def test_source_checksums(self):
        import hashlib
        for item in json.loads((ROOT/"provenance.json").read_text(encoding="utf-8"))["sources"]:
            self.assertEqual(hashlib.sha256((ROOT/"scripts"/item["file"]).read_bytes()).hexdigest(),item["packaged_sha256"])
    def test_aes_codes_not_renamed(self):
        tree=ast.parse((ROOT/"scripts/S9_inventory.py").read_text(encoding="utf-8"))
        values={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertTrue({"H17","H18","H19","H21","H22"} <= values)
        self.assertFalse({"H07","H08","H09"} & values)
    def test_headless_s1_s9_compile(self):
        for stage in ["S1","S9"]: runner.prepare(stage,{},True)
    def test_ascii_python_filenames(self):
        for source in ROOT.rglob("*.py"):
            self.assertTrue(source.name.isascii(), source.name)
    def test_stage_registry(self):
        manifest=json.loads((ROOT/"provenance.json").read_text(encoding="utf-8"))
        names={item["stage"]:item["file"] for item in manifest["sources"]}
        for n in range(1,10): self.assertEqual(names[f"S{n}"],f"S{n}.py")
        for stage in ["S9_prepare","S9_inventory","S9_inventory_legacy","S9_inventory_variants"]:
            runner.prepare(stage,{})
    def test_configured_path_compiles(self):
        runner.prepare("S2",{"paths_by_stage":{"S2":{"S1_ROOT":"fixtures/H0"}}})
if __name__=="__main__": unittest.main()
