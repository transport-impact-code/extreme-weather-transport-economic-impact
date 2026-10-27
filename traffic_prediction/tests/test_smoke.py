import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_DATA = all(importlib.util.find_spec(x) for x in ('numpy','pandas'))
HAS_MODEL = all(importlib.util.find_spec(x) for x in ('numpy','pandas','torch','matplotlib'))

class StaticTests(unittest.TestCase):
    def test_all_sources_compile_and_no_absolute_paths(self):
        for path in (ROOT/'traffic_prediction').glob('*.py'):
            text = path.read_text(encoding='utf-8-sig')
            compile(text, str(path), 'exec')
            self.assertNotIn('E:\\', text)
            self.assertNotIn('/mnt/data', text)

    def test_original_ast_fingerprints(self):
        records = json.loads((ROOT/'docs/source_equivalence.json').read_text(encoding='utf-8'))
        for row in records:
            tree = ast.parse((ROOT/'traffic_prediction'/row['module']).read_text(encoding='utf-8-sig'))
            name = 'main' if row['definition']=='main_body' else row['definition']
            node = next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
            if row['definition']=='main_body':
                node = ast.Module(body=node.body,type_ignores=[])
            self.assertEqual(hashlib.sha256(ast.dump(node).replace(", type_params=[]", "").encode()).hexdigest(), row['sha256'])

    def test_cli_config_without_dependencies(self):
        result = subprocess.run([sys.executable,'-m','traffic_prediction','train','--config','configs/default.json','--set','EPOCHS=1','--show-config'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['EPOCHS'],1)

    def test_event_window_difference_preserved(self):
        values=[]
        for name in ('label_data.py','plot_all_multi.py'):
            tree=ast.parse((ROOT/'traffic_prediction'/name).read_text(encoding='utf-8-sig'))
            node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EVENT_WINDOWS' for t in n.targets))
            values.append(ast.literal_eval(node.value))
        self.assertEqual(set(values[0]),set(values[1]))
        for month in values[0]:
            self.assertNotEqual(values[0][month][0],values[1][month])

@unittest.skipUnless(HAS_DATA,'numpy/pandas unavailable')
class DataSmokeTests(unittest.TestCase):
    def test_label_and_samples_synthetic(self):
        import numpy as np
        import pandas as pd
        from traffic_prediction import label_data as label, build_samples as samples
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            base=Path(d); inputs=base/'input'; inputs.mkdir()
            nodes=['00000002','00000001']
            nodefile=base/'nodes.csv'; pd.DataFrame({'SITEREF':nodes}).to_csv(nodefile,index=False)
            rows=[]
            times=pd.date_range('2019-03-23',periods=48,freq='h')
            for i,time in enumerate(times):
                for j,node in enumerate(nodes):
                    rows.append(dict(SITEREF=node,DATETIME=str(time),FLOW=i*10+j,TEMP=10+i/10,PRCP=i/100,WDSP=2,RH=70,MXSPD=3))
            pd.DataFrame(rows).to_csv(inputs/'201903.csv',index=False)
            cfg={'label':{'INPUT_DIR':str(inputs),'OUTPUT_DIR':str(base/'labeled')},'samples':{'DATA_DIR':str(base/'labeled'),'NODE_FILE':str(nodefile),'SAVE_DIR':str(base/'samples')}}
            config=base/'config.json'; config.write_text(json.dumps(cfg))
            for stage in ('label','samples'):
                result=subprocess.run([sys.executable,'-m','traffic_prediction',stage,'--config',str(config)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',env={**__import__('os').environ,'PYTHONIOENCODING':'utf-8'})
                self.assertEqual(result.returncode,0,result.stderr)
            with np.load(base/'samples/201903_samples.npz') as a:
                self.assertEqual(a['X_hist'].shape,(13,24,2,5))
                self.assertEqual(a['X_future'].shape,(13,12,2,4))
                self.assertEqual(a['nodes'].tolist(),nodes)
                np.testing.assert_equal(a['Y'][0,0,:,0],[240,241])
                self.assertTrue(np.all(a['EVENT_FLAG'][0]==1))
                self.assertTrue(np.isfinite(a['X_hist']).all())

@unittest.skipUnless(HAS_MODEL,'requires torch and matplotlib (not installed in validation runtime)')
class ModelSmokeTests(unittest.TestCase):
    def test_counterfactual_forward_backward(self):
        import torch
        from traffic_prediction.train import build_counterfactual_raw, weighted_extreme_mae, directional_counterfactual_loss
        from traffic_prediction.model import STGNNWeatherSeq2Seq
        torch.manual_seed(42)
        raw=torch.randn(1,12,2,4); flag=torch.zeros(1,12,2,1); flag[:,3:6]=1
        norm={'mean':[100,10,0,2,70]}
        snapshot=raw.clone(); future=raw.clone(); future_copy=future.clone()
        cf=build_counterfactual_raw(raw,flag,norm)
        self.assertTrue(torch.equal(raw,snapshot))
        self.assertTrue(torch.equal(cf[:,0],raw[:,0]))
        self.assertTrue(torch.equal(cf[:,3,0],torch.tensor([[10.,0.,2.,70.]])))
        model=STGNNWeatherSeq2Seq(2)
        hist=torch.randn(1,24,2,5); adj=torch.eye(2)
        real=model(hist,future,adj,x_future_raw=raw)
        counter=model(hist,future,adj,x_future_raw=cf)
        self.assertTrue(torch.equal(future,future_copy))
        self.assertEqual(tuple(real.shape),(1,12,2,1))
        loss=weighted_extreme_mae(real,torch.zeros_like(real),flag,5)+directional_counterfactual_loss(real,counter,flag)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

if __name__=='__main__':
    unittest.main(verbosity=2)


