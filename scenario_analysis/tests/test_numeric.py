"""Extract named functions only: avoids original scripts' top-level filesystem effects."""
import ast, unittest
from pathlib import Path
try:
 import numpy as np
 import pandas as pd
except ImportError:
 np=pd=None
ROOT=Path(__file__).resolve().parents[1]
def functions(pattern,names,env):
 tree=ast.parse(next((ROOT/"scripts").glob(pattern)).read_text(encoding="utf-8"))
 tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 exec(compile(tree,"isolated-functions","exec"),env)
 return env
@unittest.skipIf(np is None or pd is None,"numpy/pandas unavailable")
class Numeric(unittest.TestCase):
 def test_diagonal_preserves_offdiagonal(self):
  e=functions("S5*.py",["fill_diag_transport"],{"np":np,"pd":pd})
  m=pd.DataFrame([[99.,4.],[8.,88.]])
  result=e["fill_diag_transport"](m)
  np.testing.assert_array_equal(result.values,[[3.,4.],[8.,3.]])
  self.assertEqual(m.iloc[0,0],99.)
 def test_stock_bounds(self):
  e=functions("平均*.py",["make_inventory_scenario"],{"np":np,"pd":pd})
  result=e["make_inventory_scenario"](pd.DataFrame({"stock_days_for_model":[1.,30.]}),2.,1.,180.,"high")
  np.testing.assert_array_equal(result["stock_days_for_model"],[2.,60.])
 def test_sector_aggregation_conserves_inputs(self):
  e=functions("S9*.py",["get_sector_index_for_activity","get_region_index_for_activity","aggregate_inputs_by_supplier_sector"],{"np":np,"RR":15,"NN":19,"RRNN":285})
  m=np.arange(285*285,dtype=float).reshape(285,285)
  a=e["aggregate_inputs_by_supplier_sector"](m)
  np.testing.assert_array_equal(a.sum(axis=0),m.sum(axis=0))
  np.testing.assert_array_equal(a[1],m[15:30].sum(axis=0))
  self.assertEqual(e["get_region_index_for_activity"](16),1)
 def test_order_reorders_both_Z_axes(self):
  names=["build_sector_major_raw_indices","reorder_arrays_to_sector_major"]
  e=functions("统一*.py",names,{"np":np,"RR":15,"NN":19,"RRNN":285})
  idx=e["build_sector_major_raw_indices"]("region-major")
  np.testing.assert_array_equal(idx[:4],[0,19,38,57])
  z=np.arange(285**2).reshape(285,285); f=np.arange(285*15).reshape(285,15); v=np.arange(285).reshape(1,285)
  zz,ff,vv,xx=e["reorder_arrays_to_sector_major"](z,f,v,v,idx)
  self.assertEqual(zz[1,2],z[19,38]); np.testing.assert_array_equal(ff[1],f[19]); self.assertEqual(vv[0,1],v[0,19])
  inverse=np.argsort(idx); np.testing.assert_array_equal(zz[np.ix_(inverse,inverse)],z)
  np.testing.assert_array_equal(e["build_sector_major_raw_indices"]("sector-major"),np.arange(285))
 def test_s1_first_forecast_explains_december_gap(self):
  e=functions("S1生成*.py",["build_base_times","build_future_time_matrix","build_time_mask_np"],{"np":np,"pd":pd,"Tin":24,"Tout":12})
  times=e["build_base_times"]("201912",3)
  self.assertEqual(str(times[0]),"2019-12-02 00:00:00")
  matrix=e["build_future_time_matrix"](times,3,12)
  mask=e["build_time_mask_np"](matrix,"2019-12-01 00:00:00","2019-12-01 23:00:00")
  self.assertFalse(mask.any())
 def test_eta_clipping(self):
  e=functions("S8*.py",["compute_effective_factor"],{"np":np,"ETA":3.5,"USE_TRANSPORT_FACTOR_FLOOR":False})
  np.testing.assert_allclose(e["compute_effective_factor"](np.array([-1.,0.,.5,1.,2.])),[0.,0.,.5**3.5,1.,1.])
if __name__=="__main__": unittest.main()
