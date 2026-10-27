"""Read-only MRIO/date/stock audit. Reports coverage gaps without changing windows."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
WINDOWS={"201903":("2019-03-23","2019-03-27"),"201908":("2019-08-10","2019-08-14"),"201911":("2019-11-19","2019-11-22"),"201912":("2019-12-01","2019-12-08")}
def audit(mrio=None,s7_root=None,stock=None):
 result={}
 if mrio:
  d=pd.read_csv(mrio,header=None)
  columns=list(zip(d.iloc[0,2:287].astype(str).str.strip(),d.iloc[1,2:287].astype(str).str.strip()))
  rows=list(zip(d.iloc[2:287,0].astype(str).str.strip(),d.iloc[2:287,1].astype(str).str.strip()))
  regions=list(dict.fromkeys(r for r,s in columns)); sectors=list(dict.fromkeys(s for r,s in columns))
  region_major=[(r,s) for r in regions for s in sectors]; sector_major=[(r,s) for s in sectors for r in regions]
  order="region-major" if columns==region_major else ("sector-major" if columns==sector_major else "unknown")
  result["mrio"]={"activity_count":len(columns),"region_count":len(regions),"sector_count":len(sectors),"all_285_column_labels_checked":len(columns)==285 and len(regions)==15 and len(sectors)==19 and order!="unknown","row_column_labels_equal":rows==columns,"order":order}
 if s7_root:
  result["s7"]={}
  for intensity in ["I1","I2","I3"]:
   root=Path(s7_root)/"H0"/intensity
   t=pd.read_csv(root/"time_index_impact_days.csv"); dates=pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d").tolist()
   expected=[str(x.date()) for start,end in WINDOWS.values() for x in pd.date_range(start,end)]
   a=np.load(root/"regional_transport_factor_TT.npy")
   result["s7"][intensity]={"factor_shape":list(a.shape),"date_rows":len(dates),"duplicate_dates":len(dates)-len(set(dates)),"missing_official_dates":sorted(set(expected)-set(dates)),"unexpected_dates":sorted(set(dates)-set(expected)),"finite_factors":bool(np.isfinite(a).all())}
 if stock:
  d=pd.read_csv(stock); a=pd.to_numeric(d["stock_days_for_model"],errors="coerce")
  result["stock"]={"rows":len(d),"finite_positive":bool(np.isfinite(a).all() and (a>0).all()),"min_days":float(a.min()),"max_days":float(a.max()),"mean_days":float(a.mean())}
 return result
if __name__=="__main__":
 parser=argparse.ArgumentParser(); parser.add_argument("--mrio"); parser.add_argument("--s7-root"); parser.add_argument("--stock"); args=parser.parse_args()
 print(json.dumps(audit(args.mrio,args.s7_root,args.stock),ensure_ascii=False,indent=2))
