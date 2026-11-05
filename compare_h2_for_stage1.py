"""
Pick h2 from source summary statistics only: sum the final ELBO over the 5 sources.
"""
import sys
import pandas as pd
from harness import Dataset
from simulate.custom_viprs import custom_viprs, elbo

SCENARIOS = sys.argv[1:] or ["shared_effects", "ancestry_dependent", "locus_varying", "nonlinear_link"]
ANCESTRIES = ["EUR", "AFR", "EAS", "SAS", "AMR"]
H2S = [0.1, 0.15, 0.2, 0.3, 0.4, 0.6]

rows = []
for s in SCENARIOS:
    for r in range(10):
        d = Dataset(f"data/{s}", r)
        for h2 in H2S:
            total = 0.0
            for anc in ANCESTRIES:
                ss = d.sumstats(anc)
                _, st = custom_viprs(ss["beta"], d.ld(anc), ss["n"],
                                     block_of_snp=ss["block_of_snp"], h2=h2, return_state=True)
                total += elbo(ss["beta"], ss["n"], st)
            rows.append(dict(scenario=s, replicate=r, h2=h2, elbo=total))
            print(s, r, h2, round(total, 1), flush=True)

df = pd.DataFrame(rows)
df.to_csv("outputs/stage1_h2_elbo.csv", index=False)
wide = df.pivot_table(index=["scenario", "replicate"], columns="h2", values="elbo")
print("\nMean summed ELBO:\n", wide.groupby("scenario").mean().round(1))
print("\nReplicates where each h2 has the highest ELBO:\n",
      wide.idxmax(axis=1).groupby("scenario").value_counts().unstack(fill_value=0))