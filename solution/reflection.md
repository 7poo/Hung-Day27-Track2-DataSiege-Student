# Reflection

**Which fault types were hardest to catch, and why?**

The hardest cases were subtle volume/distribution shifts and near-threshold runtime or corpus-age anomalies. The published limits are calibrated near three standard deviations, so they are very precise but miss private's deliberately subtle faults. Contract schema/type violations were the most deterministic because the toolkit exposes explicit violations, while lineage structure could be checked directly when declarations were available.

**What would you change about your cost/coverage tradeoff, if you had another pass?**

I use one targeted toolkit call per event, which stays within private's budget. The scoring weights make a false negative about 4.5 times more expensive than a false positive on this stream, so the best private trade-off is intentionally recall-heavy. I tightened the high-side row-count and amount-mean bands, runtime and corpus-age limits, while keeping adaptive checks for variance, amount mean, staleness, and embedding drift. Ablation retained only rules whose recovered faults outweighed their false alarms. The result moved private recall from 83.33% to 100% and the score from 38.17 to 42.81, at the cost of a higher 23.97% false-positive rate. With another unseen stream, I would replace the small-sample z-score with a robust rolling median/MAD and calibrate thresholds on a separate validation seed.
