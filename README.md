# Dissertation Data Science

## Important data-definition notice

The public archive [`steam_reviews_model_base_20260722.zip`](https://github.com/zhuochengwang0-coder/dissertation-data-science/blob/main/steam_reviews_model_base_20260722.zip) contains 38,757 rows. Its `high_playtime` column is an early variable defined as playtime at the time of review strictly above the game-specific median, producing an approximately 50% high label.

This archived field was not used as an input or target in NLP training, was not used to stratify the 1,000-review development sample or the 300-review evaluation sample, and was not used in the final logistic regression. For the final analysis, the 1,300 labelled reviews were excluded and `high_playtime` was recalculated within each game using the 75th percentile (Q75). The final analytical sample contained 9,366 high-playtime and 28,091 lower-playtime reviews among 37,457 records (25.00% and 75.00%, respectively).

The original `high_playtime` field name is retained in the public archive only to preserve fidelity to the database export and must not be interpreted as the final outcome definition.
