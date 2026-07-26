RESULTS GUIDE
=============

Primary analysis:
  02_primary_q75_thresholds.csv
      Exact within-game Q75 thresholds used to create high_playtime.
  09_main_model_six_factors.csv
      Main dissertation table: beta, HC3 SE, OR, 95% CI, raw p-value,
      BH-FDR p-value, and average marginal effect for the six factors.
  12_main_model_or_forest.png
      Forest plot of the six adjusted odds ratios.

Diagnostics:
  05_simple_separation_cells.csv
  06_factor_correlations.csv
  11_main_model_vif.csv
  13_influence_top50_by_cooks_d.csv
  14_influence_summary.csv
  21_joint_and_model_comparison_tests.csv
  22_model_fit_statistics.csv
  23_fit_warnings.csv

Robustness / exploratory:
  16_sensitivity_q70_q75_q80_factors.csv
  17_sensitivity_leave_one_game_out.csv
  18_exploratory_game_specific_models.csv
  19_sensitivity_specification_and_influence.csv
  20_exploratory_game_interaction_terms.csv

Interpretation:
  OR > 1 means higher odds of being in the within-game high-playtime group.
  OR < 1 means lower odds.
  This is an association, not a causal effect. A factor value of 1 means the
  review mentions the factor; it does not mean the mention is positive.

Primary formula:
  high_playtime ~ combat + challenge + progression + exploration + narrative + immersion + log_review_words + C(game, Treatment(reference='Cyberpunk 2077')) + steam_purchase + received_for_free
