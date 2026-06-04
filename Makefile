PY ?= python3
DATA ?= data

.PHONY: reproduce stats clean

reproduce: stats
	@echo ">> Reproduced all tables from $(DATA)/"

stats:
	$(PY) analysis/analysis.py
	$(PY) analysis/stats_utils.py --data-dir $(DATA)

clean:
	rm -f analysis_report.txt analysis_flat.csv bootstrap_cis.csv per_claim_stats.csv \
	      residuals_for_review.csv regression_results.txt
