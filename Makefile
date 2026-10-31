PYTHON ?= python

.PHONY: install test records summaries consistency tables figures paper

install:            ## package with analysis and test extras
	$(PYTHON) -m pip install -e ".[analysis,test]"

test:               ## unit tests (record- and data-level checks run when available)
	$(PYTHON) -m pytest -q

records:            ## download and verify the run records from Zenodo into records/
	bash scripts/fetch_records.sh

summaries:          ## all statistical summaries from the run records (about 10 minutes on a laptop)
	cd analysis && $(PYTHON) export_cells.py && $(PYTHON) estimators.py && $(PYTHON) gate_validation.py \
	  && $(PYTHON) determinants.py && $(PYTHON) breadth.py && $(PYTHON) original_study.py \
	  && $(PYTHON) hardware_consistency.py

consistency:        ## test-context consistency (needs the GIFT-Eval data)
	cd analysis && $(PYTHON) test_context_consistency.py

tables:             ## LaTeX tables from results/summaries
	cd analysis && $(PYTHON) tables.py

figures:            ## Figures 1-5 from results/
	$(PYTHON) analysis/figures/fig1_method.py && $(PYTHON) analysis/figures/figures.py

paper: summaries tables figures
