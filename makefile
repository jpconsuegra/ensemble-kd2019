.PHONY: clean
clean:
	git clean -fxd

SCENARIO = 1-main
GOLD_SCENARIO = 1-main

.PHONY: do
do:
	python -m scripts.ensemble.cli \
		default/reference/submissions/ default/reference/gold/ 2 $(GOLD_SCENARIO) \
		default/target/submissions/ default/target/gold/ 2 $(SCENARIO) \
		default/output/output.txt
