.PHONY: clean
clean:
	git clean -fxd

SCENARIO = 1-main
GOLD_SCENARIO = 1-main

CNAME = test
GOLD_CNAME = ''

EXCLUDE = True

.PHONY: do
do:
	python -m scripts.ensemble.cli \
		default/reference/submissions/ default/reference/gold/ 2 $(GOLD_SCENARIO) $(GOLD_CNAME) \
		default/target/submissions/ default/target/gold/ 2 $(SCENARIO) $(CNAME) \
		default/output/output.txt \
		--exclude_gold_annotated=$(EXCLUDE)
