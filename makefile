.PHONY: clean
clean:
	git clean -fxd

.PHONY: do
do:
	python -m scripts.ensemble.cli \
		default/reference/submissions/ default/reference/gold/ 2 1-main \
		default/target/submissions/ default/target/gold/ 2 1-main \
		default/output/output.txt
