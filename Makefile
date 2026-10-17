.PHONY: all prepare train preds tables figures clean

all: prepare train preds tables figures

prepare:
	pip install -r requirements.txt
	python src/audit_split.py

train:
	python src/b0c_train_all.py

preds:
	python src/b0d_baselines.py

tables:
	python -m src.make_tables
	python -m src.b0c_decode_analysis
	python -m src.eval_phase_a

figures:
	python -m src.b0c_plot_gate
	python -m src.export_paper

clean:
	rm -rf results/preds/*.npz
	rm -rf results/tables/*.csv
	rm -rf results/tables/*.md
	rm -rf paper/figs/*
	rm -rf paper/tables/*
