python logs_to_csv.py --out analysis/fov297_fastrp_het.csv /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp_het/logs/run_20251007_170631.log

python logs_to_csv.py --out analysis/fov297_fastrp.csv /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp/logs/run_20251007_170557.log

python logs_to_csv.py --out analysis/fov216_fastrp.csv /projects/wangc/m344313/OVTMA_project/output/fov216_fastrp/logs/run_20251007_175450.log

python logs_to_csv.py --out analysis/fov216_fastrp_het.csv /projects/wangc/m344313/OVTMA_project/output/fov216_fastrp_het/logs/run_20251007_175552.log

python analyze_from_csv.py --outdir analysis/plots/fov216_overall analysis/fov216_fastrp.csv analysis/fov216_fastrp_het.csv
python analyze_from_csv.py --outdir analysis/plots/fov297_overall analysis/fov297_fastrp_het.csv analysis/fov297_fastrp.csv
python analyze_from_csv.py --outdir analysis/plots/fov297_fastrp_het analysis/fov297_fastrp_het.csv
python analyze_from_csv.py --outdir analysis/plots/fov297_fastrp analysis/fov297_fastrp.csv
python analyze_from_csv.py --outdir analysis/plots/fov216_fastrp analysis/fov216_fastrp.csv
python analyze_from_csv.py --outdir analysis/plots/fov216_fastrp_het analysis/fov216_fastrp_het.csv


python logs_to_csv.py --out /projects/wangc/m344313/OVTMA_project/analysis/demo.csv /projects/wangc/m344313/OVTMA_project/output/demo/logs/run_20251009_152517.log
python analyze_from_csv.py --outdir /projects/wangc/m344313/OVTMA_project/analysis/plots/demo/accuracy /projects/wangc/m344313/OVTMA_project/analysis/demo.csv --score-col accuracy
python analyze_from_csv.py --outdir /projects/wangc/m344313/OVTMA_project/analysis/plots/demo/average_acc /projects/wangc/m344313/OVTMA_project/analysis/demo.csv --score-col average_acc
python analyze_from_csv.py --outdir /projects/wangc/m344313/OVTMA_project/analysis/plots/demo/roc_auc /projects/wangc/m344313/OVTMA_project/analysis/demo.csv --score-col roc_auc


CUDA_VISIBLE_DEVICES=0 python main.py --config config/OVTMA_fov297_fastrp_none.yaml --outdir /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp_none --n-jobs 1 &
CUDA_VISIBLE_DEVICES=1 python main.py --config config/OVTMA_fov297_fastrp_het_none.yaml --outdir /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp_het_none --n-jobs 1 &
CUDA_VISIBLE_DEVICES=2 python main.py --config config/OVTMA_fov216_fastrp_none.yaml --outdir /projects/wangc/m344313/OVTMA_project/output/fov216_fastrp_none --n-jobs 1 &
CUDA_VISIBLE_DEVICES=3 python main.py --config config/OVTMA_fov216_fastrp_het_none.yaml --outdir /projects/wangc/m344313/OVTMA_project/output/fov216_fastrp_het_none --n-jobs 1 &

python logs_to_csv.py --out /projects/wangc/m344313/OVTMA_project/analysis/fov297_fastrp_none.csv /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp_none/logs/run_20251009_212012.log
python logs_to_csv.py --out /projects/wangc/m344313/OVTMA_project/analysis/fov297_fastrp_het_none.csv /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp_het_none/logs/run_20251009_212012.log
python logs_to_csv.py --out /projects/wangc/m344313/OVTMA_project/analysis/fov216_fastrp_none.csv /projects/wangc/m344313/OVTMA_project/output/fov216_fastrp_none/logs/run_20251009_212012.log
python logs_to_csv.py --out /projects/wangc/m344313/OVTMA_project/analysis/fov216_fastrp_het_none.csv /projects/wangc/m344313/OVTMA_project/output/fov216_fastrp_het_none/logs/run_20251009_212015.log

python analyze_from_csv.py --outdir /projects/wangc/m344313/OVTMA_project/analysis/plots/fov297_fastrp_none/accuracy /projects/wangc/m344313/OVTMA_project/analysis/fov297_fastrp_none.csv --score-col accuracy
python analyze_from_csv.py --outdir /projects/wangc/m344313/OVTMA_project/analysis/plots/fov297_fastrp_het_none/accuracy /projects/wangc/m344313/OVTMA_project/analysis/fov297_fastrp_het_none.csv --score-col accuracy
python analyze_from_csv.py --outdir /projects/wangc/m344313/OVTMA_project/analysis/plots/fov216_fastrp_none/accuracy /projects/wangc/m344313/OVTMA_project/analysis/fov216_fastrp_none.csv --score-col accuracy
python analyze_from_csv.py --outdir /projects/wangc/m344313/OVTMA_project/analysis/plots/fov216_fastrp_het_none/accuracy /projects/wangc/m344313/OVTMA_project/analysis/fov216_fastrp_het_none.csv --score-col accuracy

python analyze_from_csv.py --outdir /projects/wangc/m344313/OVTMA_project/analysis/plots/fov297_fastrp_none/roc_auc /projects/wangc/m344313/OVTMA_project/analysis/fov297_fastrp_none.csv --score-col roc_auc
python analyze_from_csv.py --outdir /projects/wangc/m344313/OVTMA_project/analysis/plots/fov297_fastrp_het_none/roc_auc /projects/wangc/m344313/OVTMA_project/analysis/fov297_fastrp_het_none.csv --score-col roc_auc
python analyze_from_csv.py --outdir /projects/wangc/m344313/OVTMA_project/analysis/plots/fov216_fastrp_none/roc_auc /projects/wangc/m344313/OVTMA_project/analysis/fov216_fastrp_none.csv --score-col roc_auc
python analyze_from_csv.py --outdir /projects/wangc/m344313/OVTMA_project/analysis/plots/fov216_fastrp_het_none/roc_auc /projects/wangc/m344313/OVTMA_project/analysis/fov216_fastrp_het_none.csv --score-col roc_auc

# none,  concat_shuffle
python main.py --config config/OVTMA_fov297_fastrp_none_shuffle.yaml --outdir /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp_none_shuffle --n-jobs 10 &
python logs_to_csv.py --out /projects/wangc/m344313/OVTMA_project/analysis/fov297_fastrp_none_shuffle.csv /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp_none_shuffle/logs/run_20251010_095931.log
python analyze_from_csv.py --outdir /projects/wangc/m344313/OVTMA_project/analysis/plots/fov297_fastrp_none_Include_shuffle/roc_auc /projects/wangc/m344313/OVTMA_project/analysis/fov297_fastrp_none.csv /projects/wangc/m344313/OVTMA_project/analysis/fov297_fastrp_none_shuffle.csv --score-col roc_auc
python analyze_from_csv.py --outdir /projects/wangc/m344313/OVTMA_project/analysis/plots/fov297_fastrp_none_Include_shuffle/accuracy /projects/wangc/m344313/OVTMA_project/analysis/fov297_fastrp_none.csv /projects/wangc/m344313/OVTMA_project/analysis/fov297_fastrp_none_shuffle.csv --score-col accuracy