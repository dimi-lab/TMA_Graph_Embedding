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

python main.py --config config/OVTMA_fov297_fastrp_none.yaml --outdir /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp_none &
python main.py --config config/OVTMA_fov297_fastrp_het_none.yaml --outdir /projects/wangc/m344313/OVTMA_project/output/fov297_fastrp_het_none &