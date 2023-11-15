python eval_refinements.py \
--ground-truth-data-dir /media/hdd2/data/chesapeake_bay_lulc \
--compute-dynamicworld-baseline | tee lulc_refine_outputs/dynamicworld_baseline.txt

python eval_refinements.py \
--refined-label-data-dir lulc_refine_outputs/boundary_unconstrained \
--ground-truth-data-dir /media/hdd2/data/chesapeake_bay_lulc | tee lulc_refine_outputs/boundary_unconstrained.txt

python eval_refinements.py \
--refined-label-data-dir lulc_refine_outputs/nll_unconstrained \
--ground-truth-data-dir /media/hdd2/data/chesapeake_bay_lulc | tee lulc_refine_outputs/nll_unconstrained.txt

python eval_refinements.py \
--refined-label-data-dir lulc_refine_outputs/weighted_nll_unconstrained \
--ground-truth-data-dir /media/hdd2/data/chesapeake_bay_lulc | tee lulc_refine_outputs/weighted_nll_unconstrained.txt
