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

python eval_refinements.py \
--refined-label-data-dir lulc_refine_outputs/boundary_naip_surface \
--ground-truth-data-dir /media/hdd2/data/chesapeake_bay_lulc \
--base-img-src naip surface_height | tee lulc_refine_outputs/boundary_naip_surface.txt


python eval_refinements.py \
--refined-label-data-dir lulc_refine_outputs/boundary_naip_ndvi \
--ground-truth-data-dir /media/hdd2/data/chesapeake_bay_lulc \
--base-img-src naip naip-ndvi | tee lulc_refine_outputs/boundary_naip_ndvi.txt

python eval_refinements.py \
--refined-label-data-dir lulc_refine_outputs/boundary_naip_ndvi_surface \
--ground-truth-data-dir /media/hdd2/data/chesapeake_bay_lulc \
--base-img-src naip naip-ndvi surface_height | tee lulc_refine_outputs/boundary_naip_ndvi_surface.txt

python eval_refinements.py \
--refined-label-data-dir lulc_refine_outputs/boundary_naip_nir \
--ground-truth-data-dir /media/hdd2/data/chesapeake_bay_lulc \
--base-img-src naip naip-nir | tee lulc_refine_outputs/boundary_naip_nir.txt

python eval_refinements.py \
--refined-label-data-dir lulc_refine_outputs/boundary_naip_nir_surface \
--ground-truth-data-dir /media/hdd2/data/chesapeake_bay_lulc \
--base-img-src naip naip-nir surface_height | tee lulc_refine_outputs/boundary_naip_nir_surface.txt

python eval_refinements.py \
--refined-label-data-dir lulc_refine_outputs/boundary_naip_nir_ndvi_surface \
--ground-truth-data-dir /media/hdd2/data/chesapeake_bay_lulc \
--base-img-src naip naip-nir naip-ndvi surface_height | tee lulc_refine_outputs/boundary_naip_nir_ndvi_surface.txt