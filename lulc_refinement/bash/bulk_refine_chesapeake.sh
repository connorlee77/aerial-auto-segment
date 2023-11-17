# bash bash/refine_chesapeake.sh weighted_nll_constrained 200 10 10 1 1 naip
# bash bash/refine_chesapeake.sh nll_constrained 200 10 10 1 1 naip
# bash bash/refine_chesapeake.sh nll_boundary_constrained 194 9 1 1 1 naip
# bash bash/refine_chesapeake.sh boundary_constrained 199 1 1 10 1 naip

# NAIP only
bash bash/refine_chesapeake.sh nll_unconstrained 200 6 196 1 1 naip -1 -1
bash bash/refine_chesapeake.sh boundary_unconstrained 149.02 1.13 33.28 47.1 2.62 naip -1 -1
bash bash/refine_chesapeake.sh weighted_nll_unconstrained 195 7 200 1 1 naip -1 -1

# Boundary loss only
bash bash/refine_chesapeake.sh boundary_naip_ndvi 199.738 '33.748 0.379 0.429 10.356' 11.663 44.263 14.044 'naip naip-ndvi' -1 -1
bash bash/refine_chesapeake.sh boundary_naip_ndvi_surface 133.682 '16.995 29.267 2.082 44.686 0.145' 140.421 48.854 1.170 'naip naip-ndvi surface_height' -1 -1
bash bash/refine_chesapeake.sh boundary_naip_nir 193.619 '128.252 0.217 125.359 2.711' 61.451 47.412 0.141 'naip naip-nir' -1 -1
bash bash/refine_chesapeake.sh boundary_naip_surface 79.878 '90.522 0.171 3.261 128.568' 100.193 30.529 0.960 'naip surface_height' -1 -1

bash bash/refine_chesapeake.sh boundary_naip_nir_surface 155.996 '164.095 153.521 9.399 7.589' 70.715 47.651 1.640 'naip naip-nir surface_height' 2.089 169.727
bash bash/refine_chesapeake.sh boundary_naip_nir_ndvi_surface 191.949 '142.961 79.844 5.528 8.860 38.080' 37.798 49.992 0.101 'naip naip-nir naip-ndvi surface_height' 135.102 189.066
