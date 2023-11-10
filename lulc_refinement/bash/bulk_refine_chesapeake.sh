bash bash/refine_chesapeake.sh weighted_nll_unconstrained 195 7 200 1 1
bash bash/refine_chesapeake.sh weighted_nll_constrained 200 10 10 1 1

# bash bash/refine_chesapeake.sh boundary_unconstrained x x x x x
bash bash/refine_chesapeake.sh boundary_constrained 199 1 1 10 1

bash bash/refine_chesapeake.sh nll_unconstrained 200 6 196 1 1
bash bash/refine_chesapeake.sh nll_constrained 200 10 10 1 1

bash bash/refine_chesapeake.sh nll_boundary_constrained 194 9 1 1 1
