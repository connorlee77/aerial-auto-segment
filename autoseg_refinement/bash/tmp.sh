taskset --cpu-list 100-120 bash bash/refine_classical.sh /home/connor/repos/aerial-auto-segment/autoseg_classical_segmentation/felzenszwalb ./classical_outputs_v2/common/felzenszwalb default

taskset --cpu-list 100-120 bash bash/refine_classical.sh /home/connor/repos/aerial-auto-segment/autoseg_classical_segmentation/felzenszwalb ./classical_outputs_v2/more_common/felzenszwalb more

taskset --cpu-list 100-120 bash bash/refine_classical.sh /home/connor/repos/aerial-auto-segment/autoseg_classical_segmentation/felzenszwalb ./classical_outputs_v2/most_common/felzenszwalb most

taskset --cpu-list 100-120 bash bash/refine_classical.sh /home/connor/repos/aerial-auto-segment/autoseg_classical_segmentation/slic ./classical_outputs_v2/common/slic default

taskset --cpu-list 100-120 bash bash/refine_classical.sh /home/connor/repos/aerial-auto-segment/autoseg_classical_segmentation/slic ./classical_outputs_v2/more_common/slic more

taskset --cpu-list 100-120 bash bash/refine_classical.sh /home/connor/repos/aerial-auto-segment/autoseg_classical_segmentation/slic ./classical_outputs_v2/most_common/slic most
